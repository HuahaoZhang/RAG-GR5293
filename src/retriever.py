"""
retriever.py
ChromaDB-backed dense retriever using OpenAI embeddings.
Supports variable top-k and returns retrieval latency.
"""

import json
import time
import os
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm


COLLECTION_NAME = "rag_chunks"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # ChromaDB upsert batch size


class DenseRetriever:
    def __init__(self, persist_dir: str = "data/chroma_db", api_key: Optional[str] = None):
        self.persist_dir = persist_dir
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBED_MODEL,
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_chunks(self, chunks_path: str = "data/chunks.json"):
        """Embed and upsert all chunks. Skips if already indexed."""
        existing = self.collection.count()
        with open(chunks_path) as f:
            chunks = json.load(f)

        if existing >= len(chunks):
            print(f"Collection already has {existing} chunks — skipping indexing.")
            return

        print(f"Indexing {len(chunks)} chunks into ChromaDB...")
        for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[i: i + BATCH_SIZE]
            self.collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[{"title": c["title"], "source_id": c["source_id"]} for c in batch],
            )
        print(f"Indexed {self.collection.count()} chunks.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> dict:
        """
        Retrieve top_k chunks for query.
        Returns:
            {
                "chunks": [{"text": ..., "title": ..., "score": ...}, ...],
                "latency_ms": float,
            }
        """
        t0 = time.perf_counter()
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "title": meta.get("title", ""),
                "score": 1 - dist,  # cosine similarity
            })

        return {"chunks": chunks, "latency_ms": latency_ms}

    # ------------------------------------------------------------------
    # Adaptive retrieval helpers
    # ------------------------------------------------------------------

    def query_complexity_score(self, query: str) -> float:
        """
        Estimate query complexity via mean similarity to top-20 chunks.
        High mean similarity → query is specific → simple (low score).
        Low mean similarity → query is vague/hard → complex (high score).
        Returns score in [0, 1] where 1 = most complex.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(20, self.collection.count()),
            include=["distances"],
        )
        distances = results["distances"][0]
        mean_sim = 1 - (sum(distances) / len(distances))
        # Invert: low similarity → high complexity
        return 1 - mean_sim
