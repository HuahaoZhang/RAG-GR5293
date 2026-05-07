"""
data_prep.py
Download HotpotQA (first 1000 samples), chunk context passages,
and save to data/ for later indexing.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm


def load_hotpotqa(n_samples: int = 1000) -> list[dict]:
    """Load n_samples from HotpotQA validation split."""
    print(f"Loading HotpotQA ({n_samples} samples)...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    samples = []
    for i, item in enumerate(tqdm(ds, total=n_samples)):
        if i >= n_samples:
            break
        # Flatten context: list of (title, sentences) pairs
        passages = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            text = " ".join(sentences)
            passages.append({"title": title, "text": text})

        samples.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "passages": passages,          # gold + distractor passages
            "supporting_facts": item["supporting_facts"],
        })
    return samples


def chunk_passages(samples: list[dict], chunk_size: int = 200) -> list[dict]:
    """
    Split each passage into fixed-size word chunks.
    Returns flat list of chunk dicts for indexing.
    """
    chunks = []
    for sample in tqdm(samples, desc="Chunking passages"):
        for passage in sample["passages"]:
            words = passage["text"].split()
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i: i + chunk_size])
                chunks.append({
                    "chunk_id": f"{sample['id']}_{passage['title']}_{i}",
                    "source_id": sample["id"],
                    "title": passage["title"],
                    "text": chunk_text,
                })
    return chunks


def main():
    os.makedirs("data", exist_ok=True)

    # 1. Load QA pairs
    samples = load_hotpotqa(n_samples=1000)
    qa_path = "data/qa_pairs.json"
    with open(qa_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"Saved {len(samples)} QA pairs → {qa_path}")

    # 2. Chunk passages
    chunks = chunk_passages(samples)
    chunks_path = "data/chunks.json"
    with open(chunks_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks → {chunks_path}")


if __name__ == "__main__":
    main()
