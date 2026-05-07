"""
generator.py
LLM answer generation via OpenAI API (gpt-4o-mini).
Measures generation latency separately from retrieval latency.
"""

import os
import time
from typing import Optional

from openai import OpenAI


GENERATION_MODEL = "gpt-4o-mini"
MAX_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the "
    "provided context. Give a SHORT answer: one word, a name, or a brief "
    "phrase. Do not write full sentences. "
    "If the context does not contain enough information, say 'I don't know'."
)


class Generator:
    def __init__(self, api_key: Optional[str] = None, model: str = GENERATION_MODEL):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, question: str, chunks: list[dict]) -> dict:
        """
        Generate an answer given a question and retrieved chunks.
        Returns:
            {
                "answer": str,
                "latency_ms": float,
            }
        """
        context = "\n\n".join(
            f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
        )
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "latency_ms": latency_ms}
