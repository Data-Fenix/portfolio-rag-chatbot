"""Retrieval backends behind one interface.

* ``BM25Retriever`` - lexical, pure Python, zero external dependencies. Fast
  to build, deterministic, great for tests and for the exact-term questions a
  portfolio receives ("SageMaker", "Snowflake").
* ``ChromaRetriever`` - dense embeddings via ChromaDB's default local
  embedding function. Better for paraphrased questions.
* ``HybridRetriever`` - reciprocal rank fusion of both.

Select with ``RETRIEVER_BACKEND=bm25|chroma|hybrid``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from .config import settings
from .knowledge import Chunk, load_chunks

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "what", "which", "who", "how", "did", "does", "do", "at", "by", "it", "his", "he", "that", "this",
    "lahiru", "lahirus", "you", "your", "me", "i", "about", "tell",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]: ...


class BM25Retriever:
    """Okapi BM25 over chunk text, with title and section terms boosted."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(f"{c.title} {c.section} {c.section} {c.text}") for c in chunks]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, len(self.docs))
        self.tf = [Counter(d) for d in self.docs]
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, q_tokens: list[str], i: int) -> float:
        s = 0.0
        tf, dl = self.tf[i], self.doc_len[i]
        for t in q_tokens:
            if t not in tf:
                continue
            f = tf[t]
            s += self.idf[t] * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = tokenize(query)
        if not q:
            return []
        scored = [(self.chunks[i], self.score(q, i)) for i in range(len(self.chunks))]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class ChromaRetriever:
    """Dense retrieval with ChromaDB (optional dependency)."""

    def __init__(self, chunks: list[Chunk]) -> None:
        import chromadb  # imported lazily so the default path has no heavy deps

        self.chunks = {c.id: c for c in chunks}
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.col = client.get_or_create_collection("portfolio", metadata={"hnsw:space": "cosine"})
        existing = set(self.col.get(include=[])["ids"])
        new = [c for c in chunks if c.id not in existing]
        if new:
            self.col.upsert(
                ids=[c.id for c in new],
                documents=[f"{c.title} / {c.section}\n{c.text}" for c in new],
                metadatas=[{"source_id": c.source_id} for c in new],
            )

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        res = self.col.query(query_texts=[query], n_results=min(k, len(self.chunks)))
        ids, dists = res["ids"][0], res["distances"][0]
        return [(self.chunks[i], 1 - d) for i, d in zip(ids, dists, strict=True) if i in self.chunks]


class HybridRetriever:
    """Reciprocal rank fusion of lexical and dense results."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60) -> None:
        self.bm25 = BM25Retriever(chunks)
        self.dense = ChromaRetriever(chunks)
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for results in (self.bm25.search(query, k * 2), self.dense.search(query, k * 2)):
            for rank, (chunk, _) in enumerate(results):
                by_id[chunk.id] = chunk
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(by_id[i], s) for i, s in ranked]


def build_retriever(backend: str | None = None, chunks: list[Chunk] | None = None) -> Retriever:
    backend = backend or settings.retriever_backend
    chunks = chunks if chunks is not None else load_chunks()
    if backend == "bm25":
        return BM25Retriever(chunks)
    if backend == "chroma":
        return ChromaRetriever(chunks)
    if backend == "hybrid":
        return HybridRetriever(chunks)
    raise ValueError(f"Unknown RETRIEVER_BACKEND={backend!r}; use bm25, chroma or hybrid")
