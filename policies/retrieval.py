"""TF-IDF + cosine-similarity retrieval over the policy chunks.

Deliberately not a neural embedding model, and deliberately not a vector
database -- same reasoning as Section F's episodic memory, applied one
notch further this time since this corpus genuinely is text, not
structured records: a neural embedder needs a model download (no
guaranteed network access here) and produces a similarity score nobody can
verify by hand; TF-IDF's score is just "how much do these two documents
share important words," fully inspectable. And at 7 chunks, brute-force
cosine similarity is instant -- a vector database's whole reason to exist
(approximate nearest-neighbor search over millions of vectors) buys
nothing here. Section F's memory store earns Chroma because incidents
accumulate indefinitely across the platform's lifetime; this corpus is a
handful of static paragraphs rebuilt fresh every run.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from policies.chunking import load_chunks

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class TfidfIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.token_lists = [tokenize(c["text"]) for c in chunks]
        self.vocabulary = sorted({t for tokens in self.token_lists for t in tokens})
        self.vocab_index = {t: i for i, t in enumerate(self.vocabulary)}
        self.idf = self._compute_idf()
        self.chunk_vectors = np.array([self._vectorize(tokens) for tokens in self.token_lists])

    def _compute_idf(self) -> np.ndarray:
        n_docs = len(self.token_lists)
        doc_freq = np.zeros(len(self.vocabulary))
        for tokens in self.token_lists:
            for t in set(tokens):
                doc_freq[self.vocab_index[t]] += 1
        # smoothed idf: log(N / (1 + df)) + 1, the standard scikit-learn-style
        # smoothing so a term appearing in every chunk still gets a small
        # positive weight instead of collapsing to zero.
        return np.log(n_docs / (1.0 + doc_freq)) + 1.0

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary))
        if not tokens:
            return vector
        counts = Counter(tokens)
        for term, count in counts.items():
            if term in self.vocab_index:
                tf = count / len(tokens)
                vector[self.vocab_index[term]] = tf * self.idf[self.vocab_index[term]]
        return vector

    def query(self, text: str, k: int = 3) -> list[dict]:
        query_vector = self._vectorize(tokenize(text))
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []  # query shares no vocabulary with the policy doc at all

        results = []
        for chunk, chunk_vector in zip(self.chunks, self.chunk_vectors):
            chunk_norm = np.linalg.norm(chunk_vector)
            if chunk_norm == 0:
                continue
            similarity = float(np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm))
            results.append({**chunk, "similarity": similarity})

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results[:k]


def build_index() -> TfidfIndex:
    return TfidfIndex(load_chunks())
