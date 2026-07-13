"""Dense and sparse embedders for the RAG pipeline."""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from rag_pipeline.config import DenseEmbeddingConfig, SparseEmbeddingConfig


class DenseEmbedder:
    """Dense embedder backed by OpenRouter's embedding endpoint."""

    def __init__(self, config: DenseEmbeddingConfig | None = None) -> None:
        self.config = config or DenseEmbeddingConfig()
        self._client = httpx.Client(
            base_url=self.config.api_base,
            timeout=self.config.timeout_seconds,
            headers={
                "Authorization": f"Bearer {self._api_key()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rag-pipeline.local",
                "X-Title": "RAG Pipeline",
            },
        )

    def _api_key(self) -> str:
        key = os.getenv(self.config.api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing API key: set the {self.config.api_key_env} environment variable"
            )
        return key

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors."""
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_results = self._embed_batch(batch)
            results.extend(batch_results)
        return results

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_exception: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(
                    "/embeddings",
                    json={
                        "model": self.config.model_name,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except Exception as exc:  # pragma: no cover - retry logic
                last_exception = exc
                if attempt < self.config.max_retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(
            f"Dense embedding failed after {self.config.max_retries} retries"
        ) from last_exception

    def close(self) -> None:
        self._client.close()


class SparseEmbedder:
    """Classic BM25 sparse embedder. No external model download required.

    The embedder builds a term vocabulary and IDF statistics from a corpus via
    :meth:`fit`, persists them to disk, and then encodes new texts into sparse
    BM25 vectors. This makes it fully offline and suitable for environments
    without HuggingFace connectivity.
    """

    _TOKEN_PATTERN = re.compile(r"[\w\u00C0-\u024F\u1EA0-\u1EFF]+", re.UNICODE)

    def __init__(self, config: SparseEmbeddingConfig | None = None) -> None:
        self.config = config or SparseEmbeddingConfig()
        self.vocab: dict[str, int] = {}
        self.idf: dict[int, float] = {}
        self.doc_freq: dict[int, int] = {}
        self.total_docs: int = 0
        self.avg_len: float = self.config.avg_len
        self._load_vocab_if_exists()

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in self._TOKEN_PATTERN.findall(text) if token.strip()]

    # ------------------------------------------------------------------
    # Vocabulary / IDF persistence
    # ------------------------------------------------------------------
    def _vocab_path(self) -> Path:
        return Path(self.config.vocab_path)

    def _load_vocab_if_exists(self) -> None:
        path = self._vocab_path()
        if path.exists():
            self.load_vocab(str(path))

    def load_vocab(self, path: str | Path) -> None:
        """Load a previously saved vocabulary and IDF statistics."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.vocab = data["vocab"]
        self.idf = {int(k): v for k, v in data["idf"].items()}
        self.doc_freq = {int(k): v for k, v in data["doc_freq"].items()}
        self.total_docs = data["total_docs"]
        self.avg_len = data["avg_len"]

    def save_vocab(self, path: str | Path | None = None) -> None:
        """Persist vocabulary and IDF statistics to disk."""
        path = path or self._vocab_path()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "vocab": self.vocab,
                    "idf": self.idf,
                    "doc_freq": self.doc_freq,
                    "total_docs": self.total_docs,
                    "avg_len": self.avg_len,
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, texts: list[str]) -> "SparseEmbedder":
        """Build vocabulary and IDF statistics from a corpus."""
        if not texts:
            return self

        tokenized_docs = [self._tokenize(text) for text in texts]
        self.total_docs = len(tokenized_docs)
        self.avg_len = sum(len(tokens) for tokens in tokenized_docs) / self.total_docs

        # Build vocabulary and document frequencies
        self.vocab = {}
        self.doc_freq = {}
        term_doc_counts: dict[str, int] = {}

        for tokens in tokenized_docs:
            seen = set(tokens)
            for term in seen:
                term_doc_counts[term] = term_doc_counts.get(term, 0) + 1

        for idx, (term, df) in enumerate(sorted(term_doc_counts.items())):
            self.vocab[term] = idx
            self.doc_freq[idx] = df

        # Compute IDF using standard BM25 IDF formula
        self.idf = {}
        for idx, df in self.doc_freq.items():
            self.idf[idx] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

        self.save_vocab()
        return self

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode texts into BM25 sparse vectors."""
        if not texts:
            return []

        results: list[dict[int, float]] = []
        for text in texts:
            tokens = self._tokenize(text)
            if not tokens or not self.vocab:
                results.append({})
                continue

            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            doc_len = len(tokens)
            sparse_vector: dict[int, float] = {}
            for term, tf in term_counts.items():
                if term not in self.vocab:
                    continue
                idx = self.vocab[term]
                idf = self.idf.get(idx, 0.0)
                # Standard BM25 formula
                numerator = tf * (self.config.k + 1.0)
                denominator = tf + self.config.k * (
                    1.0 - self.config.b + self.config.b * (doc_len / max(self.avg_len, 1.0))
                )
                sparse_vector[idx] = idf * (numerator / denominator)

            results.append(sparse_vector)
        return results

    def __enter__(self) -> "SparseEmbedder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
