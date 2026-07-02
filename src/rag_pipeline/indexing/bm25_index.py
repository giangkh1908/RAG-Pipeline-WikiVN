"""BM25 index for keyword-based retrieval."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class BM25Index:
    """BM25 index with Vietnamese tokenization.

    Supports:
    - underthesea: Vietnamese word segmentation (recommended)
    - pyvi: Vietnamese tokenizer (faster, less accurate)
    - simple: lowercase split (fallback)
    """

    index_path: Path
    tokenizer_name: str = "underthesea"
    _index: Any = field(default=None, init=False, repr=False)
    _doc_ids: list[str] = field(default_factory=list, init=False)
    _tokenized_corpus: list[list[str]] = field(default_factory=list, init=False)

    def build(self, documents: Iterable[tuple[str, str]]) -> None:
        """Build index from (doc_id, text) pairs."""
        self._doc_ids = []
        self._tokenized_corpus = []

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            self._doc_ids.append(doc_id)
            self._tokenized_corpus.append(tokens)

        if self._tokenized_corpus:
            from rank_bm25 import BM25Okapi
            self._index = BM25Okapi(self._tokenized_corpus)
        else:
            self._index = None

        self._save()

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Search and return (chunk_id, score) pairs sorted by score desc."""
        if self._index is None:
            if not self._doc_ids:
                return []
            raise RuntimeError("BM25 index not loaded. Call build() or load() first.")

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        # Sort by score descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self._doc_ids[idx], float(score)) for idx, score in ranked[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize Vietnamese text."""
        if self.tokenizer_name == "underthesea":
            try:
                from underthesea import word_tokenize
                return word_tokenize(text, format="text").split()
            except ImportError:
                pass

        if self.tokenizer_name == "pyvi":
            try:
                from pyvi import ViTokenizer
                return ViTokenizer.tokenize(text).split()
            except ImportError:
                pass

        # Fallback: simple lowercase split
        return text.lower().split()

    def _save(self) -> None:
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "doc_ids": self._doc_ids,
                "corpus": self._tokenized_corpus,
                "tokenizer": self.tokenizer_name,
            }, f)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if not self.index_path.exists():
            return False

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self._doc_ids = data["doc_ids"]
        self._tokenized_corpus = data["corpus"]
        self.tokenizer_name = data.get("tokenizer", self.tokenizer_name)

        from rank_bm25 import BM25Okapi
        self._index = BM25Okapi(self._tokenized_corpus)
        return True

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def doc_count(self) -> int:
        return len(self._doc_ids)
