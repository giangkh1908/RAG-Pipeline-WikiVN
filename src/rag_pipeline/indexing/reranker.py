"""Re-rankers for search results."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from rag_pipeline.indexing.vector_store import SearchResult


@dataclass(slots=True)
class CohereReranker:
    """Re-ranker using Cohere Rerank API.

    Free tier: 100 search units/month
    Model: rerank-v3.5 (multilingual, supports Vietnamese)
    Docs: https://docs.cohere.com/reference/rerank
    """

    model_name: str = "rerank-v3.5"
    api_base: str = "https://api.cohere.com/v2/rerank"
    api_key_env: str = "COHERE_API_KEY"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 2.0

    def rerank(
        self,
        query: str,
        passages: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rank passages using Cohere Rerank API.

        Args:
            query: The search query
            passages: Passages to re-rank
            top_k: Number of top results to return

        Returns:
            Re-ranked passages sorted by relevance score desc
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "The `httpx` package is required for Cohere reranker. "
                "Install with `pip install httpx`."
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing environment variable `{self.api_key_env}` for Cohere access."
            )

        if not passages:
            return []

        # Prepare documents
        documents = [p.text[:1000] for p in passages]  # Truncate long docs

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                response = client.post(self.api_base, json=payload, headers=headers)

                if response.status_code == 429:
                    if attempt == self.max_retries:
                        response.raise_for_status()
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue

                response.raise_for_status()
                data = response.json()

                # Update passages with rerank scores
                for result in data["results"]:
                    idx = result["index"]
                    score = result["relevance_score"]
                    passages[idx].metadata["rerank_score"] = score

                # Sort by rerank score
                passages.sort(
                    key=lambda x: x.metadata.get("rerank_score", 0),
                    reverse=True,
                )

                return passages[:top_k]

        return passages[:top_k]


@dataclass(slots=True)
class BGEReranker:
    """Cross-encoder re-ranker using BAAI/bge-reranker-v2-m3.

    Re-ranks search results by computing relevance scores
    between query and each passage.
    """

    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Load the re-ranker model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The `transformers` package is required for BGE re-ranker. "
                "Install with `pip install transformers torch`."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()

    def rerank(
        self,
        query: str,
        passages: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rank passages by relevance to query.

        Args:
            query: The search query
            passages: Passages to re-rank
            top_k: Number of top results to return

        Returns:
            Re-ranked passages sorted by relevance score desc
        """
        if self._model is None:
            self.load()

        if not passages:
            return []

        # Prepare pairs
        pairs = [(query, p.text) for p in passages]

        # Tokenize
        inputs = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Compute scores
        import torch
        with torch.no_grad():
            outputs = self._model(**inputs)
            scores = outputs.logits.squeeze(-1)

        # Update passages with rerank scores
        for i, score in enumerate(scores):
            passages[i].metadata["rerank_score"] = float(score)

        # Sort by rerank score
        passages.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)

        return passages[:top_k]


@dataclass(slots=True)
class DeterministicTestReranker:
    """Fast deterministic re-ranker for dev/test — no model loading."""

    def rerank(
        self,
        query: str,
        passages: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Re-rank by simple text overlap (for testing)."""
        query_words = set(query.lower().split())

        for p in passages:
            text_words = set(p.text.lower().split())
            overlap = len(query_words & text_words)
            p.metadata["rerank_score"] = float(overlap)

        passages.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return passages[:top_k]
