"""Embedding clients for vector indexing."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Protocol, cast

from rag_pipeline.config import EmbeddingConfig


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""


@dataclass(slots=True)
class OpenRouterEmbeddingClient:
    """Embed texts via OpenRouter API with parallel sub-batching and retry.

    Model: nvidia/llama-nemotron-embed-vl-1b-v2
    - Context window: 131K tokens
    - Free tier: ~20-60 RPM
    - Sub-batch: 500 texts/request
    - Parallel workers: 4 concurrent API calls
    - Retry: exponential backoff on 429
    """

    config: EmbeddingConfig
    parallel_workers: int = 4
    retry_base_delay: float = 2.0

    @property
    def sub_batch_size(self) -> int:
        return self.config.sub_batch_size

    @property
    def max_retries(self) -> int:
        return self.config.max_retries

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in parallel sub-batches with retry on rate limit."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "The `httpx` package is required for OpenRouter embeddings. "
                "Install with `pip install .[indexing]`."
            ) from exc

        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing environment variable `{self.config.api_key_env}` for OpenRouter access."
            )

        # Split into sub-batches
        sub_batches: list[tuple[int, list[str]]] = []
        for start in range(0, len(texts), self.sub_batch_size):
            batch = texts[start : start + self.sub_batch_size]
            sub_batches.append((start, batch))

        if len(sub_batches) == 1:
            # Single batch — no need for thread pool
            headers = {"Authorization": f"Bearer {api_key}"}
            with httpx.Client(base_url=self.config.api_base, timeout=self.config.timeout_seconds) as client:
                vectors = self._embed_batch_with_retry(client, headers, sub_batches[0][1])
            return vectors

        # Parallel execution
        all_vectors: list[list[float]] = [cast(list[float], None) for _ in texts]
        headers = {"Authorization": f"Bearer {api_key}"}

        def _process_batch(idx_start: tuple[int, list[str]]) -> tuple[int, list[list[float]]]:
            idx, batch = idx_start
            with httpx.Client(base_url=self.config.api_base, timeout=self.config.timeout_seconds) as client:
                vectors = self._embed_batch_with_retry(client, headers, batch)
            return idx, vectors

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {executor.submit(_process_batch, sb): sb[0] for sb in sub_batches}
            for future in as_completed(futures):
                idx, vectors = future.result()
                all_vectors[idx : idx + len(vectors)] = vectors

        return all_vectors

    def _embed_batch_with_retry(
        self, client, headers: dict, batch: list[str]
    ) -> list[list[float]]:
        """Send one sub-batch with exponential backoff on 429."""
        payload = {"model": self.config.model_name, "input": batch}

        for attempt in range(self.max_retries + 1):
            response = client.post("/embeddings", json=payload, headers=headers)

            if response.status_code == 429:
                if attempt == self.max_retries:
                    response.raise_for_status()
                delay = self.retry_base_delay * (2 ** attempt)
                time.sleep(delay)
                continue

            response.raise_for_status()
            data = response.json()["data"]
            return [item["embedding"] for item in data]

        return [[] for _ in batch]


@dataclass(slots=True)
class DeterministicTestEmbedder:
    """Fast deterministic embedder for dev/test — no API calls."""

    dimensions: int = 8

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            base = sum(ord(ch) for ch in text)
            vectors.append([float((base + i) % 97) / 97.0 for i in range(self.dimensions)])
        return vectors


@dataclass
class LocalEmbedder:
    """Local embedding using sentence-transformers on GPU.

    Default model: Qwen/Qwen3-Embedding-0.6B
    - 1024 dimensions
    - Supports Vietnamese text
    - Runs on CUDA GPU
    """

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 256
    device: str = "auto"  # "auto", "cuda", "cpu"
    _model: any = field(default=None, init=False, repr=False)
    _device: str = field(default="", init=False)

    def __post_init__(self):
        self._init_model()

    def _init_model(self):
        """Initialize the model and move to GPU."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Local embedding requires `sentence-transformers` and `torch`. "
                "Install with: pip install sentence-transformers torch"
            ) from exc

        # Determine device
        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.device

        print(f"[EMBEDDER] Loading model '{self.model_name}' on {self._device}...", flush=True)
        self._model = SentenceTransformer(self.model_name, device=self._device)
        print(f"[EMBEDDER] Model loaded. Embedding dim: {self._model.get_sentence_embedding_dimension()}", flush=True)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local model on GPU."""
        if self._model is None:
            raise RuntimeError("Model not initialized.")

        all_vectors: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            # Encode batch
            embeddings = self._model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            all_vectors.extend(embeddings.tolist())

            # Progress log
            if batch_num % 10 == 0 or batch_num == total_batches:
                sys.stdout.write(f"\r\033[K")
                sys.stdout.write(f"[EMBED] {batch_num}/{total_batches} batches | {len(all_vectors):,}/{len(texts):,} texts")
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()
        return all_vectors
