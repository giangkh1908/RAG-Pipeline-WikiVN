"""Tests for dense and sparse embedders."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from rag_pipeline.config import DenseEmbeddingConfig, SparseEmbeddingConfig
from rag_pipeline.indexing.embedders import DenseEmbedder, SparseEmbedder


class TestDenseEmbedder:
    @patch("rag_pipeline.indexing.embedders.httpx.Client")
    def test_embed_returns_vectors(self, mock_client_class: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        os.environ["OPENROUTER_API_KEY"] = "test-key"
        embedder = DenseEmbedder(DenseEmbeddingConfig(batch_size=10))
        results = embedder.embed(["hello", "world"])

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        assert call_kwargs["json"]["model"] == DenseEmbeddingConfig().model_name
        assert call_kwargs["json"]["input"] == ["hello", "world"]

    def test_embed_raises_without_api_key(self) -> None:
        os.environ.pop("OPENROUTER_API_KEY", None)
        with pytest.raises(RuntimeError, match="Missing API key"):
            DenseEmbedder(DenseEmbeddingConfig())


class TestSparseEmbedder:
    @pytest.fixture
    def temp_config(self) -> SparseEmbeddingConfig:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SparseEmbeddingConfig(vocab_path=os.path.join(tmpdir, "bm25_vocab.json"))

    def test_embed_requires_fit_first(self, temp_config: SparseEmbeddingConfig) -> None:
        embedder = SparseEmbedder(temp_config)
        # Without fitting, vocab is empty → all vectors empty
        results = embedder.embed(["hello world"])
        assert results == [{}]

    def test_fit_builds_vocab_and_embed_returns_bm25_vectors(
        self, temp_config: SparseEmbeddingConfig
    ) -> None:
        embedder = SparseEmbedder(temp_config)
        corpus = [
            "hello world",
            "hello vietnam",
            "vietnam tourism is great",
        ]
        embedder.fit(corpus)

        assert len(embedder.vocab) > 0
        assert embedder.total_docs == 3

        results = embedder.embed(["hello vietnam tourism"])
        assert len(results) == 1
        vector = results[0]
        assert len(vector) == 3  # hello, vietnam, tourism
        assert all(isinstance(idx, int) for idx in vector)
        assert all(value > 0 for value in vector.values())

    def test_vocab_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "bm25_vocab.json")
            config = SparseEmbeddingConfig(vocab_path=vocab_path)

            embedder = SparseEmbedder(config)
            embedder.fit(["hello world", "hello vietnam"])

            assert os.path.exists(vocab_path)

            # Load in a new instance
            new_embedder = SparseEmbedder(config)
            assert new_embedder.vocab == embedder.vocab
            assert new_embedder.total_docs == embedder.total_docs

            results = new_embedder.embed(["hello"])
            assert len(results[0]) == 1

    def test_idf_penalizes_common_terms(self, temp_config: SparseEmbeddingConfig) -> None:
        embedder = SparseEmbedder(temp_config)
        corpus = [
            "hello world",
            "hello vietnam",
            "hello thailand",
        ]
        embedder.fit(corpus)

        results = embedder.embed(["hello world"])
        vector = results[0]

        hello_idx = embedder.vocab["hello"]
        world_idx = embedder.vocab["world"]

        # "hello" appears in all docs → lower IDF than "world"
        assert vector[hello_idx] < vector[world_idx]

    def test_empty_input_returns_empty_list(self, temp_config: SparseEmbeddingConfig) -> None:
        embedder = SparseEmbedder(temp_config)
        assert embedder.embed([]) == []
