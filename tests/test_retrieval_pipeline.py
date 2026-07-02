"""Tests for retrieval pipeline."""

import tempfile
from pathlib import Path

from rag_pipeline.config import RetrievalConfig
from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.embedder import DeterministicTestEmbedder
from rag_pipeline.indexing.reranker import DeterministicTestReranker
from rag_pipeline.indexing.vector_store import InMemoryVectorStore, SearchResult
from rag_pipeline.models import ProcessedQuery
from rag_pipeline.pipelines.retrieval_pipeline import RetrievalPipeline


def _make_processed_query(query: str = "test query") -> ProcessedQuery:
    return ProcessedQuery(
        qid="test-1",
        original_query=query,
        normalized_query=query.lower(),
        rewrite_query=query,
        bm25_query=query.lower(),
        intent="general",
    )


class TestRetrievalPipeline:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = RetrievalConfig(
            dense_top_k=10,
            bm25_top_k=10,
            rrf_top_k=5,
            enable_rerank=True,
            rerank_top_k=3,
            bm25_index_path=Path(self.tmpdir) / "bm25.pkl",
        )

    def test_basic_pipeline(self):
        # Build BM25 index
        bm25 = BM25Index(index_path=self.config.bm25_index_path, tokenizer_name="simple")
        bm25.build([
            ("chunk1", "thủ đô của việt nam là hà nội"),
            ("chunk2", "dân số trung quốc rất lớn"),
        ])

        pipeline = RetrievalPipeline(
            config=self.config,
            embedder=DeterministicTestEmbedder(),
            vector_store=InMemoryVectorStore(),
            bm25_index=bm25,
            reranker=DeterministicTestReranker(),
        )

        query = _make_processed_query("thủ đô việt nam")
        result = pipeline.run(query)

        assert result.query == query
        assert isinstance(result.context, str)
        assert len(result.passages) <= self.config.rerank_top_k

    def test_empty_vector_store(self):
        bm25 = BM25Index(index_path=self.config.bm25_index_path, tokenizer_name="simple")
        bm25.build([])

        pipeline = RetrievalPipeline(
            config=self.config,
            embedder=DeterministicTestEmbedder(),
            vector_store=InMemoryVectorStore(),
            bm25_index=bm25,
            reranker=None,
        )

        query = _make_processed_query()
        result = pipeline.run(query)

        assert result.passages == []
        assert result.context == ""

    def test_no_reranker(self):
        bm25 = BM25Index(index_path=self.config.bm25_index_path, tokenizer_name="simple")
        bm25.build([("chunk1", "hello world")])

        config = RetrievalConfig(enable_rerank=False, bm25_index_path=self.config.bm25_index_path)
        pipeline = RetrievalPipeline(
            config=config,
            embedder=DeterministicTestEmbedder(),
            vector_store=InMemoryVectorStore(),
            bm25_index=bm25,
            reranker=None,
        )

        query = _make_processed_query()
        result = pipeline.run(query)

        assert result.metadata["dense_count"] >= 0

    def test_context_assembly(self):
        bm25 = BM25Index(index_path=self.config.bm25_index_path, tokenizer_name="simple")
        bm25.build([])

        pipeline = RetrievalPipeline(
            config=self.config,
            embedder=DeterministicTestEmbedder(),
            vector_store=InMemoryVectorStore(),
            bm25_index=bm25,
            reranker=None,
        )

        query = _make_processed_query()
        result = pipeline.run(query)

        # Empty context when no passages
        assert result.context == ""
