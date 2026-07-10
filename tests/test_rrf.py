"""Tests for RRF fusion."""

from rag_pipeline.indexing.rrf import rrf_fusion
from rag_pipeline.indexing.vector_store import SearchResult


class TestRRFFusion:
    def _make_result(self, chunk_id: str, score: float) -> SearchResult:
        return SearchResult(chunk_id=chunk_id, doc_id=f"doc_{chunk_id}", text=f"text {chunk_id}", score=score)

    def _make_bm25(self, chunk_id: str, score: float) -> tuple[str, str, float, str]:
        return (chunk_id, f"doc_{chunk_id}", score, f"text {chunk_id}")

    def test_basic_fusion(self):
        dense = [self._make_result("a", 0.9), self._make_result("b", 0.8)]
        bm25 = [self._make_bm25("a", 10.0), self._make_bm25("c", 5.0)]

        results = rrf_fusion(dense, bm25, k=60, top_k=3)

        # "a" appears in both → highest RRF score
        assert results[0].chunk_id == "a"
        # "c" is BM25-only, now included via chunk_map fix
        chunk_ids = [r.chunk_id for r in results]
        assert "a" in chunk_ids
        assert "b" in chunk_ids
        assert "c" in chunk_ids

    def test_rrf_score_calculation(self):
        dense = [self._make_result("a", 0.9)]
        bm25 = [self._make_bm25("a", 10.0)]

        results = rrf_fusion(dense, bm25, k=60, top_k=1)

        # RRF score = 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        assert abs(results[0].score - expected) < 1e-6

    def test_top_k_limit(self):
        dense = [self._make_result(f"d{i}", 0.9 - i * 0.1) for i in range(10)]
        bm25 = [self._make_bm25(f"d{i}", 10.0 - i) for i in range(10)]

        results = rrf_fusion(dense, bm25, k=60, top_k=5)
        assert len(results) == 5

    def test_empty_dense(self):
        bm25 = [self._make_bm25("a", 10.0), self._make_bm25("b", 5.0)]
        results = rrf_fusion([], bm25, k=60, top_k=5)
        # BM25-only results now included
        assert len(results) == 2
        assert results[0].chunk_id == "a"

    def test_empty_bm25(self):
        dense = [self._make_result("a", 0.9), self._make_result("b", 0.8)]
        results = rrf_fusion(dense, [], k=60, top_k=5)
        assert len(results) == 2
        assert results[0].chunk_id == "a"

    def test_both_empty(self):
        results = rrf_fusion([], [], k=60, top_k=5)
        assert results == []

    def test_bm25_only_results_not_lost(self):
        """BM25-only results should appear in final output."""
        dense = [self._make_result("a", 0.9)]
        bm25 = [self._make_bm25("x", 10.0), self._make_bm25("y", 8.0)]

        results = rrf_fusion(dense, bm25, k=60, top_k=10)

        chunk_ids = [r.chunk_id for r in results]
        assert "a" in chunk_ids  # dense result
        assert "x" in chunk_ids  # BM25-only
        assert "y" in chunk_ids  # BM25-only
