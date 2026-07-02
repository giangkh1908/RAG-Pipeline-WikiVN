"""Retrieval pipeline — orchestrates dense search → BM25 → RRF → re-rank."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.config import RetrievalConfig
from rag_pipeline.indexing.bm25_index import BM25Index
from rag_pipeline.indexing.embedder import Embedder
from rag_pipeline.indexing.reranker import BGEReranker, DeterministicTestReranker
from rag_pipeline.indexing.rrf import rrf_fusion
from rag_pipeline.indexing.vector_store import SearchResult, VectorStore
from rag_pipeline.models import Passage, ProcessedQuery, RetrievalResult


@dataclass(slots=True)
class RetrievalPipeline:
    """Orchestrates hybrid retrieval: dense + BM25 → RRF → re-rank.

    Flow:
    1. Embed rewrite_query → dense vector
    2. Dense search: Qdrant cosine similarity
    3. BM25 search: keyword matching
    4. RRF fusion: merge dense + BM25 results
    5. BGE re-rank (optional): cross-encoder scoring
    6. Build Passage list + context string
    """

    config: RetrievalConfig
    embedder: Embedder
    vector_store: VectorStore
    bm25_index: BM25Index
    reranker: BGEReranker | DeterministicTestReranker | None = None

    def run(self, query: ProcessedQuery) -> RetrievalResult:
        """Run full retrieval pipeline."""
        # Step 1: Dense search
        query_vector = self.embedder.embed_texts([query.rewrite_query])[0]
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=self.config.dense_top_k,
        )

        # Step 2: BM25 search
        bm25_results: list[tuple[str, float]] = []
        if self.bm25_index.is_loaded:
            bm25_results = self.bm25_index.search(
                query=query.bm25_query,
                top_k=self.config.bm25_top_k,
            )

        # Step 3: RRF fusion
        fused = rrf_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=self.config.rrf_k,
            top_k=self.config.rrf_top_k,
        )

        # Step 4: BGE re-ranking (optional)
        if self.config.enable_rerank and self.reranker is not None:
            fused = self.reranker.rerank(
                query=query.rewrite_query,
                passages=fused,
                top_k=self.config.rerank_top_k,
            )

        # Step 5: Filter by minimum score
        if self.config.min_score > 0:
            fused = [r for r in fused if r.score >= self.config.min_score]

        # Step 6: Build passages
        passages = [self._to_passage(result, rank=i + 1) for i, result in enumerate(fused)]

        # Step 7: Assemble context
        context = self._assemble_context(passages)

        return RetrievalResult(
            query=query,
            passages=passages,
            context=context,
            metadata={
                "dense_count": len(dense_results),
                "bm25_count": len(bm25_results),
                "fused_count": len(fused),
            },
        )

    def _to_passage(self, result: SearchResult, rank: int) -> Passage:
        """Convert SearchResult to Passage."""
        return Passage(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            title=result.metadata.get("title", ""),
            text=result.text,
            source_url=result.metadata.get("source_url", ""),
            dense_score=result.metadata.get("dense_score", 0.0),
            bm25_score=result.metadata.get("bm25_score", 0.0),
            rrf_score=result.metadata.get("rrf_score", result.score),
            rerank_score=result.metadata.get("rerank_score", 0.0),
            rank=rank,
        )

    def _assemble_context(self, passages: list[Passage]) -> str:
        """Assemble context string from passages."""
        if not passages:
            return ""

        parts: list[str] = []
        for p in passages:
            source = f"({p.title})" if p.title else ""
            parts.append(f"[{p.rank}] {source} {p.text}")

        return "\n\n".join(parts)
