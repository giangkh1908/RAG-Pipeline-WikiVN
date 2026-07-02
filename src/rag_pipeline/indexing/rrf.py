"""Reciprocal Rank Fusion (RRF) for merging search results."""

from __future__ import annotations

from rag_pipeline.indexing.vector_store import SearchResult


def rrf_fusion(
    dense_results: list[SearchResult],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    top_k: int = 20,
) -> list[SearchResult]:
    """Merge dense + BM25 results using Reciprocal Rank Fusion.

    RRF score(d) = Σ 1/(k + rank_i(d)) for each ranking where d appears.

    Args:
        dense_results: Results from dense vector search (sorted by score desc)
        bm25_results: Results from BM25 search as (chunk_id, score) pairs (sorted by score desc)
        k: RRF constant (default 60, standard value)
        top_k: Number of results to return

    Returns:
        Merged results sorted by RRF score desc, with rrf_score in metadata
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, SearchResult] = {}

    # Dense scores: rank 0 = best
    for rank, result in enumerate(dense_results):
        rrf_score = 1.0 / (k + rank + 1)
        rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + rrf_score
        chunk_map[result.chunk_id] = result

    # BM25 scores: rank 0 = best
    for rank, (chunk_id, _bm25_score) in enumerate(bm25_results):
        rrf_score = 1.0 / (k + rank + 1)
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

    # Sort by RRF score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results: list[SearchResult] = []
    for chunk_id, rrf_score in ranked[:top_k]:
        if chunk_id in chunk_map:
            result = chunk_map[chunk_id]
            # Create a copy with RRF score in metadata
            merged = SearchResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                text=result.text,
                score=rrf_score,  # Use RRF score as the main score
                metadata={**result.metadata, "rrf_score": rrf_score, "dense_score": result.score},
            )
            results.append(merged)

    return results
