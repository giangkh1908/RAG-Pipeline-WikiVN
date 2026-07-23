"""Benchmark per-stage latency of the RAG pipeline.

Usage:
    $env:PYTHONIOENCODING="utf-8"; python scripts/benchmark_latency.py

The script measures latency for each stage:
    - rewrite (LLM query preprocessing)
    - dense embedding
    - dense search
    - sparse embedding
    - sparse search
    - full hybrid retrieval
    - context building
    - generation (TTFT + total)
    - end-to-end

Results are saved to data/benchmarks/ as JSON and CSV.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation import CitationContextBuilder, LLMAnswerGenerator
from rag_pipeline.indexing import DenseEmbedder, QdrantVectorStore, SparseEmbedder
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage import SQLiteStorage


@dataclass
class StageTimings:
    """Latency measurements for one query."""

    query: str
    rewrite_ms: float = 0.0
    dense_embed_ms: float = 0.0
    dense_search_ms: float = 0.0
    sparse_embed_ms: float = 0.0
    sparse_search_ms: float = 0.0
    retrieve_total_ms: float = 0.0
    context_build_ms: float = 0.0
    generation_ttft_ms: float = 0.0
    generation_total_ms: float = 0.0
    generation_tokens: int = 0
    e2e_ms: float = 0.0
    e2e_ttft_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Full benchmark output."""

    queries: list[StageTimings] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)


def load_queries(path: Path, limit: int | None = None) -> list[str]:
    """Load questions from a JSONL evaluation file."""
    if not path.exists():
        return []
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            queries.append(data["question"])
            if limit and len(queries) >= limit:
                break
    return queries


def build_components(config: RAGConfig) -> dict[str, Any]:
    """Construct all pipeline components."""
    storage = SQLiteStorage(config.retrieval.storage.db_path)
    vector_store = QdrantVectorStore(config.retrieval.qdrant)
    dense_embedder = DenseEmbedder(config.retrieval.dense)
    sparse_embedder = SparseEmbedder(config.retrieval.sparse)

    cache = QueryCache(storage)
    llm_processor = LLMQueryProcessor(config.retrieval.llm_query, cache=cache)
    filter_builder = FilterBuilder()
    retriever = HybridRetriever(
        config=config.retrieval,
        storage=storage,
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
    )
    retrieval_pipeline = RetrievalPipeline(llm_processor, filter_builder, retriever)
    context_builder = CitationContextBuilder(config.context_builder)
    answer_generator = LLMAnswerGenerator(config.generation)

    return {
        "storage": storage,
        "vector_store": vector_store,
        "dense_embedder": dense_embedder,
        "sparse_embedder": sparse_embedder,
        "llm_processor": llm_processor,
        "filter_builder": filter_builder,
        "retriever": retriever,
        "retrieval_pipeline": retrieval_pipeline,
        "context_builder": context_builder,
        "answer_generator": answer_generator,
    }


def benchmark_query(components: dict[str, Any], query: str) -> StageTimings:
    """Measure latency for every stage of a single query."""
    timings = StageTimings(query=query)
    config: RAGConfig = RAGConfig()

    # 1. Rewrite / query preprocessing
    t0 = time.perf_counter()
    processed = components["llm_processor"].process(query)
    timings.rewrite_ms = (time.perf_counter() - t0) * 1000

    filters = components["filter_builder"].build(processed)
    search_query = processed.rewritten_query or processed.normalized_query

    # 2. Dense embedding
    t0 = time.perf_counter()
    dense_vectors = components["dense_embedder"].embed([search_query])
    timings.dense_embed_ms = (time.perf_counter() - t0) * 1000

    # 3. Dense search
    t0 = time.perf_counter()
    components["vector_store"].search_dense(
        dense_vectors[0],
        top_k=config.retrieval.qdrant.dense_top_k,
        filters=filters,
    )
    timings.dense_search_ms = (time.perf_counter() - t0) * 1000

    # 4. Sparse embedding
    t0 = time.perf_counter()
    sparse_vectors = components["sparse_embedder"].embed([search_query])
    timings.sparse_embed_ms = (time.perf_counter() - t0) * 1000

    # 5. Sparse search
    t0 = time.perf_counter()
    components["vector_store"].search_sparse(
        sparse_vectors[0],
        top_k=config.retrieval.qdrant.sparse_top_k,
        filters=filters,
    )
    timings.sparse_search_ms = (time.perf_counter() - t0) * 1000

    # 6. Full hybrid retrieval (dense + sparse + RRF + storage lookup)
    t0 = time.perf_counter()
    results = components["retriever"].retrieve(
        search_query,
        top_k=config.retrieval.rrf_top_k,
        filters=filters,
    )
    timings.retrieve_total_ms = (time.perf_counter() - t0) * 1000

    if not results:
        return timings

    # 7. Context building
    t0 = time.perf_counter()
    built = components["context_builder"].build(results)
    timings.context_build_ms = (time.perf_counter() - t0) * 1000

    # 8. Generation (TTFT + total)
    t0 = time.perf_counter()
    first_token_seen = False
    tokens: list[str] = []
    for token in components["answer_generator"].generate_stream(query, built.context):
        if not first_token_seen:
            timings.generation_ttft_ms = (time.perf_counter() - t0) * 1000
            first_token_seen = True
        tokens.append(token)
    timings.generation_total_ms = (time.perf_counter() - t0) * 1000
    timings.generation_tokens = len(tokens)

    return timings


def benchmark_e2e(components: dict[str, Any], query: str) -> float:
    """Measure end-to-end latency through the full RAGPipeline.

    Note: this reuses the same shared components so we do not call
    ``pipeline.close()`` here; resources are released in ``main``.
    """
    from rag_pipeline.generation import RAGPipeline

    pipeline = RAGPipeline(
        components["retrieval_pipeline"],
        components["context_builder"],
        components["answer_generator"],
    )
    t0 = time.perf_counter()
    first_token_ms: float | None = None
    for event in pipeline.answer_stream(query):
        # e2e TTFT = time until the first answer token the user sees.
        if first_token_ms is None and getattr(event, "type", None) == "token":
            first_token_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    return total_ms, first_token_ms if first_token_ms is not None else total_ms


def compute_summary(timings: list[StageTimings]) -> dict[str, dict[str, float]]:
    """Compute mean, p50, p95, p99 per metric."""
    metrics = [
        "rewrite_ms",
        "dense_embed_ms",
        "dense_search_ms",
        "sparse_embed_ms",
        "sparse_search_ms",
        "retrieve_total_ms",
        "context_build_ms",
        "generation_ttft_ms",
        "generation_total_ms",
        "e2e_ms",
        "e2e_ttft_ms",
    ]
    summary: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = sorted(getattr(t, metric) for t in timings)
        if not values:
            continue
        summary[metric] = {
            "mean": statistics.mean(values),
            "p50": values[len(values) // 2]
            if len(values) % 2
            else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2,
            "p95": values[int(len(values) * 0.95)],
            "p99": values[min(int(len(values) * 0.99), len(values) - 1)],
        }
    return summary


def save_results(result: BenchmarkResult, output_dir: Path) -> None:
    """Persist benchmark results as JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())

    json_path = output_dir / f"latency_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    csv_path = output_dir / f"latency_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        if not result.queries:
            return
        headers = list(asdict(result.queries[0]).keys())
        f.write(",".join(headers) + "\n")
        for timing in result.queries:
            row = [str(getattr(timing, h)) for h in headers]
            f.write(",".join(row) + "\n")

    print(f"\nResults saved to:\n  {json_path}\n  {csv_path}")


def print_summary(result: BenchmarkResult) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("Latency Summary (ms)")
    print("=" * 80)
    print(f"{'Metric':<25} {'Mean':>10} {'P50':>10} {'P95':>10} {'P99':>10}")
    print("-" * 80)
    for metric, stats in result.summary.items():
        print(
            f"{metric:<25} "
            f"{stats['mean']:>10.2f} "
            f"{stats['p50']:>10.2f} "
            f"{stats['p95']:>10.2f} "
            f"{stats['p99']:>10.2f}"
        )
    print("=" * 80)


def main() -> None:
    load_dotenv()

    config = RAGConfig()
    components = build_components(config)

    queries_path = Path("data/eval/queries.jsonl")
    queries = load_queries(queries_path, limit=3)
    if not queries:
        queries = [
            "Ha Long Bay nằm ở đâu?",
            "Du lịch Hội An nên đi mùa nào?",
            "Có món ăn đặc sản nào ở Đà Nẵng?",
        ]

    print(f"Benchmarking {len(queries)} queries...\n")
    result = BenchmarkResult()

    try:
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query}")
            timings = benchmark_query(components, query)
            timings.e2e_ms, timings.e2e_ttft_ms = benchmark_e2e(components, query)
            result.queries.append(timings)
    finally:
        components["dense_embedder"].close()
        components["answer_generator"].close()
        components["llm_processor"].close()

    result.summary = compute_summary(result.queries)
    save_results(result, Path("data/benchmarks"))
    print_summary(result)


if __name__ == "__main__":
    main()
