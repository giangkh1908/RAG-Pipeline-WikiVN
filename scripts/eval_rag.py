"""Run a comprehensive RAG evaluation with per-category metrics and LLM judge.

Usage:
    $env:PYTHONIOENCODING="utf-8"; python scripts/eval_rag.py

The script loads data/eval/test_suite.jsonl, runs each query through the RAG
pipeline, evaluates retrieval quality, asks an LLM judge to score the answer,
and writes a JSON/CSV report to data/eval/.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_pipeline.config import RAGConfig
from rag_pipeline.generation import CitationContextBuilder, LLMAnswerGenerator
from rag_pipeline.indexing import DenseEmbedder, QdrantVectorStore, SparseEmbedder
from rag_pipeline.retrieval import FilterBuilder, HybridRetriever, LLMQueryProcessor
from rag_pipeline.retrieval.models import RetrievalResult
from rag_pipeline.retrieval.query_cache import QueryCache
from rag_pipeline.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_pipeline.storage import SQLiteStorage


TEST_SUITE_PATH = Path("data/eval/test_suite.jsonl")
REPORT_DIR = Path("data/eval")
JUDGE_MODEL = "deepseek/deepseek-v4-flash"
API_BASE = "https://openrouter.ai/api/v1"
RETRIEVAL_TOP_K = 5


@dataclass
class EvalResult:
    """Result for a single test case."""

    id: str
    category: str
    question: str
    expected_behavior: str
    expected_keywords: list[str]
    answer: str = ""
    retrieved_chunks: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    correctness: float = 0.0
    relevance: float = 0.0
    hallucination: bool = False
    refusal_appropriate: str = "n/a"
    judge_reasoning: str = ""


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    return key


def load_test_suite(path: Path) -> list[dict[str, Any]]:
    """Load test cases from JSONL."""
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_components(config: RAGConfig) -> dict[str, Any]:
    """Construct all pipeline components."""
    storage = SQLiteStorage(config.retrieval.storage.db_path)
    vector_store = QdrantVectorStore(config.retrieval.qdrant)
    dense_embedder = DenseEmbedder(config.retrieval.dense)
    sparse_embedder = SparseEmbedder(config.retrieval.sparse)

    cache = QueryCache(config.retrieval.storage.db_path)
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


def compute_retrieval_metrics(
    results: list[RetrievalResult],
    expected_keywords: list[str],
    relevant_context: str,
    k: int = RETRIEVAL_TOP_K,
) -> tuple[float, float]:
    """Compute recall@k and MRR for retrieved chunks."""
    top_results = results[:k]
    if not top_results:
        return 0.0, 0.0

    # A chunk is "relevant" if it contains any expected keyword or the relevant context
    def is_relevant(chunk_text: str) -> bool:
        text_lower = chunk_text.lower()
        if relevant_context and relevant_context.lower() in text_lower:
            return True
        return any(kw.lower() in text_lower for kw in expected_keywords)

    recall_hits = sum(1 for r in top_results if is_relevant(r.content))
    recall_at_k = recall_hits / max(1, len(expected_keywords)) if expected_keywords else float(bool(recall_hits))
    recall_at_k = min(1.0, recall_at_k)

    mrr = 0.0
    for rank, result in enumerate(top_results, start=1):
        if is_relevant(result.content):
            mrr = 1.0 / rank
            break

    return recall_at_k, mrr


def judge_answer(
    client: httpx.Client,
    test_case: dict[str, Any],
    answer: str,
    retrieved_chunks: list[str],
) -> dict[str, Any]:
    """Ask an LLM judge to evaluate the generated answer."""
    chunks_text = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks[:3]))

    prompt = f"""Bạn là giám khảo đánh giá hệ thống RAG du lịch Việt Nam.

Category câu hỏi: {test_case['category']}
Câu hỏi: {test_case['question']}
Hành vi mong đợi: {test_case['expected_behavior']}
Từ khóa / đáp án mong đợi: {', '.join(test_case['expected_keywords']) if test_case['expected_keywords'] else 'Không có'}

Câu trả lờ i của hệ thống:
{answer}

Các đoạn văn bản đã truy xuất:
{chunks_text}

Hãy đánh giá và trả về JSON duy nhất với các trường:
{{
  "correctness": <số 1-5>,
  "relevance": <số 1-5>,
  "hallucination": <true|false>,
  "refusal_appropriate": <"yes"|"no"|"n/a">,
  "reasoning": <string ngắn>
}}

- correctness: Câu trả lờ i có đúng và đầy đủ không?
- relevance: Có liên quan đến câu hỏi không?
- hallucination: Có tự bịa ra thông tin không?
- refusal_appropriate: Chỉ đánh giá nếu category là out-of-scope hoặc missing. Hệ thống có từ chối hợp lý không?
- reasoning: Giải thích ngắn gọn.

Chỉ trả về JSON, không thêm văn bản khác."""

    response = client.post(
        "/chat/completions",
        json={
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": "You are an objective evaluator. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return json.loads(content)


def run_single_eval(
    components: dict[str, Any],
    judge_client: httpx.Client,
    test_case: dict[str, Any],
) -> EvalResult:
    """Run one test case through the full pipeline and judge."""
    result = EvalResult(
        id=test_case["id"],
        category=test_case["category"],
        question=test_case["question"],
        expected_behavior=test_case["expected_behavior"],
        expected_keywords=test_case.get("expected_keywords", []),
    )

    t0 = time.perf_counter()
    processed = components["retrieval_pipeline"].preprocess(test_case["question"])
    retrieved = components["retrieval_pipeline"].search_processed(processed, top_k=RETRIEVAL_TOP_K)

    if not retrieved:
        result.answer = "Không đủ thông tin để trả lờ i câu hỏi này."
        result.retrieved_chunks = []
    else:
        built = components["context_builder"].build(retrieved, query=test_case["question"])
        tokens: list[str] = []
        for token in components["answer_generator"].generate_stream(
            test_case["question"], built.context
        ):
            tokens.append(token)
        result.answer = "".join(tokens)
        result.retrieved_chunks = [r.content for r in retrieved]

    result.latency_ms = (time.perf_counter() - t0) * 1000

    result.recall_at_k, result.mrr = compute_retrieval_metrics(
        retrieved,
        result.expected_keywords,
        test_case.get("relevant_context", ""),
        k=RETRIEVAL_TOP_K,
    )

    # LLM judge
    try:
        judge = judge_answer(judge_client, test_case, result.answer, result.retrieved_chunks)
        result.correctness = float(judge.get("correctness", 0))
        result.relevance = float(judge.get("relevance", 0))
        result.hallucination = bool(judge.get("hallucination", False))
        result.refusal_appropriate = str(judge.get("refusal_appropriate", "n/a"))
        result.judge_reasoning = str(judge.get("reasoning", ""))
    except Exception as exc:
        result.judge_reasoning = f"Judge error: {exc}"

    return result


def compute_summary(results: list[EvalResult]) -> dict[str, Any]:
    """Compute per-category and overall summary."""
    categories = sorted({r.category for r in results})
    summary: dict[str, Any] = {"overall": {}, "by_category": {}}

    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "p50": 0.0}
        values_sorted = sorted(values)
        n = len(values_sorted)
        p50 = values_sorted[n // 2] if n % 2 else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2
        return {"mean": statistics.mean(values_sorted), "p50": p50}

    # Overall
    summary["overall"] = {
        "count": len(results),
        "recall_at_k": _stats([r.recall_at_k for r in results]),
        "mrr": _stats([r.mrr for r in results]),
        "correctness": _stats([r.correctness for r in results]),
        "relevance": _stats([r.relevance for r in results]),
        "hallucination_rate": sum(1 for r in results if r.hallucination) / max(1, len(results)),
        "avg_latency_ms": statistics.mean([r.latency_ms for r in results]) if results else 0,
    }

    # By category
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        refusal_yes = sum(1 for r in cat_results if r.refusal_appropriate == "yes")
        refusal_total = sum(1 for r in cat_results if r.refusal_appropriate in ("yes", "no"))
        summary["by_category"][cat] = {
            "count": len(cat_results),
            "recall_at_k": _stats([r.recall_at_k for r in cat_results]),
            "mrr": _stats([r.mrr for r in cat_results]),
            "correctness": _stats([r.correctness for r in cat_results]),
            "relevance": _stats([r.relevance for r in cat_results]),
            "hallucination_rate": sum(1 for r in cat_results if r.hallucination) / max(1, len(cat_results)),
            "refusal_appropriate_rate": refusal_yes / max(1, refusal_total) if refusal_total else None,
            "avg_latency_ms": statistics.mean([r.latency_ms for r in cat_results]) if cat_results else 0,
        }

    return summary


def save_report(results: list[EvalResult], summary: dict[str, Any]) -> tuple[Path, Path]:
    """Save report as JSON and CSV."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())

    json_path = REPORT_DIR / f"report_{timestamp}.json"
    report = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = REPORT_DIR / f"report_{timestamp}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        if results:
            headers = list(asdict(results[0]).keys())
            f.write(",".join(headers) + "\n")
            for r in results:
                row = []
                for h in headers:
                    val = getattr(r, h)
                    if isinstance(val, list):
                        val = "|".join(str(v) for v in val)
                    elif isinstance(val, str):
                        val = val.replace('"', '""').replace("\n", " ")
                    row.append(f'"{val}"')
                f.write(",".join(row) + "\n")

    return json_path, csv_path


def print_summary(summary: dict[str, Any]) -> None:
    """Print a readable summary table."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    overall = summary["overall"]
    print(f"Total questions: {overall['count']}")
    print(f"Hallucination rate: {overall['hallucination_rate']:.2%}")
    print(f"Avg latency: {overall['avg_latency_ms']:.0f}ms")
    print(f"Mean correctness: {overall['correctness']['mean']:.2f}/5")
    print(f"Mean relevance: {overall['relevance']['mean']:.2f}/5")
    print(f"Mean recall@k: {overall['recall_at_k']['mean']:.2%}")
    print(f"Mean MRR: {overall['mrr']['mean']:.3f}")

    print("\nBy category:")
    print(f"{'Category':<15} {'Count':>6} {'Correct':>9} {'Relevance':>10} {'Recall':>8} {'Halluc':>8} {'Refusal':>9}")
    print("-" * 80)
    for cat, stats in summary["by_category"].items():
        refusal = stats["refusal_appropriate_rate"]
        refusal_str = f"{refusal:.0%}" if refusal is not None else "n/a"
        print(
            f"{cat:<15} {stats['count']:>6} "
            f"{stats['correctness']['mean']:>8.2f} "
            f"{stats['relevance']['mean']:>9.2f} "
            f"{stats['recall_at_k']['mean']:>7.1%} "
            f"{stats['hallucination_rate']:>7.1%} "
            f"{refusal_str:>8}"
        )
    print("=" * 80)


def main() -> int:
    load_dotenv()

    if not TEST_SUITE_PATH.exists():
        print(f"Test suite not found: {TEST_SUITE_PATH}")
        print("Run: python scripts/generate_eval_suite.py")
        return 1

    test_cases = load_test_suite(TEST_SUITE_PATH)
    print(f"Loaded {len(test_cases)} test cases from {TEST_SUITE_PATH}\n")

    config = RAGConfig()
    components = build_components(config)
    judge_client = httpx.Client(
        base_url=API_BASE,
        timeout=120.0,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://rag-pipeline.local",
            "X-Title": "RAG Pipeline Eval",
        },
    )

    results: list[EvalResult] = []
    try:
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] {test_case['id']} ({test_case['category']}): {test_case['question']}")
            result = run_single_eval(components, judge_client, test_case)
            results.append(result)
            print(f"    -> latency={result.latency_ms:.0f}ms recall@k={result.recall_at_k:.2f} "
                  f"correct={result.correctness:.1f} rel={result.relevance:.1f} "
                  f"hallucination={result.hallucination}")
    finally:
        components["dense_embedder"].close()
        components["answer_generator"].close()
        components["llm_processor"].close()
        judge_client.close()

    summary = compute_summary(results)
    json_path, csv_path = save_report(results, summary)
    print_summary(summary)
    print(f"\nReports saved to:\n  {json_path}\n  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
