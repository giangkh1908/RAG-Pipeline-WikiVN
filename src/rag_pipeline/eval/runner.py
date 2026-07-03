"""Eval runner — runs RAGAS evaluation on the RAG pipeline."""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_pipeline.config import EvalConfig
from rag_pipeline.eval.report import EvalReport
from rag_pipeline.pipelines.answer_pipeline import AnswerPipeline


@dataclass(slots=True)
class LatencyMetrics:
    """Latency metrics for a single query."""

    query_processing_ms: float = 0.0
    retrieval_ms: float = 0.0
    ttft_ms: float = 0.0  # Time to first token
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "query_processing_ms": round(self.query_processing_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "ttft_ms": round(self.ttft_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


@dataclass
class EvalRunner:
    """Runs RAGAS evaluation on the RAG pipeline.

    Flow:
    1. Load eval CSV (question, expected_answer, source_urls)
    2. Run full pipeline for each sample → get answer + retrieved contexts
    3. Create SingleTurnSample for RAGAS
    4. Call ragas.evaluate() with 4 metrics
    5. Return EvalReport with RAGAS scores + latency metrics
    """

    pipeline: AnswerPipeline
    config: EvalConfig

    def run(self, dataset_path: Path | None = None, limit: int = 50) -> EvalReport:
        """Run evaluation on the dataset.

        Args:
            dataset_path: Path to eval CSV (uses config default if None)
            limit: Max number of samples to evaluate

        Returns:
            EvalReport with scores, latency metrics, and per-sample breakdown
        """
        if dataset_path is None:
            dataset_path = self.config.eval_dataset_path

        # Step 1: Load eval dataset
        samples = self._load_dataset(dataset_path)[:limit]

        # Step 2: Run pipeline for each sample (with latency tracking)
        ragas_samples = self._run_pipeline(samples)

        # Step 3: Evaluate with RAGAS
        scores = self._evaluate(ragas_samples)

        # Step 4: Compute latency summary
        latency_summary = self._compute_latency_summary(samples)

        # Step 5: Build report
        thresholds = {
            "faithfulness": self.config.faithfulness_threshold,
            "answer_relevancy": self.config.answer_relevance_threshold,
            "context_precision": self.config.context_precision_threshold,
            "context_recall": self.config.context_recall_threshold,
        }

        return EvalReport(
            scores=scores,
            samples=[
                {
                    "question": s["question"],
                    "answer": s.get("answer", ""),
                    "expected_answer": s.get("expected_answer", ""),
                    "scores": s.get("ragas_scores", {}),
                    "latency": s.get("latency", {}).to_dict() if s.get("latency") else {},
                }
                for s in samples
            ],
            thresholds=thresholds,
            latency_summary=latency_summary,
        )

    def _load_dataset(self, path: Path) -> list[dict[str, Any]]:
        """Load eval CSV file.

        Expected format: question, expected_answer, source_urls
        """
        samples = []
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append({
                    "question": row["question"].strip(),
                    "expected_answer": row.get("expected_answer", "").strip(),
                    "source_urls": row.get("source_urls", "").strip(),
                })
        return samples

    def _run_pipeline(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run RAG pipeline for each sample with latency tracking."""
        from ragas import SingleTurnSample

        ragas_samples = []

        for i, sample in enumerate(samples):
            question = sample["question"]
            print(f"  [{i + 1}/{len(samples)}] Processing: {question[:50]}...")

            try:
                # Measure latency for each step
                latency = LatencyMetrics()

                # Step 1: Query processing
                t0 = time.perf_counter()
                processed_query = self.pipeline._run_query_processing(question)
                latency.query_processing_ms = (time.perf_counter() - t0) * 1000

                # Step 2: Retrieval
                t0 = time.perf_counter()
                retrieval_result = self.pipeline._run_retrieval(processed_query)
                latency.retrieval_ms = (time.perf_counter() - t0) * 1000

                # Step 3: Generation with TTFT measurement
                t0 = time.perf_counter()
                chunk_gen, build_result = self.pipeline.answer_generator.generate_stream(retrieval_result)

                # Measure TTFT
                ttft_start = time.perf_counter()
                first_token_time = None
                full_text = ""

                for chunk in chunk_gen:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    full_text += chunk

                if first_token_time:
                    latency.ttft_ms = (first_token_time - ttft_start) * 1000

                # Build result
                answer_result = build_result(full_text)
                latency.generation_ms = (time.perf_counter() - t0) * 1000

                # Step 4: Guardrails
                t0 = time.perf_counter()
                result = self.pipeline._run_output_guardrails(answer_result, retrieval_result)
                guardrails_ms = (time.perf_counter() - t0) * 1000

                # Total latency
                latency.total_ms = (
                    latency.query_processing_ms
                    + latency.retrieval_ms
                    + latency.generation_ms
                    + guardrails_ms
                )

                # Get retrieved contexts from passages
                retrieved_contexts = [
                    p.text for p in retrieval_result.passages
                ] or [result.answer]

                # Create RAGAS sample
                ragas_sample = SingleTurnSample(
                    user_input=question,
                    retrieved_contexts=retrieved_contexts,
                    response=result.answer,
                    reference=sample["expected_answer"] if sample["expected_answer"] else None,
                )

                sample["answer"] = result.answer
                sample["ragas_sample"] = ragas_sample
                sample["latency"] = latency

                print(f"    ✅ {latency.total_ms:.0f}ms (TTFT: {latency.ttft_ms:.0f}ms)")

            except Exception as e:
                print(f"    ⚠️ Error: {e}")
                sample["answer"] = ""
                sample["ragas_sample"] = None
                sample["latency"] = None

            ragas_samples.append(sample)

        return ragas_samples

    def _compute_latency_summary(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        """Compute aggregate latency metrics across all samples."""
        latencies = [
            s["latency"] for s in samples
            if s.get("latency") is not None
        ]

        if not latencies:
            return {}

        def percentile(data: list[float], p: float) -> float:
            """Calculate p-th percentile."""
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "ttft_p50_ms": round(percentile([l.ttft_ms for l in latencies], 50), 2),
            "ttft_p90_ms": round(percentile([l.ttft_ms for l in latencies], 90), 2),
            "ttft_p99_ms": round(percentile([l.ttft_ms for l in latencies], 99), 2),
            "ttft_avg_ms": round(sum(l.ttft_ms for l in latencies) / len(latencies), 2),
            "total_p50_ms": round(percentile([l.total_ms for l in latencies], 50), 2),
            "total_p90_ms": round(percentile([l.total_ms for l in latencies], 90), 2),
            "total_p99_ms": round(percentile([l.total_ms for l in latencies], 99), 2),
            "total_avg_ms": round(sum(l.total_ms for l in latencies) / len(latencies), 2),
            "query_processing_avg_ms": round(sum(l.query_processing_ms for l in latencies) / len(latencies), 2),
            "retrieval_avg_ms": round(sum(l.retrieval_ms for l in latencies) / len(latencies), 2),
            "generation_avg_ms": round(sum(l.generation_ms for l in latencies) / len(latencies), 2),
        }

    def _evaluate(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        """Run RAGAS evaluation on prepared samples."""
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        # Filter valid samples
        valid_samples = [
            s["ragas_sample"] for s in samples
            if s.get("ragas_sample") is not None
        ]

        if not valid_samples:
            print("  ⚠️ No valid samples to evaluate")
            return {}

        # Create RAGAS dataset
        dataset = EvaluationDataset(samples=valid_samples)

        # Setup LLM for RAGAS
        llm = self._create_ragas_llm()

        # Define metrics
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ]

        # Run evaluation
        print(f"  Running RAGAS evaluation on {len(valid_samples)} samples...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            show_progress=True,
        )

        return dict(result)

    def _create_ragas_llm(self):
        """Create LLM instance for RAGAS evaluation."""
        from ragas.llms.litellm_llm import LiteLLMStructuredLLM

        api_key = os.getenv(self.config.llm_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {self.config.llm_api_key_env} for RAGAS evaluation."
            )

        # LiteLLM uses OPENAI_API_KEY by default, set it
        os.environ["OPENAI_API_KEY"] = api_key

        return LiteLLMStructuredLLM(
            client=None,
            model=self.config.llm_model,
            provider="openrouter",
        )
