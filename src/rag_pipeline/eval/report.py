"""Eval report — stores and exports evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalReport:
    """Stores evaluation results with per-metric scores, latency, and per-sample breakdown."""

    scores: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    latency_summary: dict[str, float] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Export report to JSON file.

        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "latency": self.latency_summary,
            "thresholds": self.thresholds,
            "passed": self._check_thresholds(),
            "sample_count": len(self.samples),
            "samples": self.samples,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def to_markdown(self, path: Path) -> None:
        """Export report to Markdown file.

        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# RAG Evaluation Report\n"]

        # Scores table
        lines.append("## Quality Metrics\n")
        lines.append("| Metric | Score | Threshold | Pass |")
        lines.append("|--------|-------|-----------|------|")
        for metric, score in self.scores.items():
            threshold = self.thresholds.get(metric, 0.0)
            passed = score >= threshold
            status = "✅" if passed else "❌"
            lines.append(f"| {metric} | {score:.4f} | {threshold} | {status} |")

        # Latency table
        if self.latency_summary:
            lines.append("\n## Latency Metrics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| TTFT (P50) | {self.latency_summary.get('ttft_p50_ms', 0):.0f}ms |")
            lines.append(f"| TTFT (P90) | {self.latency_summary.get('ttft_p90_ms', 0):.0f}ms |")
            lines.append(f"| TTFT (P99) | {self.latency_summary.get('ttft_p99_ms', 0):.0f}ms |")
            lines.append(f"| TTFT (avg) | {self.latency_summary.get('ttft_avg_ms', 0):.0f}ms |")
            lines.append(f"| Total (P50) | {self.latency_summary.get('total_p50_ms', 0):.0f}ms |")
            lines.append(f"| Total (P90) | {self.latency_summary.get('total_p90_ms', 0):.0f}ms |")
            lines.append(f"| Total (avg) | {self.latency_summary.get('total_avg_ms', 0):.0f}ms |")
            lines.append(f"| Query Processing (avg) | {self.latency_summary.get('query_processing_avg_ms', 0):.0f}ms |")
            lines.append(f"| Retrieval (avg) | {self.latency_summary.get('retrieval_avg_ms', 0):.0f}ms |")
            lines.append(f"| Generation (avg) | {self.latency_summary.get('generation_avg_ms', 0):.0f}ms |")

        # Summary
        passed = self._check_thresholds()
        lines.append(f"\n**Overall: {'✅ PASS' if passed else '❌ FAIL'}**")
        lines.append(f"\nSamples evaluated: {len(self.samples)}")

        # Per-sample breakdown
        if self.samples:
            lines.append("\n## Per-Sample Results\n")
            for i, sample in enumerate(self.samples[:10]):  # Show first 10
                lines.append(f"### Sample {i + 1}")
                lines.append(f"- **Question:** {sample.get('question', 'N/A')}")
                lines.append(f"- **Answer:** {sample.get('answer', 'N/A')[:100]}...")
                for metric in self.scores:
                    val = sample.get("scores", {}).get(metric, "N/A")
                    lines.append(f"- **{metric}:** {val}")
                latency = sample.get("latency", {})
                if latency:
                    lines.append(f"- **TTFT:** {latency.get('ttft_ms', 0):.0f}ms")
                    lines.append(f"- **Total:** {latency.get('total_ms', 0):.0f}ms")
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print("\n📊 Evaluation Results:")
        print("=" * 60)

        # Quality metrics
        print("\n  Quality Metrics:")
        print("  " + "-" * 56)
        for metric, score in self.scores.items():
            threshold = self.thresholds.get(metric, 0.0)
            passed = score >= threshold
            status = "✅" if passed else "❌"
            print(f"    {metric:.<30} {score:.4f}  (threshold: {threshold}) {status}")

        # Latency metrics
        if self.latency_summary:
            print("\n  Latency Metrics:")
            print("  " + "-" * 56)
            print(f"    {'TTFT (P50)':.<30} {self.latency_summary.get('ttft_p50_ms', 0):>8.0f}ms")
            print(f"    {'TTFT (P90)':.<30} {self.latency_summary.get('ttft_p90_ms', 0):>8.0f}ms")
            print(f"    {'TTFT (avg)':.<30} {self.latency_summary.get('ttft_avg_ms', 0):>8.0f}ms")
            print(f"    {'Total (P50)':.<30} {self.latency_summary.get('total_p50_ms', 0):>8.0f}ms")
            print(f"    {'Total (P90)':.<30} {self.latency_summary.get('total_p90_ms', 0):>8.0f}ms")
            print(f"    {'Total (avg)':.<30} {self.latency_summary.get('total_avg_ms', 0):>8.0f}ms")
            print(f"    {'Query Processing (avg)':.<30} {self.latency_summary.get('query_processing_avg_ms', 0):>8.0f}ms")
            print(f"    {'Retrieval (avg)':.<30} {self.latency_summary.get('retrieval_avg_ms', 0):>8.0f}ms")
            print(f"    {'Generation (avg)':.<30} {self.latency_summary.get('generation_avg_ms', 0):>8.0f}ms")

        print("\n" + "=" * 60)
        passed = self._check_thresholds()
        print(f"  Overall: {'✅ PASS' if passed else '❌ FAIL'}")
        print(f"  Samples: {len(self.samples)}")

    def _check_thresholds(self) -> bool:
        """Check if all scores meet their thresholds."""
        for metric, score in self.scores.items():
            threshold = self.thresholds.get(metric, 0.0)
            if score < threshold:
                return False
        return True
