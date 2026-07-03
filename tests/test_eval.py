"""Tests for Phase 5: RAGAS evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rag_pipeline.config import EvalConfig
from rag_pipeline.eval.report import EvalReport


class TestEvalReport:
    def test_json_export(self) -> None:
        report = EvalReport(
            scores={"faithfulness": 0.85, "answer_relevancy": 0.78},
            samples=[{"question": "Test?", "answer": "Answer", "scores": {"faithfulness": 0.85}}],
            thresholds={"faithfulness": 0.8, "answer_relevancy": 0.7},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.to_json(path)

            assert path.exists()
            import json
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["scores"]["faithfulness"] == 0.85
            assert data["passed"] is True
            assert data["sample_count"] == 1

    def test_markdown_export(self) -> None:
        report = EvalReport(
            scores={"faithfulness": 0.85, "answer_relevancy": 0.65},
            samples=[{"question": "Test?", "answer": "Answer", "scores": {}}],
            thresholds={"faithfulness": 0.8, "answer_relevancy": 0.7},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.md"
            report.to_markdown(path)

            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "faithfulness" in content
            assert "answer_relevancy" in content
            assert "PASS" in content or "FAIL" in content

    def test_threshold_pass(self) -> None:
        report = EvalReport(
            scores={"faithfulness": 0.9, "answer_relevancy": 0.8},
            thresholds={"faithfulness": 0.8, "answer_relevancy": 0.7},
        )
        assert report._check_thresholds() is True

    def test_threshold_fail(self) -> None:
        report = EvalReport(
            scores={"faithfulness": 0.5, "answer_relevancy": 0.8},
            thresholds={"faithfulness": 0.8, "answer_relevancy": 0.7},
        )
        assert report._check_thresholds() is False

    def test_empty_report(self) -> None:
        report = EvalReport()
        assert report.scores == {}
        assert report.samples == []
        assert report._check_thresholds() is True  # No thresholds to check


class TestEvalConfig:
    def test_default_config(self) -> None:
        config = EvalConfig()
        assert config.eval_dataset_path == Path("documents/eval.csv")
        assert config.faithfulness_threshold == 0.8
        assert config.answer_relevance_threshold == 0.7
        assert config.context_precision_threshold == 0.7
        assert config.context_recall_threshold == 0.6

    def test_custom_config(self) -> None:
        config = EvalConfig(
            eval_dataset_path=Path("custom/eval.csv"),
            faithfulness_threshold=0.9,
        )
        assert config.eval_dataset_path == Path("custom/eval.csv")
        assert config.faithfulness_threshold == 0.9


class TestEvalDataset:
    def test_load_eval_csv(self) -> None:
        """Test loading the eval CSV file."""
        import csv

        path = Path("documents/eval.csv")
        assert path.exists()

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        assert "question" in rows[0]
        assert "expected_answer" in rows[0]
        assert "source_urls" in rows[0]
