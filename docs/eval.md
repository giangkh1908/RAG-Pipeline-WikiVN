# Evaluation Pipeline (Phase 5) — v1

## Overview

> **v1**: RAGAS metrics + LangSmith tracing. Đánh giá offline trên eval dataset.

Phase 5 adds **RAGAS evaluation** to measure RAG pipeline quality. It runs the full pipeline (Phase 2→3→4) on a set of eval questions, then scores the results using 4 RAGAS metrics + latency metrics (TTFT, total latency).

## Quality Metrics

| Metric | Mô tả | Threshold | Cần LLM? |
|--------|--------|-----------|----------|
| **Faithfulness** | Answer có dựa trên context không? (hallucination detection) | ≥ 0.8 | ✅ |
| **Answer Relevancy** | Answer có liên quan đến question không? | ≥ 0.7 | ✅ |
| **Context Precision** | Retrieved context có chính xác không? (noise ratio) | ≥ 0.7 | ❌ |
| **Context Recall** | Retrieved context có đầy đủ không? (coverage) | ≥ 0.6 | ❌ |

## Latency Metrics

| Metric | Mô tả |
|--------|--------|
| **TTFT (Time to First Token)** | Thời gian từ khi gửi request đến token đầu tiên trả về |
| **Total Latency** | Tổng thời gian xử lý (query → retrieval → generation → guardrails) |
| **Query Processing** | Thời gian xử lý query (normalize + rewrite) |
| **Retrieval** | Thời gian tìm kiếm (dense + BM25 + RRF) |
| **Generation** | Thời gian sinh câu trả lời (LLM call) |

### Percentiles

| Percentile | Ý nghĩa |
|------------|----------|
| P50 | Median - 50% requests nhanh hơn giá trị này |
| P90 | 90% requests nhanh hơn giá trị này |
| P99 | 99% requests nhanh hơn giá trị này |
| Avg | Trung bình |

### Metric Details

#### Faithfulness
Kiểm tra answer có "bịa" thông tin không. So sánh từng claim trong answer với retrieved context.

| Context | Answer | Faithfulness |
|---------|--------|--------------|
| "Wikipedia ra đời năm 2001" | "Wikipedia ra đời năm 2001" | ✅ 1.0 |
| "Wikipedia ra đời năm 2001" | "Wikipedia ra đời năm 2003" | ❌ 0.0 |
| "Wikipedia ra đời năm 2001" | "Wikipedia ra đời năm 2001, do Jimmy Wales sáng lập" | ⚠️ 0.5 |

#### Answer Relevancy
Kiểm tra answer có trả lời đúng câu hỏi không. LLM generate câu hỏi từ answer, so sánh với question gốc.

| Question | Answer | Relevancy |
|----------|--------|-----------|
| "Thủ đô Việt Nam ở đâu?" | "Thủ đô Việt Nam là Hà Nội" | ✅ 1.0 |
| "Thủ đô Việt Nam ở đâu?" | "Việt Nam nằm ở Đông Nam Á" | ❌ 0.0 |

#### Context Precision
Kiểm tra retrieved context có "sạch" không, hay lẫn nhiều noise.

| Retrieved Context | Ground Truth | Precision |
|-------------------|--------------|-----------|
| [Passage 1: đúng, Passage 2: đúng] | 2 passages | ✅ 1.0 |
| [Passage 1: đúng, Passage 2: sai, Passage 3: sai] | 1 passage | ❌ 0.33 |

#### Context Recall
Kiểm tra retrieved context có chứa đủ thông tin cần thiết không.

| Ground Truth cần | Retrieved Context | Recall |
|------------------|-------------------|--------|
| "2001" + "Jimmy Wales" | Cả 2 facts | ✅ 1.0 |
| "2001" + "Jimmy Wales" | Chỉ có "2001" | ⚠️ 0.5 |

## Architecture

```
documents/eval.csv
    │
    ▼
┌─────────────────┐
│  EvalRunner      │  Load CSV → run pipeline → prepare RAGAS samples
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ AnswerPipeline   │  Phase 2→3→4 cho mỗi sample
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RAGAS evaluate() │  Faithfulness + AnswerRelevancy + ContextPrecision + ContextRecall
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   EvalReport     │  JSON + Markdown export
└─────────────────┘
```

## Eval Dataset Format

CSV file với 3 columns:

```csv
question,expected_answer,source_urls
"Wikipedia là gì?","Wikipedia là bách khoa toàn thư mở...","https://vi.wikipedia.org/wiki/Wikipedia"
```

| Column | Required | Mô tả |
|--------|----------|-------|
| `question` | ✅ | Câu hỏi eval |
| `expected_answer` | ✅ | Ground truth answer |
| `source_urls` | ❌ | URL nguồn tham khảo |

## CLI Usage

```bash
# Chạy eval với dataset mặc định
python -m rag_pipeline.main eval

# Chạy eval với custom dataset
python -m rag_pipeline.main eval --dataset documents/eval.csv --limit 10

# Specify output file
python -m rag_pipeline.main eval --output eval_report.json

# Use InMemory instead of Qdrant (for testing)
python -m rag_pipeline.main eval --no-qdrant --limit 5
```

### Output

Console output:
```
  [1/5] Processing: Wikipedia là gì?...
    ✅ 2500ms (TTFT: 350ms)
  [2/5] Processing: Thủ đô Việt Nam ở đâu?...
    ✅ 2200ms (TTFT: 320ms)
  ...

  📊 Evaluation Results:
  ============================================================

    Quality Metrics:
    --------------------------------------------------------
      faithfulness.................. 0.8500  (threshold: 0.8) ✅
      answer_relevancy.............. 0.7800  (threshold: 0.7) ✅
      context_precision............. 0.7200  (threshold: 0.7) ✅
      context_recall................ 0.6500  (threshold: 0.6) ✅

    Latency Metrics:
    --------------------------------------------------------
      TTFT (P50)..................     320ms
      TTFT (P90)..................     450ms
      TTFT (avg)..................     340ms
      Total (P50)..................    2200ms
      Total (P90)..................    2800ms
      Total (avg)..................    2350ms
      Query Processing (avg).......     650ms
      Retrieval (avg)..............     190ms
      Generation (avg).............    1500ms

  ============================================================
    Overall: ✅ PASS
    Samples: 5

  📄 Report saved to: eval_report.json
```

### Report Files

**JSON report** (`eval_report.json`):
```json
{
  "scores": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.78,
    "context_precision": 0.72,
    "context_recall": 0.65
  },
  "latency": {
    "ttft_p50_ms": 320.5,
    "ttft_p90_ms": 450.2,
    "ttft_p99_ms": 520.1,
    "ttft_avg_ms": 340.3,
    "total_p50_ms": 2200.5,
    "total_p90_ms": 2800.2,
    "total_p99_ms": 3100.1,
    "total_avg_ms": 2350.3,
    "query_processing_avg_ms": 650.2,
    "retrieval_avg_ms": 190.5,
    "generation_avg_ms": 1500.1
  },
  "thresholds": {
    "faithfulness": 0.8,
    "answer_relevancy": 0.7,
    "context_precision": 0.7,
    "context_recall": 0.6
  },
  "passed": true,
  "sample_count": 5,
  "samples": [
    {
      "question": "Wikipedia là gì?",
      "answer": "Wikipedia là bách khoa toàn thư mở...",
      "expected_answer": "Wikipedia là bách khoa toàn thư mở...",
      "scores": {"faithfulness": 0.9, "answer_relevancy": 0.85},
      "latency": {
        "query_processing_ms": 650.2,
        "retrieval_ms": 185.3,
        "ttft_ms": 320.5,
        "generation_ms": 1450.2,
        "total_ms": 2300.5
      }
    }
  ]
}
```

**Markdown report** (`eval_report.md`):
```markdown
# RAG Evaluation Report

## Metric Scores

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| faithfulness | 0.85 | 0.8 | ✅ |
| answer_relevancy | 0.78 | 0.7 | ✅ |
...

**Overall: ✅ PASS**
```

## Python API

```python
from pathlib import Path
from rag_pipeline.config import EvalConfig
from rag_pipeline.eval.runner import EvalRunner
from rag_pipeline.main import build_generation_pipeline, build_query_pipeline, build_retrieval_pipeline
from rag_pipeline.config import QueryConfig

# Build pipeline
pipeline = build_generation_pipeline(
    retrieval_pipeline=build_retrieval_pipeline(use_qdrant=True),
    query_pipeline=build_query_pipeline(QueryConfig(), use_llm=True),
)

# Run evaluation
config = EvalConfig(eval_dataset_path=Path("documents/eval.csv"))
runner = EvalRunner(pipeline=pipeline, config=config)
report = runner.run(limit=10)

# Print results
report.print_summary()

# Export
report.to_json(Path("eval_report.json"))
report.to_markdown(Path("eval_report.md"))
```

## Configuration

### EvalConfig (`config.py`)

| Field | Default | Description |
|-------|---------|-------------|
| `eval_dataset_path` | `documents/eval.csv` | Path to eval CSV |
| `llm_model` | `deepseek/deepseek-v4-flash` | LLM model for RAGAS (faithfulness, answer_relevancy) |
| `llm_api_base` | `https://openrouter.ai/api/v1` | OpenRouter API base |
| `llm_api_key_env` | `OPENROUTER_API_KEY` | Env var for API key |
| `faithfulness_threshold` | 0.8 | Min faithfulness score |
| `answer_relevance_threshold` | 0.7 | Min answer relevancy score |
| `context_precision_threshold` | 0.7 | Min context precision score |
| `context_recall_threshold` | 0.6 | Min context recall score |

## Dependencies

- `ragas` — RAGAS evaluation framework
- `litellm` — LLM provider abstraction (used by RAGAS)
- `openrouter` API key — for LLM-as-judge (faithfulness, answer_relevancy)

## Cost

Eval costs ~2 LLM calls per sample (faithfulness + answer_relevancy):

| Samples | LLM Calls | Estimated Cost |
|---------|-----------|----------------|
| 10 | ~20 | ~$0.01 |
| 50 | ~100 | ~$0.05 |
| 100 | ~200 | ~$0.10 |

Context precision and context recall are computed without LLM (free).

## Tests

8 tests covering all components:

| Category | Tests | Description |
|----------|-------|-------------|
| EvalReport | 5 | JSON export, Markdown export, threshold pass/fail, empty report |
| EvalConfig | 2 | Default config, custom config |
| EvalDataset | 1 | Load CSV format |

```bash
python -m pytest tests/test_eval.py -v
```

## Dependencies

- `ragas` — RAGAS evaluation framework
- `litellm` — LLM provider abstraction (used by RAGAS)
- `openrouter` API key — for LLM-as-judge (faithfulness, answer_relevancy)
- `time` — built-in module for latency measurement (no extra install)
