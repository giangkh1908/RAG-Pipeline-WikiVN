# Generation Pipeline (Phase 4) — v1

## Overview

Phase 4 adds the **answer generation** layer on top of the existing retrieval pipeline. It takes `RetrievalResult` (from Phase 3) and produces a structured `AnswerResult` with citations, confidence scores, and output guardrail checks.

> **v1**: Gọi LLM trực tiếp (OpenRouter API). V2 sẽ dùng tool calling + agent orchestration.

## Architecture

```
User Question
     │
     ▼
┌─────────────────┐
│  QueryPipeline   │  Phase 2: guardrails → normalize → rewrite
└────────┬────────┘
         │ ProcessedQuery
         ▼
┌─────────────────┐
│ RetrievalPipeline│  Phase 3: dense + BM25 → RRF → rerank
└────────┬────────┘
         │ RetrievalResult
         ▼
┌─────────────────┐
│ PromptBuilder    │  Build system + user messages for LLM
└────────┬────────┘
         │ messages
         ▼
┌─────────────────┐
│ AnswerGenerator  │  LLM chat_json → parse AnswerResult
└────────┬────────┘
         │ AnswerResult (raw)
         ▼
┌─────────────────┐
│ OutputGuardrails │  hallucination + safety + quality check
└────────┬────────┘
         │ AnswerResult (checked)
         ▼
    Final Answer + Citations
```

## Components

### PromptBuilder (`generation/prompt_builder.py`)

Builds structured chat messages for the LLM with:
- **System message**: Instructions for Vietnamese answer generation, JSON output format, citation rules
- **User message**: Formatted passages with `[1], [2]...` numbering + the user question

### AnswerGenerator (`generation/answer_generator.py`)

Calls the LLM and parses the response:
1. Builds prompt via `PromptBuilder`
2. Calls `llm_client.chat_json()` for structured output
3. Parses JSON response into `AnswerResult` with `Citation` objects
4. Maps `source_index` from LLM response back to original passages
5. Fallback: wraps raw text if JSON parsing fails

### OutputGuardrails (`generation/output_guardrails.py`)

Three checks on generated answers:

| Check | What it does | Flag |
|-------|-------------|------|
| **Hallucination** | Verifies claims are backed by retrieved passages | `hallucination_no_context`, `hallucination_high_unbacked_ratio` |
| **Safety** | Detects unsafe content (weapons, hacking, etc.) | `unsafe_content_detected` |
| **Quality** | Checks answer length, citation count | `answer_too_long`, `answer_too_short`, `insufficient_citations` |

Each flag reduces the confidence score by 0.2.

## Data Models

### Citation (`models.py`)

```python
@dataclass(slots=True)
class Citation:
    claim: str          # Claim in the answer
    chunk_id: str       # Source passage ID
    doc_id: str         # Source document ID
    title: str          # Article title
    source_url: str     # Wikipedia URL
    confidence: float   # Citation confidence (0-1)
```

### AnswerResult (`models.py`)

```python
@dataclass(slots=True)
class AnswerResult:
    question: str                   # Original question
    answer: str                     # LLM-generated answer
    citations: list[Citation]       # Source citations
    confidence: float               # Overall confidence (0-1)
    passages_used: int              # Number of passages used
    metadata: dict[str, Any]        # Guardrail flags, parse mode, etc.
```

## Configuration

### GenerationConfig (`config.py`)

| Field | Default | Description |
|-------|---------|-------------|
| `max_answer_tokens` | 1024 | Max tokens for LLM response |
| `temperature` | 0.1 | LLM temperature |
| `prompt_template` | `"vietnamese_rag"` | Template name |

### OutputGuardrailsConfig (`config.py`)

| Field | Default | Description |
|-------|---------|-------------|
| `enable_hallucination_check` | True | Enable hallucination detection |
| `enable_safety_check` | True | Enable unsafe content detection |
| `enable_quality_check` | True | Enable quality checks |
| `min_answer_confidence` | 0.3 | Minimum confidence threshold |
| `min_citations` | 1 | Minimum citation count |
| `max_answer_length` | 2000 | Max answer character length |

## CLI Usage

```bash
# Full RAG pipeline: question → answer with citations
python -m rag_pipeline.main ask --question "Wikipedia là gì?"

# With options
python -m rag_pipeline.main ask \
    --question "Ai là người sáng lập Wikipedia?" \
    --no-qdrant \      # Use InMemory instead of Qdrant
    --rerank \          # Use re-ranker
    --no-llm            # Disable LLM query rewrite

# Streaming mode: tokens appear in real-time (lower perceived latency)
python -m rag_pipeline.main ask \
    --question "Lịch sử Wikipedia?" \
    --text --stream
```

### Output Format

```json
{
  "question": "Wikipedia là gì?",
  "answer": "Wikipedia là bách khoa toàn thư mở [1], được viết bởi các tình nguyện viên trên toàn thế giới.",
  "citations": [
    {
      "claim": "Wikipedia là bách khoa toàn thư mở",
      "title": "Wikipedia",
      "source_url": "https://vi.wikipedia.org/wiki/Wikipedia",
      "confidence": 0.8
    }
  ],
  "confidence": 0.6,
  "passages_used": 5,
  "metadata": {
    "parse_mode": "structured_json",
    "guardrail_flags": [],
    "guardrail_checked": true
  }
}
```

## Python API

```python
from rag_pipeline.main import ask

# Simple usage
result = ask("Wikipedia là gì?")
print(result.answer)
print(result.citations)
print(result.confidence)

# With options
result = ask(
    "Ai sáng lập Wikipedia?",
    use_qdrant=True,
    use_reranker=True,
    use_llm=True,
)

# Streaming mode
from rag_pipeline.main import build_ask_pipeline

pipeline = build_ask_pipeline()
processed = pipeline._run_query_processing("Wikipedia là gì?")
retrieval = pipeline._run_retrieval(processed)

chunk_gen, build_result = pipeline.answer_generator.generate_stream(retrieval)

for chunk in chunk_gen:
    print(chunk, end="", flush=True)

result = build_result(full_text)
```

## Tests

18 tests covering all components:

| Category | Tests | Description |
|----------|-------|-------------|
| PromptBuilder | 5 | Message format, passages, questions, empty passages |
| AnswerGenerator | 5 | Basic generation, citations, confidence, fallback |
| OutputGuardrails | 6 | Safe answers, hallucination, safety, quality |
| AnswerPipeline | 2 | End-to-end, citation mapping |

All tests use `DeterministicTestLLM` — no API keys required.

```bash
python -m pytest tests/test_generation.py -v
```

## Dependencies

- `rag_pipeline.indexing.llm_client` — LLMClient Protocol, OpenRouterLLMClient, DeterministicTestLLM
- `rag_pipeline.generation.prompt_builder` — PromptBuilder
- `rag_pipeline.generation.answer_generator` — AnswerGenerator
- `rag_pipeline.generation.output_guardrails` — OutputGuardrails
- `rag_pipeline.pipelines.answer_pipeline` — AnswerPipeline
