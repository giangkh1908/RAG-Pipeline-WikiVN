"""Generate a diverse evaluation test suite using an LLM.

Usage:
    $env:PYTHONIOENCODING="utf-8"; python scripts/generate_eval_suite.py

The script reads the Vietnam Tourism dataset and existing eval queries, then
asks an LLM to generate 25 test cases across 5 categories:
    - easy: directly answerable from the corpus
    - hard: paraphrased / requires light inference
    - out-of-scope: unrelated to Vietnam tourism
    - missing: about Vietnam but not in the corpus
    - ambiguous: vague or multi-intent

Output is written to data/eval/test_suite.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DATASET_PATH = Path("documents/vietnam_tourism_v2.json")
QUERIES_PATH = Path("data/eval/queries.jsonl")
OUTPUT_PATH = Path("data/eval/test_suite.jsonl")

JUDGE_MODEL = "deepseek/deepseek-v4-flash"
API_BASE = "https://openrouter.ai/api/v1"


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set OPENROUTER_API_KEY environment variable")
    return key


def load_corpus_sample(path: Path, max_topics: int = 5) -> str:
    """Load a small sample of the corpus to keep the prompt short."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    topics = data.get("data", [])[:max_topics]
    sample: list[dict[str, Any]] = []
    for topic in topics:
        paragraphs = topic.get("paragraphs", [])
        if not paragraphs:
            continue
        # Take first paragraph context and first QA
        para = paragraphs[0]
        sample.append(
            {
                "title": topic.get("title", ""),
                "context": para.get("context", "")[:800],
                "example_question": para.get("qas", [{}])[0].get("question", ""),
                "example_answer": para.get("qas", [{}])[0].get("answers", [{}])[0].get("text", ""),
            }
        )
    return json.dumps(sample, ensure_ascii=False, indent=2)


def load_existing_queries(path: Path, limit: int = 10) -> str:
    """Load a few existing eval queries as examples."""
    if not path.exists():
        return ""
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
            if len(queries) >= limit:
                break
    return json.dumps(queries, ensure_ascii=False, indent=2)


def build_prompt(corpus_sample: str, existing_queries: str) -> str:
    return f"""Bạn là chuyên gia tạo bộ test cho hệ thống RAG du lịch Việt Nam.

Dưới đây là một phần corpus và một số câu hỏi mẫu:

--- CORPUS SAMPLE ---
{corpus_sample}

--- EXAMPLE QUESTIONS ---
{existing_queries}

Hãy tạo chính xác 25 câu hỏi thuộc 5 category sau (mỗi category 5 câu):

1. **easy**: Hỏi trực tiếp, đáp án nằm rõ ràng trong corpus.
2. **hard**: Hỏi lắt léo, dùng từ đồng nghĩa, cần suy luận nhẹ.
3. **out-of-scope**: Không liên quan đến du lịch Việt Nam.
4. **missing**: Về Việt Nam nhưng thông tin không có trong corpus.
5. **ambiguous**: Mơ hồ, có thể hiểu theo nhiều cách.

Yêu cầu output:
- Trả về một JSON array duy nhất.
- Mỗi phần tử có các trường: id, category, question, expected_behavior, expected_keywords, relevant_context, notes.
- expected_behavior là một trong: answerable, refuse, ambiguous.
- expected_keywords là list các từ khóa cần xuất hiện trong câu trả lờ i đúng.
- relevant_context: đoạn context từ corpus chứa đáp án (nếu có), hoặc chuỗi rỗng.
- notes: giải thích ngắn tại sao chọn câu này.

Chỉ trả về JSON array, không thêm giải thích ngoài.
"""


def call_llm(prompt: str) -> str:
    client = httpx.Client(
        base_url=API_BASE,
        timeout=120.0,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://rag-pipeline.local",
            "X-Title": "RAG Pipeline Eval",
        },
    )
    response = client.post(
        "/chat/completions",
        json={
            "model": JUDGE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that returns valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 4000,
        },
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def parse_json_response(content: str) -> list[dict[str, Any]]:
    """Extract JSON array from LLM response."""
    original = content
    content = content.strip()
    # Remove markdown code fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        dump_path = Path("data/eval/_raw_llm_response.txt")
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(original, encoding="utf-8")
        print(f"[ERROR] Raw response saved to {dump_path} for inspection")
        raise


def validate_test_suite(items: list[dict[str, Any]]) -> None:
    required_fields = {"id", "category", "question", "expected_behavior", "expected_keywords"}
    categories = {"easy", "hard", "out-of-scope", "missing", "ambiguous"}
    counts: dict[str, int] = {}

    for item in items:
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"Test case {item.get('id')} missing fields: {missing}")
        cat = item["category"]
        if cat not in categories:
            raise ValueError(f"Invalid category '{cat}' in {item.get('id')}")
        counts[cat] = counts.get(cat, 0) + 1

    for cat in categories:
        if counts.get(cat, 0) != 5:
            raise ValueError(f"Category '{cat}' should have 5 items, got {counts.get(cat, 0)}")


def main() -> int:
    load_dotenv()

    print("Loading corpus sample and existing queries...")
    corpus_sample = load_corpus_sample(DATASET_PATH)
    existing_queries = load_existing_queries(QUERIES_PATH)

    print(f"Calling {JUDGE_MODEL} to generate test suite...")
    prompt = build_prompt(corpus_sample, existing_queries)
    content = call_llm(prompt)

    print("Parsing response...")
    items = parse_json_response(content)
    validate_test_suite(items)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(items)} test cases -> {OUTPUT_PATH}")
    for cat in ["easy", "hard", "out-of-scope", "missing", "ambiguous"]:
        count = sum(1 for item in items if item["category"] == cat)
        print(f"  {cat}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
