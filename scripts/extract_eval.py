"""Extract evaluation queries from a SQuAD-style JSON dataset.

Usage:
    python scripts/extract_eval.py documents/vietnam_tourism_v2.json data/eval/queries.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def extract_queries(input_path: str, output_path: str) -> int:
    with Path(input_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for topic in data.get("data", []):
        title = topic.get("title", "")
        # Pick first paragraph and first QA for each title
        paragraphs = topic.get("paragraphs", [])
        if not paragraphs:
            continue
        paragraph = paragraphs[0]
        context = paragraph.get("context", "")
        qas = paragraph.get("qas", [])
        if not qas:
            continue
        qa = qas[0]
        answers = qa.get("answers", [])
        answer_text = answers[0].get("text", "") if answers else ""
        queries.append(
            {
                "id": qa.get("id", ""),
                "title": title,
                "question": qa.get("question", ""),
                "answer": answer_text,
                "context": context,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    return len(queries)


def compute_recall(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved[:k])
    hits = len(retrieved_set & set(relevant))
    return hits / len(relevant)


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "documents/vietnam_tourism_v2.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/eval/queries.jsonl"

    count = extract_queries(input_file, output_file)
    print(f"Extracted {count} eval queries to {output_file}")
