# Đánh giá RAG (Eval)

Tài liệu này mô tả cách chạy đánh giá hệ thống RAG, bao gồm sinh test case,
chạy eval, và đọc kết quả.

---

## Bộ test hiện tại

File: `data/eval/test_suite.jsonl`

Gồm **25 câu hỏi** chia đều thành 5 category:

| Category | Số câu | Mô tả |
|----------|--------|-------|
| `easy` | 5 | Hỏi trực tiếp, đáp án rõ ràng trong corpus |
| `hard` | 5 | Dùng từ đồng nghĩa, cần suy luận nhẹ |
| `out-of-scope` | 5 | Không liên quan đến du lịch Việt Nam |
| `missing` | 5 | Về Việt Nam nhưng không có trong corpus |
| `ambiguous` | 5 | Mơ hồ, có thể hiểu nhiều cách |

Mỗi câu có:
- `id`, `category`, `question`
- `expected_behavior`: `answerable`, `refuse`, hoặc `ambiguous`
- `expected_keywords`: từ khóa cần xuất hiện trong câu trả lờ i đúng
- `relevant_context`: đoạn context từ corpus chứa đáp án
- `notes`: ghi chú

---

## Sinh test case mới

Nếu muốn regenerate bộ test (ví dụ sau khi cập nhật corpus):

```bash
$env:PYTHONIOENCODING="utf-8"
python scripts/generate_eval_suite.py
```

Script đọc `documents/vietnam_tourism_v2.json` và `data/eval/queries.jsonl`,
sau đó gọi `deepseek/deepseek-v4-flash` qua OpenRouter để tạo 25 test case.

Output: `data/eval/test_suite.jsonl`

> **Lưu ý:** Nên review test case sau khi generate vì LLM có thể tạo câu
> không chính xác hoặc expected_keywords không khớp corpus.

---

## Chạy eval

```bash
$env:PYTHONIOENCODING="utf-8"
python scripts/eval_rag.py
```

Script sẽ:
1. Load 25 test cases
2. Chạy mỗi câu qua full RAG pipeline (rewrite → retrieve → generate)
3. Tính retrieval metrics: recall@k, MRR
4. Gọi LLM judge (`deepseek/deepseek-v4-flash`) chấm điểm câu trả lờ i
5. Xuất báo cáo JSON + CSV vào `data/eval/`

---

## Metrics

### Retrieval metrics

| Metric | Ý nghĩa |
|--------|---------|
| `recall@k` | % câu có ít nhất 1 chunk chứa expected keywords trong top-k |
| `mrr` | Mean Reciprocal Rank của chunk đầu tiên chứa đáp án |

### LLM judge metrics

| Metric | Ý nghĩa |
|--------|---------|
| `correctness` | 1-5, câu trả lờ i có đúng không |
| `relevance` | 1-5, có liên quan đến câu hỏi không |
| `hallucination` | true/false, có bịa thông tin không |
| `refusal_appropriate` | yes/no/n/a, từ chối có hợp lý không (out-of-scope/missing) |

---

## Kết quả mẫu

```text
Total questions: 25
Hallucination rate: 4.00%
Avg latency: 10256ms
Mean correctness: 3.48/5
Mean relevance: 3.44/5
Mean recall@k: 44.00%
Mean MRR: 0.370

By category:
Category         Count   Correct  Relevance   Recall   Halluc   Refusal
--------------------------------------------------------------------------------
ambiguous            5     4.00      5.00   60.0%   20.0%      n/a
easy                 5     4.60      5.00  100.0%    0.0%      n/a
hard                 5     3.20      3.20   60.0%    0.0%      n/a
missing              5     2.40      2.20    0.0%    0.0%      67%
out-of-scope         5     3.20      1.80    0.0%    0.0%     100%
```

---

## Giải thích kết quả mẫu

- **easy**: Hệ thống trả lờ i rất tốt, recall 100%, correctness cao.
- **hard**: Một số câu khó bị miss retrieval (recall 60%), cần cải thiện
  embedding hoặc rewrite.
- **out-of-scope**: Từ chối hợp lý 100%, nhưng correctness/relevance vẫn bị
  judge chấm thấp vì câu trả lờ i ngắn.
- **missing**: Chỉ 67% từ chối hợp lý, có nghĩa là 1/3 câu missing hệ thống
  vẫn cố trả lờ i dựa trên thông tin không liên quan.
- **ambiguous**: Relevance cao nhưng có 1 câu bị hallucination.

---

## Cải thiện dựa trên eval

Dựa vào kết quả, có thể tối ưu:

1. **Tăng hard recall**: Thử model embedding khác hoặc tăng số chunks retrieve.
2. **Giảm hallucination trên ambiguous**: Thêm guardrail hoặc prompt yêu cầu
   không trả lờ i khi không chắc chắn.
3. **Cải thiện refusal cho missing**: Điều chỉnh prompt generation để từ chối
   rõ ràng hơn khi context không đủ.
4. **Giảm latency**: Xem `docs/latency.md`.
