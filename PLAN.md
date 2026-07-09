# PLAN.md

RAG pipeline cho Wikipedia tiếng Việt. V1 đã xong end-to-end. V2 đào sâu chất lượng.

---

## V1 — Đã hoàn thành ✅

**Pipeline:** query → normalize → guardrails → rewrite → hybrid retrieve (dense + BM25 + RRF + Cohere rerank) → generate (cited answer) → output guardrails → SSE stream.

| Thành phần | Ghi chú |
|------------|---------|
| Ingest | 433K vectors từ 1.1M Wikipedia articles → Qdrant |
| Query processing | Normalize, guardrails, LLM rewrite |
| Retrieval | Dense (Qdrant) + BM25 (rank-bm25) + RRF + Cohere rerank |
| Generation | PromptBuilder → OpenRouter DeepSeek → citation injection → guardrails |
| Eval | RAGAS 4 metrics + LangSmith tracing + latency P50/P90/P99 |
| API | FastAPI + SSE streaming + conversation memory |
| Frontend | React 19 + Vite + Tailwind, ChatGPT-style UI |
| Deploy | Docker + GHCR + GitHub Actions CD → VPS + Cloudflare + SSL |
| Tests | 113 tests pass |

**Demo:** https://wikivn.top

---

## V2 — Hướng thực thi

V2 không thêm tính năng. V2 đào sâu từng khâu để đi từ "chạy được" sang "đáng tin".

### 7 trụ cột

1. **Eval nâng cao** — answerable/unanswerable test set + confusion matrix + hallucination classification + coverage. Biết chính xác hệ thống sai ở đâu, sai kiểu gì.
2. **Hiểu sâu retrieval** — phân tích failure mode của dense vs BM25 vs hybrid vs rerank. Khi nào cái nào thắng, khi nào cùng fail.
3. **Chunking strategy** — structure-aware (parse heading/list/table), contextual prefix, semantic overlap, parent retrieval. Đo context recall trước/sau.
4. **Claim verification** — tách answer thành claims → NLI verify từng claim (supported/contradicted/unsupported). Citation chỉ là "chỉ tay", verify mới kiểm tra thật.
5. **Abstention policy** — evidence threshold + confidence threshold + out-of-domain detection. Vẽ trade-off curve: abstain rate vs accuracy.
6. **Agent graph / CRAG** — route → retrieve → grade → refine → generate → verify → finalize. Graph cố định, mỗi node test được độc lập. Không phải agent demo gọi tool lung tung.
7. **Scale benchmark** — index size, build time, recall@K theo scale, memory. Luôn báo cáo latency cùng quality, không để latency tốt đánh lừa.

### Lộ trình

```
Phase A (2 tuần):  Eval nâng cao + Scale benchmark    → biết đang đứng ở đâu
Phase B (2 tuần):  Chunking strategy + Hiểu sâu retrieval → cải thiện từ gốc
Phase C (2 tuần):  Claim verification + Abstention     → production readiness
Phase D (3 tuần):  Agent graph / CRAG                  → kiến trúc agent có kiểm soát
```

Mỗi phase có deliverable đo được. Không code trước khi có baseline numbers từ phase trước.
