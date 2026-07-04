"""LLM-based query rewrite for better Wikipedia retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_pipeline.indexing.llm_client import LLMClient

REWRITE_PROMPT = """Bạn là trợ lý tìm kiếm Wikipedia tiếng Việt. Nhiệm vụ: chuẩn hóa câu hỏi để tối ưu tìm kiếm.

{history_section}Cho câu hỏi sau:
"{query}"

QUY TẮC QUAN TRỌNG:
- Nếu có ngữ cảnh hội thoại, câu hỏi hiện tại LUÔN liên quan đến chủ đề đang bàn.
- Thay TẤT CẢ đại từ, tham chiếu mập mờ bằng danh từ cụ thể từ ngữ cảnh.
- Nếu câu hỏi không nêu rõ chủ đề (ví dụ: "năm mới nhất", "bao nhiêu người", "thông tin đó"), hãy MẶC ĐỊNH nối tiếp chủ đề từ câu trước.

Trả về JSON với các trường:
1. "normalized_query": Câu hỏi đã chuẩn hóa. Thay đại từ bằng danh từ cụ thể. Ví dụ: "có thông tin năm mới nhất ko?" khi đang bàn về Việt Nam → "có thông tin dân số việt nam năm mới nhất ko?"
2. "rewrite_query": Viết lại câu hỏi đầy đủ, rõ ràng. Thêm chủ đề từ ngữ cảnh nếu câu hỏi thiếu. KHÔNG thêm câu trả lời.
3. "bm25_query": Từ khóa chính để tìm kiếm. LUÔN bao gồm chủ đề chính từ ngữ cảnh. Ví dụ: "năm mới nhất là bao nhiêu" khi đang bàn Việt Nam → "dân số việt nam năm 2024 2025"
4. "intent": "definition" | "person" | "location" | "time" | "number" | "history" | "comparison" | "general"

Ví dụ có ngữ cảnh:
Ngữ cảnh: Người dùng hỏi về Việt Nam, trợ lý trả lời về dân số Việt Nam.
Câu hỏi: "có thông tin năm mới nhất là bao nhiêu người ko?"
→ normalized: "dân số việt nam năm mới nhất là bao nhiêu người?"
→ rewrite: "dân số việt nam năm mới nhất là bao nhiêu?"
→ bm25: "dân số việt nam năm 2024 2025"
→ intent: "number"

Chỉ trả về JSON, không giải thích."""

HISTORY_SECTION = """Ngữ cảnh hội thoại (đang bàn về chủ đề này):
{history}

→ Câu hỏi hiện tại tiếp nối chủ đề trên. Thay đại từ/tham chiếu bằng danh từ cụ thể.

"""


@dataclass(slots=True)
class RewriteResult:
    """Result of LLM query rewrite."""

    normalized_query: str
    rewrite_query: str
    bm25_query: str
    intent: str


@dataclass(slots=True)
class QueryRewriter:
    """Rewrite queries using LLM for better Wikipedia retrieval.

    Produces:
    - normalized_query: cleaned, expanded abbreviations
    - rewrite_query: semantically expanded version
    - bm25_query: keyword-optimized for BM25 search
    - intent: query intent classification

    Supports conversation history for pronoun resolution.
    """

    llm: LLMClient

    def rewrite(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> RewriteResult:
        """Rewrite a query using LLM.

        Args:
            query: Current user question
            history: Optional conversation history for context
        """
        history_section = self._format_history(history)
        prompt = REWRITE_PROMPT.format(query=query, history_section=history_section)
        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.llm.chat_json(messages)
            return RewriteResult(
                normalized_query=result.get("normalized_query", query.lower().strip()),
                rewrite_query=result.get("rewrite_query", query),
                bm25_query=result.get("bm25_query", query.lower().strip()),
                intent=result.get("intent", "general"),
            )
        except Exception:
            # Fallback to simple normalization if LLM fails
            return RewriteResult(
                normalized_query=query.lower().strip(),
                rewrite_query=query,
                bm25_query=query.lower().strip(),
                intent="general",
            )

    def _format_history(self, history: list[dict[str, str]] | None) -> str:
        """Format history into a readable string for the prompt."""
        if not history:
            return ""

        lines: list[str] = []
        for turn in history:
            role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
            lines.append(f"- {role}: {turn['content']}")
        history_text = "\n".join(lines)
        return HISTORY_SECTION.format(history=history_text)
