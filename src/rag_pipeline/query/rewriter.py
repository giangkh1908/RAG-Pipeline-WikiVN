"""LLM-based query rewrite for better Wikipedia retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_pipeline.indexing.llm_client import LLMClient

REWRITE_PROMPT = """Bạn là trợ lý tìm kiếm Wikipedia tiếng Việt. Nhiệm vụ: chuẩn hóa câu hỏi để tối ưu tìm kiếm.

{history_section}Cho câu hỏi sau:
"{query}"

Trả về JSON với các trường:
1. "normalized_query": Câu hỏi đã chuẩn hóa (lowercase, viết đầy đủ, không viết tắt). Nếu câu hỏi có đại từ mập mờ (nó, chúng, thành phố đó, người này...), hãy thay bằng danh từ cụ thể dựa trên ngữ cảnh hội thoại.
2. "rewrite_query": Viết lại câu hỏi dưới dạng khác nhưng GIỮ NGUYÊN Ý NGHĨA. KHÔNG thêm câu trả lời vào câu hỏi. Chỉ viết lại cách diễn đạt, thêm từ đồng nghĩa nếu cần. Thay đại từ bằng danh từ cụ thể.
3. "bm25_query": Chỉ giữ lại từ khóa chính, bỏ stopwords và câu hỏi. Ví dụ: "Thủ đô Việt Nam ở đâu?" → "thủ đô việt nam". Nếu có đại từ, thay bằng danh từ cụ thể.
4. "intent": Loại câu hỏi - một trong: "definition", "person", "location", "time", "number", "history", "comparison", "general"

Ví dụ:
- "Thủ đô Việt Nam ở?" → normalized: "thủ đô việt nam ở đâu?", rewrite: "thủ đô của nước việt nam nằm ở đâu?", bm25: "thủ đô việt nam"
- "Sơn Tùng là ai?" → normalized: "sơn tùng là ai?", rewrite: "ca sĩ sơn tùng có tiểu sử như thế nào?", bm25: "sơn tùng"

Chỉ trả về JSON, không giải thích."""

HISTORY_SECTION = """Ngữ cảnh hội thoại trước đó:
{history}

Dựa vào ngữ cảnh trên, hãy hiểu rõ các đại từ và tham chiếu trong câu hỏi hiện tại.

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
