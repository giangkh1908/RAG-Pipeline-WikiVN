"""Prompt builder for RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_pipeline.config import GenerationConfig
from rag_pipeline.models import RetrievalResult

# Token estimation: ~4 chars per token (Vietnamese/English mix)
_CHARS_PER_TOKEN = 4

# Default context window budget (tokens)
_DEFAULT_MAX_CONTEXT = 16_000

# Reserved tokens for non-history parts
_RESERVED_SYSTEM = 600       # system message
_RESERVED_PASSAGES = 4_000   # retrieved passages
_RESERVED_QUESTION = 300     # user question + formatting
_RESERVED_RESPONSE = 2_000   # expected LLM response

# Summarization thresholds
_SUMMARIZE_THRESHOLD = 6     # Summarize when history > N turns
_KEEP_RECENT_TURNS = 2       # Keep last N turns as-is when summarizing

SUMMARIZE_PROMPT = """Tóm tắt cuộc hội thoại sau thành 2-3 câu ngắn gọn, giữ lại thông tin quan trọng (chủ đề, facts, entities):

{history}

Chỉ trả về text tóm tắt, không giải thích."""


def _estimate_tokens(text: str) -> int:
    """Rough token count estimation."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


@dataclass(slots=True)
class PromptBuilder:
    """Builds system + user messages for LLM answer generation.

    The prompt instructs the LLM to:
    - Answer in Vietnamese based on the provided context
    - Cite sources using [1], [2]... numbering
    - Return structured JSON with answer and citations

    Supports conversation history with:
    - Token-budget-based truncation (always active)
    - Summarization of older turns when history is long (optional, needs LLM)
    """

    config: GenerationConfig
    llm_client: Any = None  # Optional LLM client for summarization

    @property
    def max_context_tokens(self) -> int:
        return self.config.max_context_tokens

    def build(
        self,
        retrieval_result: RetrievalResult,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for the LLM (structured JSON output).

        Args:
            retrieval_result: Output from Phase 3 retrieval
            history: Optional conversation history [{role, content}, ...]

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        system_msg = self._build_system_message()
        user_msg = self._build_user_message(retrieval_result)
        processed_history = self._process_history(history, system_msg, user_msg)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        messages.extend(processed_history)
        messages.append({"role": "user", "content": user_msg})
        return messages

    def build_streaming(
        self,
        retrieval_result: RetrievalResult,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build chat messages for streaming (plain text output).

        No JSON format — just answer text for natural token-by-token streaming.
        """
        system_msg = self._build_streaming_system_message()
        user_msg = self._build_user_message_plain(retrieval_result)
        processed_history = self._process_history(history, system_msg, user_msg)
        messages: list[dict[str, str]] = [{"role": "system", "content": system_msg}]
        messages.extend(processed_history)
        messages.append({"role": "user", "content": user_msg})
        return messages

    def _process_history(
        self,
        history: list[dict[str, str]] | None,
        system_msg: str,
        user_msg: str,
    ) -> list[dict[str, str]]:
        """Process history: summarize if long, then trim to fit token budget.

        Strategy:
        1. If history <= threshold: send full history
        2. If history > threshold: summarize older turns + keep recent turns
        3. Always: trim to fit token budget
        """
        if not history:
            return []

        # Step 1: Summarize if history is long
        if len(history) > _SUMMARIZE_THRESHOLD and self.llm_client is not None:
            processed = self._summarize_and_keep_recent(history)
        else:
            processed = list(history)

        # Step 2: Trim to fit token budget
        return self._trim_history(processed, system_msg, user_msg)

    def _summarize_and_keep_recent(
        self,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Summarize older turns, keep recent turns as-is.

        Example:
            Input:  [turn1, turn2, turn3, turn4, turn5, turn6, turn7, turn8]
            Output: [summary_turn, turn7, turn8]
        """
        # Split: older turns to summarize + recent turns to keep
        split_at = len(history) - _KEEP_RECENT_TURNS
        older_turns = history[:split_at]
        recent_turns = history[split_at:]

        # Summarize older turns
        summary_text = self._call_llm_summarize(older_turns)

        # Build result: summary as system message + recent turns
        result: list[dict[str, str]] = []
        if summary_text:
            result.append({
                "role": "system",
                "content": f"[Tóm tắt {len(older_turns)} lượt hội thoại trước: {summary_text}]",
            })
        result.extend(recent_turns)
        return result

    def _call_llm_summarize(self, turns: list[dict[str, str]]) -> str:
        """Call LLM to summarize conversation turns."""
        if not self.llm_client:
            return ""

        # Format turns for summarization
        lines: list[str] = []
        for turn in turns:
            role = "Người dùng" if turn["role"] == "user" else "Trợ lý"
            # Truncate long messages for summarization
            content = turn["content"][:500]
            lines.append(f"{role}: {content}")
        history_text = "\n".join(lines)

        prompt = SUMMARIZE_PROMPT.format(history=history_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            return self.llm_client.chat(messages, max_tokens=200, temperature=0.3)
        except Exception:
            # Fallback: return empty (will use truncation instead)
            return ""

    def _trim_history(
        self,
        history: list[dict[str, str]],
        system_msg: str,
        user_msg: str,
    ) -> list[dict[str, str]]:
        """Trim history to fit within token budget.

        Budget = max_context - reserved (system + passages + question + response).
        Keeps most recent turns, drops oldest when over budget.
        """
        if not history:
            return []

        # Calculate how many tokens we have for history
        reserved = (
            _estimate_tokens(system_msg)
            + _estimate_tokens(user_msg)
            + _RESERVED_RESPONSE
        )
        history_budget = max(0, self.max_context_tokens - reserved)

        # Walk from newest to oldest, accumulate until budget exceeded
        total = 0
        keep_from = len(history)
        for i in range(len(history) - 1, -1, -1):
            turn_tokens = _estimate_tokens(history[i].get("content", ""))
            if total + turn_tokens > history_budget:
                break
            total += turn_tokens
            keep_from = i

        trimmed = history[keep_from:]

        # If we dropped turns, add a summary marker so LLM knows context was truncated
        if len(trimmed) < len(history) and trimmed:
            dropped = len(history) - len(trimmed)
            # Check if first item is already a summary
            if not trimmed[0].get("content", "").startswith("[Tóm tắt"):
                trimmed.insert(0, {
                    "role": "system",
                    "content": f"[Đã lược bỏ {dropped} lượt hội thoại cũ để tiết kiệm ngữ cảnh]",
                })

        return trimmed

    def _build_system_message(self) -> str:
        return (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ trả lời dựa trên thông tin trong ngữ_context. "
            "Nếu không đủ thông tin, nói rõ 'Tôi không tìm thấy thông tin đủ để trả lời'.\n"
            "2. Trả lời bằng tiếng Việt, ngắn gọn và chính xác.\n"
            "3. Trích dẫn nguồn bằng cách đánh số [1], [2]... tương ứng với thứ tự passage.\n"
            "4. Trả về JSON với format:\n"
            '   {"answer": "câu trả lời", "citations": [{"claim": "câu claim", "source_index": 1}, ...], "confidence": 0.0-1.0}\n'
            "5. confidence: mức độ tin cậy (0.0 = không chắc, 1.0 = rất chắc)\n"
            "6. citations: mỗi claim quan trọng phải có source_index tham chiếu passage nguồn"
        )

    def _build_user_message(self, retrieval_result: RetrievalResult) -> str:
        passages_text = self._format_passages(retrieval_result.passages)
        question = retrieval_result.query.original_query

        return (
            f"NGỮ CẢNH:\n{passages_text}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "Hãy trả lời dựa trên ngữ_context trên và trả về JSON."
        )

    def _build_streaming_system_message(self) -> str:
        return (
            "Bạn là trợ lý AI trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.\n\n"
            "QUY TẮC:\n"
            "1. Chỉ trả lời dựa trên thông tin trong ngữ cảnh. "
            "Nếu không đủ thông tin, nói rõ 'Tôi không tìm thấy thông tin đủ để trả lời'.\n"
            "2. Trả lời bằng tiếng Việt, ngắn gọn và chính xác.\n"
            "3. Trả lời trực tiếp bằng văn bản thường. KHÔNG trả về JSON.\n"
            "4. Không thêm phần mở đầu như 'Dựa trên ngữ cảnh...' hay 'Theo thông tin...'. Trả lời thẳng vào câu hỏi."
        )

    def _build_user_message_plain(self, retrieval_result: RetrievalResult) -> str:
        passages_text = self._format_passages(retrieval_result.passages)
        question = retrieval_result.query.original_query

        return (
            f"NGỮ CẢNH:\n{passages_text}\n\n"
            f"CÂU HỎI: {question}\n\n"
            "Hãy trả lời câu hỏi trên dựa trên ngữ cảnh."
        )

    def _format_passages(self, passages: list) -> str:
        if not passages:
            return "(Không tìm thấy passage nào)"

        parts: list[str] = []
        for p in passages:
            source = f" — {p.title}" if p.title else ""
            parts.append(f"[{p.rank}]{source}\n{p.text}")

        return "\n\n".join(parts)
