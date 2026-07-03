"""Prompt builder for RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.config import GenerationConfig
from rag_pipeline.models import RetrievalResult


@dataclass(slots=True)
class PromptBuilder:
    """Builds system + user messages for LLM answer generation.

    The prompt instructs the LLM to:
    - Answer in Vietnamese based on the provided context
    - Cite sources using [1], [2]... numbering
    - Return structured JSON with answer and citations
    """

    config: GenerationConfig

    def build(self, retrieval_result: RetrievalResult) -> list[dict[str, str]]:
        """Build chat messages for the LLM (structured JSON output).

        Args:
            retrieval_result: Output from Phase 3 retrieval

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        system_msg = self._build_system_message()
        user_msg = self._build_user_message(retrieval_result)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def build_streaming(self, retrieval_result: RetrievalResult) -> list[dict[str, str]]:
        """Build chat messages for streaming (plain text output).

        No JSON format — just answer text for natural token-by-token streaming.
        """
        system_msg = self._build_streaming_system_message()
        user_msg = self._build_user_message_plain(retrieval_result)
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

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
