"""Build the chat-history portion of LLM messages from persisted turns.

This module:

* estimates token counts with a Vietnamese-friendly heuristic,
* loads completed turns + cached summary from ``ConversationStore``,
* triggers the compactor (Phase 2) when the raw history grows past the
  computed budget, falling back to truncation on failure,
* assembles the message list in the order documented in
  ``docs/memory-plan.md`` §4.7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rag_pipeline.config import MemoryConfig
from rag_pipeline.storage.conversation import ChatTurn, ConversationStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rag_pipeline.generation.compactor import MemoryCompactor


@dataclass
class BuiltHistory:
    """The LLM message list plus a flag indicating whether memory was used."""

    messages: list[dict[str, str]]
    used: bool
    raw_turn_count: int = 0
    raw_tokens: int = 0
    summary_used: bool = False
    compacted: bool = False


def est_tokens(text: str, divisor: int = 3) -> int:
    """Token estimation heuristic.

    Vietnamese combines latin letters with diacritics, so we use
    ``len(text) // divisor`` (default ``3``) to err on the generous side.
    The estimator is intentionally cheap; a real tokenizer can be swapped
    in later without changing call sites.
    """
    if not text:
        return 0
    return (len(text) + divisor - 1) // divisor


def turn_tokens(turn: ChatTurn, divisor: int) -> int:
    """Tokens of a single turn (question + answer)."""
    return est_tokens(turn.question, divisor) + est_tokens(turn.answer or "", divisor)


class ConversationMemory:
    """Assemble LLM messages from a session's persisted turns."""

    def __init__(
        self,
        config: MemoryConfig,
        store: ConversationStore,
        compactor: "MemoryCompactor | None" = None,
    ) -> None:
        self.config = config
        self.store = store
        self.compactor = compactor

    def compute_threshold(self) -> int:
        """Runtime budget = ``keep_raw_turns * (input+output) * 0.7``."""
        max_input = (
            self.config.max_input_chars + self.config.char_per_token - 1
        ) // self.config.char_per_token
        single_turn = max_input + self.config.max_output_tokens
        return int(0.7 * self.config.keep_raw_turns * single_turn)

    def needs_compact(self, raw_tokens: int) -> bool:
        return raw_tokens >= self.compute_threshold()

    def build_history(
        self,
        session_id: str,
        current_question: str,
        system_guideline: str,
        rag_context: str,
    ) -> BuiltHistory:
        """Assemble the LLM message list for the current turn.

        When the raw history exceeds the budget the compactor is invoked.
        If the compactor is unavailable or fails, the function degrades
        gracefully by keeping only the most recent ``keep_raw_turns``
        turns (no summary).
        """
        all_turns = self.store.load_completed_turns(session_id)
        raw_tokens = sum(
            turn_tokens(t, self.config.char_per_token) for t in all_turns
        )
        threshold = self.compute_threshold()

        selected_turns = all_turns
        summary_text: str | None = None
        summary_used = False
        compacted = False

        if raw_tokens >= threshold and self.compactor is not None:
            summary_text = self.compactor.compact(session_id)
            if summary_text:
                compacted = True
                summary_used = True
                keep = self.config.keep_raw_turns
                if len(all_turns) > keep:
                    selected_turns = all_turns[-keep:]
        elif raw_tokens >= threshold and self.compactor is None:
            # No compactor available: truncate silently to stay under budget.
            keep = self.config.keep_raw_turns
            if len(all_turns) > keep:
                selected_turns = all_turns[-keep:]

        # If the compactor didn't run (still below threshold) but a
        # previous summary exists, surface it.
        if summary_text is None:
            existing = self.store.load_latest_summary(session_id)
            if existing is not None:
                summary_text = existing
                summary_used = True

        messages: list[dict[str, str]] = []
        # 1. System guideline + RAG context.
        system_content = system_guideline
        if rag_context:
            system_content = f"{system_guideline}\n\nNGỮ CẢNH:\n{rag_context}"
        messages.append({"role": "system", "content": system_content})

        # 2. Cached summary (if any) — kept as a user-role turn so the LLM
        # treats it as factual input, not instruction.
        if summary_text:
            messages.append(
                {"role": "user", "content": f"[TÓM TẮT LỊCH SỬ]: {summary_text}"}
            )

        # 3. Raw turn pairs.
        for turn in selected_turns:
            messages.append({"role": "user", "content": turn.question})
            if turn.answer is not None:
                messages.append({"role": "assistant", "content": turn.answer})

        # 4. Current question.
        messages.append({"role": "user", "content": current_question})

        return BuiltHistory(
            messages=messages,
            used=True,
            raw_turn_count=len(selected_turns),
            raw_tokens=raw_tokens,
            summary_used=summary_used,
            compacted=compacted,
        )
