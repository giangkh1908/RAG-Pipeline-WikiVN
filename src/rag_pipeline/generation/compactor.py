"""Summarise older turns of a chat session to free up memory budget.

The compactor is invoked when the raw history of a session exceeds the
``ConversationMemory`` budget. It:

1. acquires a per-session lock (``compacting=1`` on ``chat_sessions``)
   so concurrent requests don't double-summarise the same range;
2. re-reads the turns + the latest cached summary after the lock to
   avoid stale state;
3. calls the configured LLM (non-streaming) with the older turns plus
   the existing summary;
4. caches the new summary tagged on the highest summarised turn, so
   the next pass only needs to fold in the turns that arrived since;
5. returns the new summary text — or ``None`` on failure so the
   caller can fall back to a raw-history truncation.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import httpx

from rag_pipeline.config import MemoryConfig
from rag_pipeline.storage.conversation import ChatTurn, ConversationStore

if TYPE_CHECKING:
    pass


class MemoryCompactor:
    """LLM-backed summariser for older chat turns.

    The class is intentionally side-effect driven: ``compact()`` is the
    single entry point and the only public state lives in
    :class:`ConversationStore`.
    """

    _SYSTEM_PROMPT = (
        "Bạn là trợ lý tóm tắt hội thoại. Nhiệm vụ: tóm tắt đoạn hội thoại "
        "thành 3-5 câu ngắn gọn, giữ tên địa danh, địa điểm và intent chính. "
        "Không thêm thông tin mới, không bình luận, không chào hỏi."
    )

    _USER_PROMPT_TEMPLATE = (
        "{old_block}Hãy tóm tắt đoạn hội thoại sau thành 3-5 câu, giữ ý chính:\n\n"
        "{turns_text}"
    )

    def __init__(
        self,
        config: MemoryConfig,
        store: ConversationStore,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._owns_client = client is None
        self._client = client or self._build_client()

    def _build_client(self) -> httpx.Client:
        api_key = os.getenv(self.config.summary_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set the {self.config.summary_api_key_env} "
                f"environment variable"
            )
        return httpx.Client(
            base_url=self.config.summary_api_base,
            timeout=self.config.summary_timeout_seconds,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rag-pipeline.local",
                "X-Title": "RAG Pipeline",
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compact(self, session_id: str) -> str | None:
        """Try to summarise older turns for ``session_id``.

        Returns the new (or existing) summary text, or ``None`` if the
        session has nothing worth summarising or another request holds
        the compact lock.
        """
        if not self.store.acquire_compact_lock(session_id):
            return None

        try:
            all_turns = self.store.load_completed_turns(session_id)
            keep = self.config.keep_raw_turns

            if len(all_turns) <= keep:
                # Not enough history to compact.
                old_summary, _ = self.store.load_latest_summary_with_turn(session_id)
                return old_summary

            # The cutoff is the turn right before the kept ones. With
            # ``keep=3`` and 7 turns, the kept ones are turns 5-7, so the
            # cutoff is turn 4.
            keep_cutoff_turn_no = all_turns[-(keep + 1)].turn_no
            old_summary, summary_turn_no = self.store.load_latest_summary_with_turn(
                session_id
            )

            if summary_turn_no is not None:
                to_summarize = [
                    t
                    for t in all_turns
                    if summary_turn_no < t.turn_no <= keep_cutoff_turn_no
                ]
            else:
                to_summarize = all_turns[:-keep]

            if not to_summarize and old_summary is None:
                return None
            if not to_summarize:
                # Nothing new since the last summary; reuse it.
                return old_summary

            new_summary = self._call_llm_with_retry(
                old_summary=old_summary,
                turns=to_summarize,
            )

            if new_summary is None:
                # LLM failed; return the old summary so the caller can
                # still inject some context.
                return old_summary

            last_summarised = max(t.turn_no for t in to_summarize)
            self.store.save_summary(
                session_id, new_summary, up_to_turn_no=last_summarised
            )
            return new_summary
        finally:
            self.store.release_compact_lock(session_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _call_llm_with_retry(
        self, old_summary: str | None, turns: list[ChatTurn]
    ) -> str | None:
        prompt = self._build_prompt(old_summary, turns)
        last_exception: Exception | None = None
        for attempt in range(self.config.summary_max_retries):
            try:
                response = self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.summary_model_name,
                        "messages": [
                            {"role": "system", "content": self._SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": self.config.summary_max_tokens,
                        "temperature": self.config.summary_temperature,
                    },
                )
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                content = data["choices"][0]["message"]["content"]
                if content:
                    return content.strip()
                return None
            except Exception as exc:
                last_exception = exc
                if attempt < self.config.summary_max_retries - 1:
                    time.sleep(2**attempt)
        return None

    @staticmethod
    def _build_prompt(old_summary: str | None, turns: list[ChatTurn]) -> str:
        old_block = f"Tóm tắt trước:\n{old_summary}\n\n" if old_summary else ""
        body = "\n\n".join(
            f"User: {t.question}\nAssistant: {t.answer or ''}" for t in turns
        )
        return MemoryCompactor._USER_PROMPT_TEMPLATE.format(
            old_block=old_block, turns_text=body
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
