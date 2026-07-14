"""Tests for the memory compactor (Phase 2)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from rag_pipeline.config import MemoryConfig
from rag_pipeline.generation.compactor import MemoryCompactor
from rag_pipeline.generation.memory import ConversationMemory
from rag_pipeline.storage.conversation import ChatTurn, ConversationStore
from rag_pipeline.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage() -> SQLiteStorage:
    db = SQLiteStorage(":memory:")
    yield db
    db.close()


@pytest.fixture
def store(storage: SQLiteStorage) -> ConversationStore:
    return ConversationStore(storage)


@pytest.fixture
def mem_config() -> MemoryConfig:
    return MemoryConfig(
        enabled=True,
        keep_raw_turns=3,
        max_input_chars=500,
        max_output_tokens=800,
        char_per_token=3,
        summary_max_retries=2,
    )


def _seed_turns(
    store: ConversationStore,
    session_id: str,
    count: int,
    *,
    start: int = 1,
) -> None:
    """Seed ``count`` completed turns for the session starting at ``start``."""
    store.upsert_session(session_id)
    for i in range(start, start + count):
        store.insert_turn(
            ChatTurn(
                session_id=session_id,
                turn_no=i,
                question=f"Q{i}",
                answer=f"A{i}",
            )
        )
        store.update_turn_answer(session_id, i, f"A{i}", None, 10)


def _make_mock_client(payload: dict | list[dict]) -> MagicMock:
    """Build a mock httpx.Client that returns the given JSON body."""
    client = MagicMock(spec=httpx.Client)
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    client.post.return_value = response
    return client


def _summary_payload(text: str) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
    }


def _summary_response(text: str) -> MagicMock:
    """Mock httpx response carrying the given summary text."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = _summary_payload(text)
    return response


# ─── MemoryCompactor basics ───────────────────────────────────────────────────


class TestCompactorBasics:
    def test_returns_existing_summary_when_under_keep(
        self, mem_config: MemoryConfig, store: ConversationStore
    ) -> None:
        client = _make_mock_client(_summary_payload("anything"))
        compactor = MemoryCompactor(mem_config, store, client=client)

        # Seed turn 1, then save summary tagged on it.
        store.upsert_session("s")
        _seed_turns(store, "s", 1)
        store.save_summary("s", "prior summary", up_to_turn_no=1)

        # Add one more turn. Total 2 < keep_raw_turns=3.
        _seed_turns(store, "s", 1, start=2)
        result = compactor.compact("s")
        assert result == "prior summary"
        # LLM should not have been called.
        client.post.assert_not_called()

    def test_compact_summarises_older_turns_and_caches(
        self, mem_config: MemoryConfig, store: ConversationStore
    ) -> None:
        client = _make_mock_client(_summary_payload("TOM TAT MOI"))
        compactor = MemoryCompactor(mem_config, store, client=client)

        _seed_turns(store, "s", 5)
        result = compactor.compact("s")

        assert result == "TOM TAT MOI"
        # Cached.
        cached = store.load_latest_summary_with_turn("s")
        assert cached[0] == "TOM TAT MOI"
        # Summarised up to turn 2 (turns 1-2, keep 3-5 raw).
        assert cached[1] == 2
        # LLM was called exactly once.
        client.post.assert_called_once()

    def test_subsequent_compact_only_folds_new_turns(
        self, mem_config: MemoryConfig, store: ConversationStore
    ) -> None:
        # First call returns "TOM TAT MOI", second returns "TOM TAT CAP NHAT".
        client = MagicMock(spec=httpx.Client)
        client.post.side_effect = [
            _summary_response("TOM TAT MOI"),
            _summary_response("TOM TAT CAP NHAT"),
        ]
        compactor = MemoryCompactor(mem_config, store, client=client)

        # 5 turns → compact → summary covers up to turn 2.
        _seed_turns(store, "s", 5)
        compactor.compact("s")

        # 7 turns now → next compact should only fold turns 3-4.
        _seed_turns(store, "s", 2, start=6)

        result = compactor.compact("s")
        assert result == "TOM TAT CAP NHAT"
        assert client.post.call_count == 2

        # Inspect the prompt of the SECOND call: it should reference the
        # previous summary ("TOM TAT MOI") AND only the new turns (3, 4).
        second_call_kwargs = client.post.call_args_list[1].kwargs
        prompt = second_call_kwargs["json"]["messages"][1]["content"]
        assert "TOM TAT MOI" in prompt
        assert "Q3" in prompt
        assert "Q4" in prompt
        assert "Q5" not in prompt
        assert "Q6" not in prompt

    def test_compact_returns_old_summary_when_llm_fails(
        self, mem_config: MemoryConfig, store: ConversationStore
    ) -> None:
        client = MagicMock(spec=httpx.Client)
        client.post.side_effect = RuntimeError("LLM boom")
        compactor = MemoryCompactor(mem_config, store, client=client)

        # Seed turn 1, then save summary on it.
        _seed_turns(store, "s", 1)
        store.save_summary("s", "prior", up_to_turn_no=1)
        # Add 4 more turns. Total 5.
        _seed_turns(store, "s", 4, start=2)

        result = compactor.compact("s")
        assert result == "prior"
        # Retry budget exhausted (2 attempts).
        assert client.post.call_count == mem_config.summary_max_retries

    def test_compact_lock_blocks_second_request(
        self, mem_config: MemoryConfig, store: ConversationStore
    ) -> None:
        client = _make_mock_client(_summary_payload("S1"))
        compactor = MemoryCompactor(mem_config, store, client=client)

        _seed_turns(store, "s", 5)

        # Manually hold the lock to simulate another request compacting.
        assert store.acquire_compact_lock("s") is True
        result = compactor.compact("s")
        assert result is None
        client.post.assert_not_called()
        store.release_compact_lock("s")


# ─── ConversationMemory integration ──────────────────────────────────────────


class TestMemoryWithCompactor:
    def test_build_history_triggers_compact(
        self,
        mem_config: MemoryConfig,
        store: ConversationStore,
    ) -> None:
        # Small enough threshold to trigger easily.
        mem_config.keep_raw_turns = 2
        mem_config.max_input_chars = 60
        mem_config.max_output_tokens = 30

        client = _make_mock_client(_summary_payload("summary text"))
        compactor = MemoryCompactor(mem_config, store, client=client)
        memory = ConversationMemory(mem_config, store, compactor=compactor)

        # 5 long turns forces a compact.
        store.upsert_session("s")
        for i in range(1, 6):
            long_q = "Q" + "a" * 200
            long_a = "A" + "b" * 200
            store.insert_turn(
                ChatTurn(session_id="s", turn_no=i, question=long_q, answer=long_a)
            )
            store.update_turn_answer("s", i, long_a, None, 100)

        history = memory.build_history(
            session_id="s",
            current_question="next",
            system_guideline="Guide",
            rag_context="ctx",
        )

        assert history.compacted is True
        # Summary message should appear at index 1 (after system).
        assert any(
            "[TÓM TẮT LỊCH SỬ]" in m["content"] for m in history.messages
        )
        # Exactly keep_raw_turns (2) raw pairs follow the summary.
        raw_pairs = sum(
            1 for m in history.messages if m["role"] in {"user", "assistant"}
        ) - 1  # minus the summary user message
        # The summary counts as a user message, plus 2 turns × 2 = 4, plus current = 1.
        # Raw_turn_count reports just the raw turns.
        assert history.raw_turn_count == 2

    def test_build_history_falls_back_to_truncate_when_no_compactor(
        self,
        mem_config: MemoryConfig,
        store: ConversationStore,
    ) -> None:
        # Tight budget so 5 long turns blow past it.
        mem_config.keep_raw_turns = 3
        mem_config.max_input_chars = 30
        mem_config.max_output_tokens = 10
        memory = ConversationMemory(mem_config, store, compactor=None)

        store.upsert_session("s")
        for i in range(1, 6):
            long_q = "Q" + "a" * 200
            long_a = "A" + "b" * 200
            store.insert_turn(
                ChatTurn(session_id="s", turn_no=i, question=long_q, answer=long_a)
            )
            store.update_turn_answer("s", i, long_a, None, 100)

        history = memory.build_history(
            session_id="s",
            current_question="next",
            system_guideline="Guide",
            rag_context="ctx",
        )

        # Truncated to keep_raw_turns, no summary emitted.
        assert history.compacted is False
        assert history.summary_used is False
        assert history.raw_turn_count == mem_config.keep_raw_turns
