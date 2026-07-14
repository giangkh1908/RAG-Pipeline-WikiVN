"""Tests for the chat memory layer (Phase 1)."""

from __future__ import annotations

import pytest

from rag_pipeline.config import MemoryConfig
from rag_pipeline.generation.memory import (
    ConversationMemory,
    BuiltHistory,
    est_tokens,
    turn_tokens,
)
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
    )


@pytest.fixture
def memory(mem_config: MemoryConfig, store: ConversationStore) -> ConversationMemory:
    return ConversationMemory(mem_config, store)


# ─── ConversationStore ────────────────────────────────────────────────────────


class TestSessionOps:
    def test_upsert_session_creates_row(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        assert store.session_exists("session-abc")

    def test_upsert_session_touches_existing(
        self, store: ConversationStore
    ) -> None:
        store.upsert_session("session-abc")
        store.upsert_session("session-abc")
        assert store.get_token_total("session-abc") == 0

    def test_add_to_token_total(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        store.add_to_token_total("session-abc", 100)
        store.add_to_token_total("session-abc", 50)
        assert store.get_token_total("session-abc") == 150

    def test_compact_lock(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        assert store.acquire_compact_lock("session-abc") is True
        # Second acquire should fail because the lock is held.
        assert store.acquire_compact_lock("session-abc") is False
        store.release_compact_lock("session-abc")
        # Now it should succeed again.
        assert store.acquire_compact_lock("session-abc") is True
        store.release_compact_lock("session-abc")


class TestTurnOps:
    def test_next_turn_no_increments(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        assert store.next_turn_no("session-abc") == 1
        store.insert_turn(ChatTurn(session_id="session-abc", turn_no=1, question="Q1"))
        assert store.next_turn_no("session-abc") == 2

    def test_insert_and_update_turn(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        turn = ChatTurn(session_id="session-abc", turn_no=1, question="Q1")
        store.insert_turn(turn)
        store.update_turn_answer("session-abc", 1, "A1", "factual", 50)
        loaded = store.load_completed_turns("session-abc")
        assert len(loaded) == 1
        assert loaded[0].question == "Q1"
        assert loaded[0].answer == "A1"
        assert loaded[0].intent == "factual"
        assert loaded[0].tokens_hint == 50

    def test_load_skips_pending_turns(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        store.insert_turn(ChatTurn(session_id="session-abc", turn_no=1, question="Q1"))
        store.update_turn_answer("session-abc", 1, "A1", None, 0)
        store.insert_turn(ChatTurn(session_id="session-abc", turn_no=2, question="Q2"))
        loaded = store.load_completed_turns("session-abc")
        assert len(loaded) == 1
        assert loaded[0].turn_no == 1

    def test_unique_turn_no_constraint(self, store: ConversationStore) -> None:
        import sqlite3

        store.upsert_session("session-abc")
        store.insert_turn(ChatTurn(session_id="session-abc", turn_no=1, question="Q1"))
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_turn(
                ChatTurn(session_id="session-abc", turn_no=1, question="dup")
            )

    def test_save_and_load_summary(self, store: ConversationStore) -> None:
        store.upsert_session("session-abc")
        store.insert_turn(ChatTurn(session_id="session-abc", turn_no=1, question="Q1"))
        store.update_turn_answer("session-abc", 1, "A1", None, 0)
        store.save_summary("session-abc", "summary text", up_to_turn_no=1)
        assert store.load_latest_summary("session-abc") == "summary text"

    def test_load_latest_summary_with_turn(self, store: ConversationStore) -> None:
        store.upsert_session("s")
        store.insert_turn(ChatTurn(session_id="s", turn_no=1, question="Q1"))
        store.update_turn_answer("s", 1, "A1", None, 0)
        store.insert_turn(ChatTurn(session_id="s", turn_no=2, question="Q2"))
        store.update_turn_answer("s", 2, "A2", None, 0)
        store.save_summary("s", "first", up_to_turn_no=1)
        store.save_summary("s", "second", up_to_turn_no=2)
        text, turn = store.load_latest_summary_with_turn("s")
        assert text == "second"
        assert turn == 2

    def test_delete_session_removes_turns_and_session(
        self, store: ConversationStore
    ) -> None:
        store.upsert_session("s")
        store.insert_turn(ChatTurn(session_id="s", turn_no=1, question="Q1"))
        store.update_turn_answer("s", 1, "A1", None, 0)
        deleted = store.delete_session("s")
        assert deleted == 1
        assert store.session_exists("s") is False
        assert store.load_completed_turns("s") == []


# ─── ConversationMemory ───────────────────────────────────────────────────────


class TestTokenEstimation:
    def test_est_tokens_basic(self) -> None:
        assert est_tokens("") == 0
        assert est_tokens("a" * 3) == 1
        assert est_tokens("a" * 6) == 2

    def test_turn_tokens_sums_question_and_answer(self) -> None:
        turn = ChatTurn(
            session_id="s", turn_no=1, question="abc", answer="abcdefgh"
        )
        # "abc" -> 1 token, "abcdefgh" -> 3 tokens (8/3 rounded up)
        assert turn_tokens(turn, divisor=3) == 4


class TestBuildHistory:
    def test_empty_session_returns_system_plus_question(
        self, memory: ConversationMemory
    ) -> None:
        memory.store.upsert_session("session-abc")
        history = memory.build_history(
            session_id="session-abc",
            current_question="Ha Long o dau?",
            system_guideline="Guide",
            rag_context="context",
        )
        assert isinstance(history, BuiltHistory)
        assert history.used is True
        assert history.raw_turn_count == 0
        # system + current question
        assert len(history.messages) == 2
        assert history.messages[0]["role"] == "system"
        assert "Guide" in history.messages[0]["content"]
        assert "context" in history.messages[0]["content"]
        assert history.messages[-1]["content"] == "Ha Long o dau?"

    def test_includes_prior_turns(self, memory: ConversationMemory) -> None:
        sid = "session-abc"
        memory.store.upsert_session(sid)
        memory.store.insert_turn(ChatTurn(session_id=sid, turn_no=1, question="Q1"))
        memory.store.update_turn_answer(sid, 1, "A1", "factual", 10)
        memory.store.insert_turn(ChatTurn(session_id=sid, turn_no=2, question="Q2"))
        memory.store.update_turn_answer(sid, 2, "A2", "factual", 12)

        history = memory.build_history(
            session_id=sid,
            current_question="Q3",
            system_guideline="Guide",
            rag_context="ctx",
        )
        assert history.raw_turn_count == 2
        # system + 2 turns * 2 messages + current
        assert len(history.messages) == 1 + 4 + 1

    def test_includes_cached_summary(self, memory: ConversationMemory) -> None:
        sid = "session-abc"
        memory.store.upsert_session(sid)
        memory.store.insert_turn(ChatTurn(session_id=sid, turn_no=1, question="Q1"))
        memory.store.update_turn_answer(sid, 1, "A1", None, 0)
        memory.store.save_summary(sid, "short summary", up_to_turn_no=1)

        history = memory.build_history(
            session_id=sid,
            current_question="Q2",
            system_guideline="Guide",
            rag_context="ctx",
        )
        assert history.summary_used is True
        assert any("[TÓM TẮT LỊCH SỬ]" in m["content"] for m in history.messages)

    def test_compute_threshold_matches_formula(
        self, memory: ConversationMemory
    ) -> None:
        # max_input_tokens = ceil(500/3) = 167
        # single_turn = 167 + 800 = 967
        # memory_budget = 3 * 967 = 2901
        # threshold = 0.7 * 2901 = 2030
        threshold = memory.compute_threshold()
        assert 2000 <= threshold <= 2100


# ─── Strip question echo ─────────────────────────────────────────────────────


class TestStripQuestionEcho:
    """Tests for ``RAGPipeline._strip_question_echo``."""

    from rag_pipeline.generation.rag_pipeline import RAGPipeline

    def test_strips_echo_with_newline(self) -> None:
        q = "5 cái trên có gì chơi?"
        a = "5 cái trên có gì chơi?\n\nCác điểm trên có thể tham quan..."
        result = self.RAGPipeline._strip_question_echo(q, a)
        assert result == "Các điểm trên có thể tham quan..."

    def test_strips_echo_with_punctuation(self) -> None:
        q = "Đà Nẵng ở đâu?"
        a = "đà nẵng ở đâu?\n\nĐà Nẵng là thành phố..."
        result = self.RAGPipeline._strip_question_echo(q, a)
        assert result == "Đà Nẵng là thành phố..."

    def test_no_echo_returns_original(self) -> None:
        q = "Hà Nội"
        a = "Hà Nội là thủ đô Việt Nam."
        result = self.RAGPipeline._strip_question_echo(q, a)
        assert result == "Hà Nội là thủ đô Việt Nam."

    def test_echo_only_returns_original(self) -> None:
        q = "test?"
        a = "test?"
        result = self.RAGPipeline._strip_question_echo(q, a)
        assert result == "test?"

    def test_empty_inputs(self) -> None:
        assert self.RAGPipeline._strip_question_echo("", "answer") == "answer"
        assert self.RAGPipeline._strip_question_echo("query", "") == ""

    def test_case_insensitive_match(self) -> None:
        q = "Vịnh Hạ Long ở đâu?"
        a = "vịnh hạ long ở đâu? Vịnh Hạ Long nằm ở tỉnh Quảng Ninh."
        result = self.RAGPipeline._strip_question_echo(q, a)
        assert result == "Vịnh Hạ Long nằm ở tỉnh Quảng Ninh."
