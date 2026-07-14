"""Persistent storage for chat sessions and turns (anonymous, no auth).

Two tables live in the same SQLite file as the RAG corpus:
- ``chat_sessions``: one row per session, tracks activity + token totals.
- ``chat_turns``: one row per user question (and the assistant answer that
  fills it in after streaming finishes).
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from rag_pipeline.storage.sqlite import SQLiteStorage

ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def _parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, ISO).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass
class ChatTurn:
    """A single user → assistant exchange."""

    session_id: str
    turn_no: int
    question: str
    answer: str | None = None
    intent: str | None = None
    tokens_hint: int = 0
    summary: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=_now)

    def to_message_pair(self) -> list[dict[str, str]]:
        """Return the [user, assistant] pair of messages for LLM history."""
        pair: list[dict[str, str]] = [{"role": "user", "content": self.question}]
        if self.answer is not None:
            pair.append({"role": "assistant", "content": self.answer})
        return pair


class ConversationStore:
    """DAO for ``chat_sessions`` and ``chat_turns`` tables.

    The store borrows the SQLite connection from an existing
    :class:`SQLiteStorage` so that we share the same WAL-mode file. A
    ``threading.Lock`` serialises all access because FastAPI runs
    ``_resolve_session_id`` in the async event-loop thread while
    ``answer_stream`` runs in a ``ThreadPoolExecutor`` worker — both
    touch the same ``sqlite3.Connection``.
    """

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage
        self._conn: sqlite3.Connection = storage._connection  # type: ignore[attr-defined]
        self._lock: threading.Lock = storage._lock  # type: ignore[attr-defined]

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run a SQL statement under the thread lock."""
        with self._lock:
            return self._conn.execute(sql, params)

    def _executescript(self, sql: str) -> None:
        with self._lock:
            self._conn.executescript(sql)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def upsert_session(self, session_id: str) -> None:
        """Insert or touch a session row (updates ``last_active_at``)."""
        now = _now()
        self._execute(
            """
            INSERT INTO chat_sessions (session_id, created_at, last_active_at, token_total, compacting)
            VALUES (?, ?, ?, 0, 0)
            ON CONFLICT(session_id) DO UPDATE SET last_active_at=excluded.last_active_at
            """,
            (session_id, now, now),
        )

    def touch_session(self, session_id: str) -> None:
        """Update only ``last_active_at`` for an existing session."""
        self._execute(
            "UPDATE chat_sessions SET last_active_at=? WHERE session_id=?",
            (_now(), session_id),
        )

    def session_exists(self, session_id: str) -> bool:
        row = self._execute(
            "SELECT 1 FROM chat_sessions WHERE session_id=? LIMIT 1",
            (session_id,),
        ).fetchone()
        return row is not None

    def get_token_total(self, session_id: str) -> int:
        row = self._execute(
            "SELECT token_total FROM chat_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return int(row["token_total"]) if row else 0

    def add_to_token_total(self, session_id: str, delta: int) -> None:
        if delta == 0:
            return
        self._execute(
            "UPDATE chat_sessions SET token_total = token_total + ? WHERE session_id=?",
            (delta, session_id),
        )

    def acquire_compact_lock(self, session_id: str) -> bool:
        """Try to mark the session as ``compacting=1``. Returns True on success."""
        cur = self._execute(
            "UPDATE chat_sessions SET compacting=1 "
            "WHERE session_id=? AND compacting=0",
            (session_id,),
        )
        return cur.rowcount > 0

    def release_compact_lock(self, session_id: str) -> None:
        self._execute(
            "UPDATE chat_sessions SET compacting=0 WHERE session_id=?",
            (session_id,),
        )

    def gc_sessions(self, ttl_hours: int) -> int:
        """Delete sessions whose last activity is older than ``ttl_hours``."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        cutoff_iso = cutoff.strftime(ISO)
        cur = self._execute(
            """
            DELETE FROM chat_sessions
            WHERE last_active_at < ?
              AND NOT EXISTS (
                  SELECT 1 FROM chat_turns
                  WHERE chat_turns.session_id = chat_sessions.session_id
                    AND chat_turns.answer IS NULL
              )
            """,
            (cutoff_iso,),
        )
        return cur.rowcount

    def gc_sessions_ttl(self) -> int:
        """Delete sessions inactive for longer than ``MemoryConfig.session_ttl_hours``.

        Reads the TTL from environment (default 24h) to keep the storage
        layer decoupled from the config dataclass.
        """
        from rag_pipeline.config import RAGConfig  # local import to avoid cycle

        return self.gc_sessions(RAGConfig().memory.session_ttl_hours)

    def delete_session(self, session_id: str) -> int:
        """Hard-delete a session and all its turns. Returns turn count removed."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chat_turns WHERE session_id=?", (session_id,)
            )
            turns_deleted = cur.rowcount
            self._conn.execute(
                "DELETE FROM chat_sessions WHERE session_id=?", (session_id,)
            )
        return turns_deleted

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------
    def next_turn_no(self, session_id: str) -> int:
        row = self._execute(
            "SELECT COALESCE(MAX(turn_no), 0) AS m FROM chat_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return int(row["m"]) + 1

    def insert_turn(self, turn: ChatTurn) -> ChatTurn:
        self._execute(
            """
            INSERT INTO chat_turns
                (id, session_id, turn_no, question, answer, intent, tokens_hint, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn.id,
                turn.session_id,
                turn.turn_no,
                turn.question,
                turn.answer,
                turn.intent,
                turn.tokens_hint,
                turn.summary,
                turn.created_at,
            ),
        )
        return turn

    def update_turn_answer(
        self,
        session_id: str,
        turn_no: int,
        answer: str,
        intent: str | None,
        tokens_hint: int,
    ) -> None:
        self._execute(
            """
            UPDATE chat_turns
            SET answer=?, intent=?, tokens_hint=?
            WHERE session_id=? AND turn_no=?
            """,
            (answer, intent, tokens_hint, session_id, turn_no),
        )

    def load_completed_turns(self, session_id: str) -> list[ChatTurn]:
        """Return every turn with a non-null ``answer``, ordered by turn_no ASC."""
        rows = self._execute(
            """
            SELECT * FROM chat_turns
            WHERE session_id=? AND answer IS NOT NULL
            ORDER BY turn_no ASC
            """,
            (session_id,),
        ).fetchall()
        return [self._row_to_turn(row) for row in rows]

    def load_latest_summary(self, session_id: str) -> str | None:
        """Return the most recent cached summary text for a session, if any."""
        row = self._execute(
            """
            SELECT summary FROM chat_turns
            WHERE session_id=? AND summary IS NOT NULL
            ORDER BY turn_no DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        return row["summary"] if row else None

    def load_latest_summary_with_turn(
        self, session_id: str
    ) -> tuple[str | None, int | None]:
        """Return ``(summary_text, up_to_turn_no)`` for the most recent summary.

        ``up_to_turn_no`` is the highest turn number covered by that summary.
        The compactor uses it to decide which turns still need to be folded
        into the next summary pass.
        """
        row = self._execute(
            """
            SELECT summary, turn_no FROM chat_turns
            WHERE session_id=? AND summary IS NOT NULL
            ORDER BY turn_no DESC LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None, None
        return row["summary"], int(row["turn_no"])

    def save_summary(
        self, session_id: str, summary: str, up_to_turn_no: int
    ) -> None:
        """Cache a summary in the latest summarized turn's row.

        Storing the summary on the turn row makes it trivial to fetch later
        and keeps the schema flat (no extra join table for Phase 1).
        """
        self._execute(
            "UPDATE chat_turns SET summary=? WHERE session_id=? AND turn_no=?",
            (summary, session_id, up_to_turn_no),
        )

    def count_turns(self, session_id: str) -> int:
        row = self._execute(
            "SELECT COUNT(*) AS c FROM chat_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> ChatTurn:
        return ChatTurn(
            id=row["id"],
            session_id=row["session_id"],
            turn_no=int(row["turn_no"]),
            question=row["question"],
            answer=row["answer"],
            intent=row["intent"],
            tokens_hint=int(row["tokens_hint"]),
            summary=row["summary"],
            created_at=row["created_at"],
        )

    def close(self) -> None:  # pragma: no cover - delegate
        self._storage.close()
