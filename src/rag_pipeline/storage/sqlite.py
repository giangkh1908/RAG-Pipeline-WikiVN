"""SQLite implementation of the storage layer."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from rag_pipeline.storage.models import Chunk, Document, IndexEntry, Source


class SQLiteStorage:
    """Persistent SQLite storage for Source → Document → Chunk → Index.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file. The special value ``":memory:"``
        creates an in-memory database useful for tests.
    """

    def __init__(self, db_path: str | Path = "data/rag_storage.db") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode
        )
        self._connection.row_factory = sqlite3.Row
        # Serialise access from multiple threads (async loop + executor).
        # Use RLock so methods that call _exec (which locks) don't deadlock
        # when they're themselves called from within a locked context.
        self._lock = threading.RLock()
        # Wait up to 10s for a lock instead of failing immediately.
        self._exec("PRAGMA busy_timeout=10000")
        if self._db_path != ":memory:":
            self._exec("PRAGMA journal_mode=WAL")
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            type TEXT NOT NULL,
            version TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            checksum TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (source_id) REFERENCES sources(id)
        );

        CREATE INDEX IF NOT EXISTS idx_documents_source_status
            ON documents(source_id, status);

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_order INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_document_order
            ON chunks(document_id, chunk_order);

        CREATE TABLE IF NOT EXISTS index_entries (
            chunk_id TEXT PRIMARY KEY,
            dense_vector TEXT,
            sparse_vector TEXT,
            metadata TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        );

        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_active_at TEXT NOT NULL,
            token_total INTEGER NOT NULL DEFAULT 0,
            compacting INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_chat_sessions_last_active
            ON chat_sessions(last_active_at);

        CREATE TABLE IF NOT EXISTS chat_turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            turn_no INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            intent TEXT,
            tokens_hint INTEGER NOT NULL DEFAULT 0,
            summary TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(session_id, turn_no),
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_chat_turns_session_turn
            ON chat_turns(session_id, turn_no);
        """
        with self._lock:
            self._connection.executescript(ddl)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL under the thread lock (serialises concurrent access)."""
        with self._lock:
            return self._connection.execute(sql, params)

    @staticmethod
    def _dump_json(value: Any) -> str:
        return json.dumps(value, default=str, ensure_ascii=False)

    @staticmethod
    def _load_json(text: str | None) -> Any:
        if text is None or text == "":
            return None
        return json.loads(text)

    # ------------------------------------------------------------------
    # Source operations
    # ------------------------------------------------------------------
    def save_source(self, source: Source) -> Source:
        self._exec(
            """
            INSERT INTO sources (id, tenant_id, type, version, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tenant_id=excluded.tenant_id,
                type=excluded.type,
                version=excluded.version,
                metadata=excluded.metadata
            """,
            (
                str(source.id),
                source.tenant_id,
                source.type,
                source.version,
                self._dump_json(source.metadata),
            ),
        )
        return source

    def get_source(self, source_id: UUID) -> Source | None:
        row = self._exec(
            "SELECT * FROM sources WHERE id = ?", (str(source_id),)
        ).fetchone()
        if row is None:
            return None
        return Source(
            id=UUID(row["id"]),
            tenant_id=row["tenant_id"],
            type=row["type"],
            version=row["version"],
            metadata=self._load_json(row["metadata"]) or {},
        )

    def list_sources(self, tenant_id: str | None = None) -> list[Source]:
        if tenant_id is not None:
            rows = self._exec(
                "SELECT * FROM sources WHERE tenant_id = ? ORDER BY id",
                (tenant_id,),
            ).fetchall()
        else:
            rows = self._exec("SELECT * FROM sources ORDER BY id").fetchall()
        return [
            Source(
                id=UUID(row["id"]),
                tenant_id=row["tenant_id"],
                type=row["type"],
                version=row["version"],
                metadata=self._load_json(row["metadata"]) or {},
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------
    def save_document(self, document: Document) -> Document:
        self._exec(
            """
            INSERT INTO documents (id, source_id, checksum, status, metadata)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id,
                checksum=excluded.checksum,
                status=excluded.status,
                metadata=excluded.metadata
            """,
            (
                str(document.id),
                str(document.source_id),
                document.checksum,
                document.status,
                self._dump_json(document.metadata),
            ),
        )
        return document

    def get_document(self, document_id: UUID) -> Document | None:
        row = self._exec(
            "SELECT * FROM documents WHERE id = ?", (str(document_id),)
        ).fetchone()
        if row is None:
            return None
        return Document(
            id=UUID(row["id"]),
            source_id=UUID(row["source_id"]),
            checksum=row["checksum"],
            status=row["status"],
            metadata=self._load_json(row["metadata"]) or {},
        )

    def list_documents(
        self,
        source_id: UUID | None = None,
        status: str | None = None,
    ) -> list[Document]:
        conditions: list[str] = []
        params: list[Any] = []

        if source_id is not None:
            conditions.append("source_id = ?")
            params.append(str(source_id))
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self._exec(
            f"SELECT * FROM documents {where_clause} ORDER BY id", params
        ).fetchall()

        return [
            Document(
                id=UUID(row["id"]),
                source_id=UUID(row["source_id"]),
                checksum=row["checksum"],
                status=row["status"],
                metadata=self._load_json(row["metadata"]) or {},
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------
    def save_chunk(self, chunk: Chunk) -> Chunk:
        self._exec(
            """
            INSERT INTO chunks (id, document_id, chunk_order, content, token_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_order=excluded.chunk_order,
                content=excluded.content,
                token_count=excluded.token_count,
                metadata=excluded.metadata
            """,
            (
                str(chunk.id),
                str(chunk.document_id),
                chunk.chunk_order,
                chunk.content,
                chunk.token_count,
                self._dump_json(chunk.metadata),
            ),
        )
        return chunk

    def get_chunk(self, chunk_id: UUID) -> Chunk | None:
        row = self._exec(
            "SELECT * FROM chunks WHERE id = ?", (str(chunk_id),)
        ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=UUID(row["id"]),
            document_id=UUID(row["document_id"]),
            chunk_order=row["chunk_order"],
            content=row["content"],
            token_count=row["token_count"],
            metadata=self._load_json(row["metadata"]) or {},
        )

    def list_chunks(self, document_id: UUID) -> list[Chunk]:
        rows = self._exec(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_order",
            (str(document_id),),
        ).fetchall()
        return [
            Chunk(
                id=UUID(row["id"]),
                document_id=UUID(row["document_id"]),
                chunk_order=row["chunk_order"],
                content=row["content"],
                token_count=row["token_count"],
                metadata=self._load_json(row["metadata"]) or {},
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------
    def save_index_entry(self, entry: IndexEntry) -> IndexEntry:
        self._exec(
            """
            INSERT INTO index_entries (chunk_id, dense_vector, sparse_vector, metadata)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                dense_vector=excluded.dense_vector,
                sparse_vector=excluded.sparse_vector,
                metadata=excluded.metadata
            """,
            (
                str(entry.chunk_id),
                self._dump_json(entry.dense_vector) if entry.dense_vector is not None else None,
                self._dump_json(entry.sparse_vector) if entry.sparse_vector is not None else None,
                self._dump_json(entry.metadata),
            ),
        )
        return entry

    def get_index_entry(self, chunk_id: UUID) -> IndexEntry | None:
        row = self._exec(
            "SELECT * FROM index_entries WHERE chunk_id = ?", (str(chunk_id),)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_index_entry(row)

    def list_index_entries(self, chunk_ids: list[UUID] | None = None) -> list[IndexEntry]:
        if chunk_ids is None:
            rows = self._exec(
                "SELECT * FROM index_entries ORDER BY chunk_id"
            ).fetchall()
        else:
            placeholders = ",".join("?" for _ in chunk_ids)
            rows = self._exec(
                f"SELECT * FROM index_entries WHERE chunk_id IN ({placeholders})",
                [str(cid) for cid in chunk_ids],
            ).fetchall()
        return [self._row_to_index_entry(row) for row in rows]

    def _row_to_index_entry(self, row: sqlite3.Row) -> IndexEntry:
        sparse_loaded = self._load_json(row["sparse_vector"])
        if isinstance(sparse_loaded, dict):
            sparse_vector = {int(k): v for k, v in sparse_loaded.items()}
        else:
            sparse_vector = sparse_loaded

        return IndexEntry(
            chunk_id=UUID(row["chunk_id"]),
            dense_vector=self._load_json(row["dense_vector"]),
            sparse_vector=sparse_vector,
            metadata=self._load_json(row["metadata"]) or {},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._connection.close()

    def __del__(self) -> None:
        try:
            self._connection.close()
        except Exception:
            pass

    def __enter__(self) -> "SQLiteStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
