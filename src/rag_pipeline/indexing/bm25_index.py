"""BM25 index for keyword-based retrieval using SQLite FTS5."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class BM25Index:
    """BM25 index backed by SQLite FTS5 with Vietnamese tokenization.

    Features:
    - Instant open (no RAM load)
    - Incremental insert (INSERT into SQLite)
    - FTS5 full-text search with rank scoring
    - Pre-tokenized with underthesea for Vietnamese

    Thread safety: NOT thread-safe. Callers must ensure single-threaded
    access or external synchronization. In the ingest pipeline, all BM25
    operations run on the main thread.

    Supports:
    - underthesea: Vietnamese word segmentation (recommended)
    - pyvi: Vietnamese tokenizer (faster, less accurate)
    - simple: lowercase split (fallback)
    """

    index_path: Path
    tokenizer_name: str = "underthesea"
    _conn: sqlite3.Connection | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.open_existing()

    # ── Connection management ───────────────────────────────────────

    def open_existing(self) -> bool:
        """Open SQLite file if it exists. Returns True if successful."""
        if not self.index_path.exists():
            return False
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.index_path))
        return True

    def _ensure_connection(self) -> None:
        """Create tables if connection is fresh."""
        if self._conn is None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.index_path))
            self._create_tables()

    def _create_tables(self) -> None:
        """Create chunks table and FTS5 virtual table."""
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                raw_content TEXT NOT NULL,
                full_text TEXT NOT NULL,
                section_path TEXT,
                checksum TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                raw_content,
                tokenize='unicode61'
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Insert ──────────────────────────────────────────────────────

    def insert(
        self,
        chunk_id: str,
        doc_id: str,
        raw_content: str,
        full_text: str,
        section_path: list[str],
        checksum: str,
    ) -> None:
        """Insert a single chunk. Calls _ensure_connection on first use."""
        self._ensure_connection()
        assert self._conn is not None

        tokenized = self._tokenize(raw_content)
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, doc_id, raw_content, full_text, section_path, checksum) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (chunk_id, doc_id, raw_content, full_text, json.dumps(section_path, ensure_ascii=False), checksum),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO chunks_fts (chunk_id, raw_content) VALUES (?, ?)",
            (chunk_id, tokenized),
        )
        self._conn.commit()

    def insert_batch(
        self,
        items: Iterable[dict[str, Any]],
    ) -> int:
        """Insert multiple chunks in a single transaction.

        Each item must have keys:
            chunk_id, doc_id, raw_content, full_text, section_path, checksum

        Returns the number of inserted rows.
        """
        self._ensure_connection()
        assert self._conn is not None

        count = 0
        cursor = self._conn.cursor()
        try:
            for item in items:
                tokenized = self._tokenize(item["raw_content"])
                cursor.execute(
                    "INSERT OR REPLACE INTO chunks "
                    "(chunk_id, doc_id, raw_content, full_text, section_path, checksum) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        item["chunk_id"],
                        item["doc_id"],
                        item["raw_content"],
                        item["full_text"],
                        json.dumps(item["section_path"], ensure_ascii=False),
                        item["checksum"],
                    ),
                )
                cursor.execute(
                    "INSERT OR REPLACE INTO chunks_fts (chunk_id, raw_content) VALUES (?, ?)",
                    (item["chunk_id"], tokenized),
                )
                count += 1
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        return count

    # ── Search ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, str, float, str]]:
        """Search the index and return (chunk_id, doc_id, score, full_text).

        Results are sorted by BM25 score descending.
        """
        if self._conn is None:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query.strip():
            return []

        try:
            rows = self._conn.execute(
                """SELECT c.chunk_id, c.doc_id, f.rank, c.full_text
                   FROM chunks_fts f
                   JOIN chunks c ON c.chunk_id = f.chunk_id
                   WHERE chunks_fts MATCH ?
                   ORDER BY f.rank
                   LIMIT ?""",
                (tokenized_query, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS5 MATCH can fail on malformed queries
            return []

        # SQLite FTS5 rank is negative (lower = better), convert to positive score
        return [(r[0], r[1], abs(r[2]), r[3]) for r in rows]

    # ── Tokenizer ───────────────────────────────────────────────────

    def _tokenize(self, text: str) -> str:
        """Pre-tokenize Vietnamese text for FTS5 indexing.

        Returns space-separated tokens suitable for FTS5 unicode61 tokenizer.
        """
        if self.tokenizer_name == "underthesea":
            try:
                from underthesea import word_tokenize
                return " ".join(word_tokenize(text, format="text").split())
            except ImportError:
                pass

        if self.tokenizer_name == "pyvi":
            try:
                from pyvi import ViTokenizer
                return " ".join(ViTokenizer.tokenize(text).split())
            except ImportError:
                pass

        # Fallback: simple lowercase split
        return " ".join(text.lower().split())

    # ── Properties ──────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Return True if the index is ready for search."""
        return self._conn is not None

    @property
    def doc_count(self) -> int:
        """Return the number of chunks in the index."""
        if self._conn is None:
            return 0
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    # ── Legacy compatibility ────────────────────────────────────────

    def build(self, documents: Iterable[tuple[str, str]]) -> None:
        """Build index from (doc_id, text) pairs. For test compatibility.

        In production, use insert() or insert_batch() instead.
        """
        self._ensure_connection()
        items = []
        for doc_id, text in documents:
            items.append({
                "chunk_id": doc_id,
                "doc_id": doc_id,
                "raw_content": text,
                "full_text": text,
                "section_path": [],
                "checksum": "",
            })
        self.insert_batch(items)

    def load(self) -> bool:
        """Load existing index. For backward compatibility."""
        return self.open_existing()
