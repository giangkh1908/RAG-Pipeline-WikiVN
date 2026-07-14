"""SQLite-backed cache for LLM query preprocessing results."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class CachedQuery:
    """A cached query preprocessing result."""

    raw_query: str
    rewritten_query: str
    intent: str
    model_name: str
    prompt_version: str


class QueryCache:
    """Cache rewritten queries and intents in SQLite.

    The cache key is derived from the model name, prompt version, and raw
    query text so that changes to either invalidate previous entries.
    """

    def __init__(self, storage: "SQLiteStorage") -> None:
        """Cache rewritten queries and intents in SQLite.

        The cache key is derived from the model name, prompt version, and raw
        query text so that changes to either invalidate previous entries.
        """
        self._storage = storage
        self._connection = storage._connection  # type: ignore[attr-defined]
        self._lock = storage._lock  # type: ignore[attr-defined]
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key TEXT PRIMARY KEY,
            raw_query TEXT NOT NULL,
            rewritten_query TEXT NOT NULL,
            intent TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_query_cache_created_at
            ON query_cache(created_at);
        """
        with self._lock:
            self._connection.executescript(ddl)

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def make_key(model_name: str, prompt_version: str, raw_query: str) -> str:
        """Build a deterministic cache key."""
        payload = f"{model_name}|{prompt_version}|{raw_query}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(
        self,
        model_name: str,
        prompt_version: str,
        raw_query: str,
        ttl_days: int | None = 30,
    ) -> CachedQuery | None:
        """Return a cached result if it exists and is not expired."""
        cache_key = self.make_key(model_name, prompt_version, raw_query)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM query_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        if ttl_days is not None:
            created = datetime.fromisoformat(row["created_at"])
            if datetime.now(timezone.utc) - created > timedelta(days=ttl_days):
                return None

        return CachedQuery(
            raw_query=row["raw_query"],
            rewritten_query=row["rewritten_query"],
            intent=row["intent"],
            model_name=row["model_name"],
            prompt_version=row["prompt_version"],
        )

    def set(self, cached: CachedQuery) -> None:
        """Persist a query preprocessing result."""
        cache_key = self.make_key(cached.model_name, cached.prompt_version, cached.raw_query)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO query_cache
                    (cache_key, raw_query, rewritten_query, intent, model_name,
                     prompt_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    raw_query=excluded.raw_query,
                    rewritten_query=excluded.rewritten_query,
                    intent=excluded.intent,
                    model_name=excluded.model_name,
                    prompt_version=excluded.prompt_version,
                    created_at=excluded.created_at
                """,
                (
                    cache_key,
                    cached.raw_query,
                    cached.rewritten_query,
                    cached.intent,
                    cached.model_name,
                    cached.prompt_version,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def clear_expired(self, ttl_days: int = 30) -> int:
        """Remove expired entries. Returns number deleted."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
        with self._lock:
            cursor = self._connection.execute(
                "DELETE FROM query_cache WHERE created_at < ?",
                (cutoff,),
            )
        return cursor.rowcount

    def clear_all(self) -> int:
        """Remove all cached entries. Returns number deleted."""
        with self._lock:
            cursor = self._connection.execute("DELETE FROM query_cache")
        return cursor.rowcount
