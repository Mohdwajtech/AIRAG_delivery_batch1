"""src/rag/cache.py — Semantic query cache (Week 10).

Three-layer caching strategy:
  Layer 1 (this file): Semantic cache — SQLite-backed, cosine ≥ 0.95.
      If a new question's embedding is close enough to a cached question,
      return the cached answer instantly (no retrieve/generate).
  Layer 2: Prompt cache — automatic from OpenAI (50% off repeated prefixes).
  Layer 3: Embedding cache — file-system cache in qdrant_store.py (W7).

Cache invalidation rules (NON-NEGOTIABLE):
  - Clear on EVERY document update (tombstone or re-ingest).
  - Tag each entry with kb_version — reject mismatches on lookup.
  - Never serve a stale answer — when in doubt, miss.

Usage:
    from src.rag.cache import SemanticCache
    cache = SemanticCache("results.db")
    hit = cache.lookup(query_vector, kb_version="v2.w10", threshold=0.95)
    if hit:
        return hit["answer"]  # skip retrieve + generate
    else:
        answer = run_full_pipeline(question)
        cache.store(query_vector, question, answer, kb_version="v2.w10")
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import numpy as np


# ── Schema ────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    question     TEXT    NOT NULL,
    answer       TEXT    NOT NULL,
    vector_json  TEXT    NOT NULL,
    kb_version   TEXT    NOT NULL,
    created_at   REAL    NOT NULL,
    hit_count    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_cache_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SemanticCache:
    """SQLite-backed semantic query cache with cosine similarity matching."""

    def __init__(self, db_path: str | Path = "results.db"):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # ── Lookup ────────────────────────────────────────────────

    def lookup(
        self,
        query_vector: list[float],
        kb_version: str,
        threshold: float = 0.95,
    ) -> dict | None:
        """Find a cached answer with cosine similarity ≥ threshold.

        Returns {"answer", "matched_question", "similarity", "cache_hit": True}
        or None on miss.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id, question, answer, vector_json "
                "FROM query_cache WHERE kb_version = ?",
                (kb_version,),
            ).fetchall()

            if not rows:
                return None

            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return None

            best_sim = -1.0
            best_row = None

            for row_id, question, answer, vec_json in rows:
                cached_vec = np.array(json.loads(vec_json), dtype=np.float32)
                c_norm = np.linalg.norm(cached_vec)
                if c_norm == 0:
                    continue
                sim = float(np.dot(q_vec, cached_vec) / (q_norm * c_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_row = (row_id, question, answer)

            if best_row and best_sim >= threshold:
                row_id, question, answer = best_row
                conn.execute(
                    "UPDATE query_cache SET hit_count = hit_count + 1 WHERE id = ?",
                    (row_id,),
                )
                conn.commit()
                return {
                    "answer": answer,
                    "matched_question": question,
                    "similarity": round(best_sim, 4),
                    "cache_hit": True,
                }

            return None
        finally:
            conn.close()

    # ── Store ─────────────────────────────────────────────────

    def store(
        self,
        query_vector: list[float],
        question: str,
        answer: str,
        kb_version: str,
    ) -> None:
        """Cache a query→answer pair with its embedding vector."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO query_cache (question, answer, vector_json, kb_version, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, answer, json.dumps(query_vector), kb_version, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Invalidation (NON-NEGOTIABLE on every doc update) ────

    def clear(self, kb_version: str | None = None) -> int:
        """Clear cache entries. If kb_version given, clear only that version.

        Call this on EVERY document update — tombstone, re-ingest, or delete.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if kb_version:
                result = conn.execute(
                    "DELETE FROM query_cache WHERE kb_version = ?",
                    (kb_version,),
                )
            else:
                result = conn.execute("DELETE FROM query_cache")
            conn.commit()
            return result.rowcount
        finally:
            conn.close()

    def clear_all(self) -> int:
        """Clear ALL cache entries regardless of version."""
        return self.clear(kb_version=None)

    # ── Stats ─────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cache statistics."""
        conn = sqlite3.connect(self.db_path)
        try:
            total = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            total_hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM query_cache").fetchone()[0]
            versions = conn.execute(
                "SELECT kb_version, COUNT(*) FROM query_cache GROUP BY kb_version"
            ).fetchall()
            return {
                "total_entries": total,
                "total_hits": total_hits,
                "versions": {v: c for v, c in versions},
            }
        finally:
            conn.close()
