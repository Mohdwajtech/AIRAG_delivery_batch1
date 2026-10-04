"""wk10_pipeline.py — shared helpers for W10 Track A notebooks.

Reuses the W9 Acme Analytics Platform corpus (thematic continuity).
Adds:
  - Semantic cache primitives (SQLite backend)
  - Prompt-token accounting helper
  - Tombstone + IsEmpty payload filter helpers
  - Cost/latency measurement helpers
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

# ─── Corpus paths (same as W9) ──────────────────────────────────────

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
ACME_CORPUS_PATH  = SAMPLE_DOCS_DIR / "acme_docs.jsonl"
ACME_QUERIES_PATH = SAMPLE_DOCS_DIR / "acme_test_queries.jsonl"


def load_acme_corpus() -> list[dict]:
    """Return the 15-doc Acme corpus (same as W9)."""
    assert ACME_CORPUS_PATH.exists(), (
        f"Missing {ACME_CORPUS_PATH.name}. "
        f"Copy from W9's sample_docs/ folder or run W9's generate_acme_corpus.py."
    )
    return [json.loads(line) for line in ACME_CORPUS_PATH.read_text().splitlines() if line]


def load_test_queries() -> list[dict]:
    """Return the 6 test queries with expected-winner annotations (same as W9)."""
    assert ACME_QUERIES_PATH.exists(), f"Missing {ACME_QUERIES_PATH.name}"
    return [json.loads(line) for line in ACME_QUERIES_PATH.read_text().splitlines() if line]


# ─── Prompt token accounting (for Day 1 Cell 2) ─────────────────────

def build_rag_prompt(question: str, retrieved_chunks: list[dict],
                     system_prompt: str = None) -> tuple[str, dict]:
    """Assemble a canonical RAG prompt from the parts and return the prompt + a
    per-section token count.
    
    Uses tiktoken if available; falls back to whitespace approximation.
    Returns (full_prompt, {"system": N, "retrieved": N, "question": N, "total": N}).
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant for the Acme Analytics Platform. "
            "Answer the user's question using ONLY the provided context. "
            "If the context does not contain the answer, say 'I don't know'. "
            "Be concise. Cite the relevant section titles when helpful. "
            "Always ground your answer in the retrieved context — do not fabricate details. "
            "If multiple context chunks are relevant, synthesise across them faithfully."
        )
    
    context_block = "\n\n".join(
        f"[{i+1}] {c['doc'].get('title', '')}\n{c['doc'].get('text', '')}"
        for i, c in enumerate(retrieved_chunks)
    )
    
    full = (
        f"{system_prompt}\n\n"
        f"=== Retrieved Context ===\n{context_block}\n\n"
        f"=== Question ===\n{question}\n\n"
        f"=== Answer ==="
    )
    
    counts = {
        "system":    count_tokens(system_prompt),
        "retrieved": count_tokens(context_block),
        "question":  count_tokens(question),
    }
    counts["total"] = sum(counts.values())
    return full, counts


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Token count via tiktoken if usable; else whitespace approximation.
    
    Catches any tiktoken failure (missing package, cached-encoding not
    downloadable, unknown model) — the approximation is always safe.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Approximation: ~1.3 tokens per whitespace-separated word for English
        return int(len(text.split()) * 1.3)


# ─── Semantic cache (SQLite-backed, for Day 1 Cell 5-6) ─────────────

CACHE_DB_DEFAULT = "wk10_query_cache.db"

def cache_init(db_path: str = CACHE_DB_DEFAULT) -> sqlite3.Connection:
    """Create the query_cache table if absent; return an open connection."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id                 TEXT PRIMARY KEY,
            question           TEXT NOT NULL,
            embedding_blob     BLOB NOT NULL,
            answer             TEXT NOT NULL,
            hit_count          INTEGER NOT NULL DEFAULT 0,
            created_ts         REAL NOT NULL,
            last_hit_ts        REAL
        )
    """)
    conn.commit()
    return conn


def cache_clear(conn: sqlite3.Connection) -> int:
    """Wipe the cache table. Returns rows deleted. Used on document update."""
    cur = conn.execute("DELETE FROM query_cache")
    conn.commit()
    return cur.rowcount


def cache_key(question: str) -> str:
    """Deterministic id for a question — sha256 of normalised question."""
    normalised = question.lower().strip()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def cache_put(conn: sqlite3.Connection, question: str,
              embedding: list[float], answer: str) -> None:
    """Insert (or replace) a cache row."""
    import struct
    emb_blob = struct.pack(f"{len(embedding)}f", *embedding)
    conn.execute("""
        INSERT OR REPLACE INTO query_cache
        (id, question, embedding_blob, answer, hit_count, created_ts, last_hit_ts)
        VALUES (?, ?, ?, ?, 0, ?, NULL)
    """, (cache_key(question), question, emb_blob, answer, time.time()))
    conn.commit()


def cache_lookup(conn: sqlite3.Connection, query_embedding: list[float],
                 threshold: float = 0.95) -> dict | None:
    """Scan all cache entries; return the best cosine match if ≥ threshold, else None.
    
    On a hit: increment hit_count, update last_hit_ts, and include the similarity
    score in the returned dict.
    
    On production scale this would be a vector index (Qdrant, pgvector), not a
    full scan. For teaching + capstone scale (<1K cached questions) a scan is fine.
    """
    import math
    import struct
    
    def cosine(a: list[float], b: list[float]) -> float:
        dot   = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(y*y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    best_score, best_row = -1.0, None
    for row in conn.execute("SELECT id, question, embedding_blob, answer FROM query_cache"):
        row_id, row_q, emb_blob, row_ans = row
        n_floats = len(emb_blob) // 4
        row_emb = list(struct.unpack(f"{n_floats}f", emb_blob))
        s = cosine(query_embedding, row_emb)
        if s > best_score:
            best_score, best_row = s, (row_id, row_q, row_ans)
    
    if best_row is None or best_score < threshold:
        return None
    
    row_id, matched_q, cached_ans = best_row
    conn.execute("""
        UPDATE query_cache 
        SET hit_count = hit_count + 1, last_hit_ts = ?
        WHERE id = ?
    """, (time.time(), row_id))
    conn.commit()
    
    return {
        "answer":         cached_ans,
        "matched_question": matched_q,
        "similarity":     best_score,
        "cache_hit":      True,
    }


# ─── Tombstone helpers (for Day 2) ──────────────────────────────────

def make_live_filter():
    """Return a Qdrant Filter that matches only live (not-yet-tombstoned) chunks.
    
    Uses IsEmptyCondition (NOT IsNullCondition). Rationale:
    - IsNullCondition matches ONLY rows where the field EXISTS with value null.
    - IsEmptyCondition matches rows where the field is absent OR null OR empty.
    
    Because fresh chunks may not have `deleted_at` set at all (depending on
    ingestion path), IsEmpty is the robust choice. Using IsNull creates a
    silent bug: live chunks get filtered OUT, retrieval returns nothing.
    
    We learn this the hard way in Day 2 Cell 4 (bug demo) and fix in Cell 5.
    """
    from qdrant_client.models import Filter, IsEmptyCondition, PayloadField
    return Filter(must=[
        IsEmptyCondition(is_empty=PayloadField(key="deleted_at"))
    ])


def make_broken_filter():
    """Return the BROKEN filter (IsNullCondition) — for the Day 2 Cell 4 bug demo.
    
    DON'T USE IN PRODUCTION. Kept here for pedagogical purposes only.
    """
    from qdrant_client.models import Filter, IsNullCondition, PayloadField
    return Filter(must=[
        IsNullCondition(is_null=PayloadField(key="deleted_at"))
    ])


# ─── Cost + latency measurement (for Day 1 Cell 9 + Day 2 Cell 12) ──

# Prices as of Aug 2026, per 1M tokens
PRICING = {
    "text-embedding-3-small":  0.02,
    "text-embedding-3-large":  0.13,
    "gpt-4o-mini-input":       0.15,
    "gpt-4o-mini-output":      0.60,
    "gpt-4o-mini-cached":      0.075,   # 50% off input on prompt cache hit
}


def estimate_cost_usd(n_input_tokens: int, n_output_tokens: int,
                      n_cached_input_tokens: int = 0,
                      n_embed_tokens: int = 0,
                      embed_model: str = "text-embedding-3-small") -> float:
    """Estimate one query's cost from token counts.
    
    Regular input tokens billed at full rate; cached input tokens at 50% off
    (OpenAI's implicit auto-cache behaviour on gpt-4o-mini as of Aug 2026).
    """
    fresh_input = max(0, n_input_tokens - n_cached_input_tokens)
    embed_cost  = n_embed_tokens        * PRICING[embed_model]         / 1_000_000
    input_cost  = fresh_input           * PRICING["gpt-4o-mini-input"] / 1_000_000
    cached_cost = n_cached_input_tokens * PRICING["gpt-4o-mini-cached"] / 1_000_000
    output_cost = n_output_tokens       * PRICING["gpt-4o-mini-output"] / 1_000_000
    return embed_cost + input_cost + cached_cost + output_cost
