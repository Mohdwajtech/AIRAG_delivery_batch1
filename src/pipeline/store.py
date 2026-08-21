"""Tiny SQLite persistence — two tables, one writer per table.

Schema:
  runs      — one row per pipeline execution (mirrors RunSummary fields)
  answers   — one row per LLM call, FK-linked to runs.id
  eval_runs — one row per judged answer (Week 5 evaluation harness)
  rag_runs  — one row per RAG answer over the golden set (Week 6)
"""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from .pipeline import Answer
from .settings import RunSummary


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       REAL    NOT NULL,
    elapsed_seconds  REAL    NOT NULL,
    n_questions      INTEGER NOT NULL,
    n_succeeded      INTEGER NOT NULL,
    n_retries_total  INTEGER NOT NULL,
    total_cost_usd   REAL    NOT NULL,
    fail_rate        REAL    NOT NULL,
    use_fake         INTEGER NOT NULL                       -- 0 / 1 (SQLite has no native bool)
);

CREATE TABLE IF NOT EXISTS answers (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL,                          -- ties an answer to its run
    question     TEXT    NOT NULL,
    answer       TEXT    NOT NULL,
    cost_usd     REAL    NOT NULL,
    retries      INTEGER NOT NULL DEFAULT 0,
    model        TEXT,                                      -- W4: which model produced it
    confidence   REAL,                                      -- W4: structured-output field
    sources_json TEXT,                                      -- W4: JSON-encoded sources list
    ts           REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL,                          -- groups one eval run
    golden_id    TEXT,                                      -- which golden entry
    question     TEXT,
    candidate    TEXT,                                      -- the answer that was judged
    accuracy     INTEGER,
    groundedness INTEGER,
    format       INTEGER,
    reasoning    TEXT,
    ts           REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS rag_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL,                          -- groups one RAG eval run
    golden_id    TEXT,
    question     TEXT,
    answer       TEXT,
    sources_json TEXT,                                      -- retrieved chunk ids
    hit          INTEGER,                                   -- retrieval hit: expected source in top-k (0/1)
    latency_ms   REAL,
    cost_usd     REAL,
    ts           REAL    NOT NULL
);
"""


def connect(path: str | Path = "results.db") -> sqlite3.Connection:
    """Open (or create) the database, ensure both tables exist, return the connection."""
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    con.commit()
    return con


def write_run(con: sqlite3.Connection, summary: RunSummary) -> int:
    """Insert one row into `runs`. Returns the new row id (use for write_answers)."""
    cur = con.execute(
        "INSERT INTO runs (started_at, elapsed_seconds, n_questions, n_succeeded, "
        "                  n_retries_total, total_cost_usd, fail_rate, use_fake) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            summary.started_at,
            summary.elapsed_seconds,
            summary.n_questions,
            summary.n_succeeded,
            summary.n_retries_total,
            summary.total_cost_usd,
            summary.fail_rate,
            1 if summary.use_fake else 0,
        ),
    )
    con.commit()
    return cur.lastrowid


def write_answers(
    con: sqlite3.Connection,
    run_id: int,
    answers: Iterable[Answer],
    model: str = "",
) -> int:
    """Bulk-insert all answers for a given run. Returns the number of rows inserted."""
    import json
    ts = time.time()
    rows = [
        (run_id, a.question, a.text, a.cost_usd, a.retries,
         model, getattr(a, "confidence", None),
         json.dumps(getattr(a, "sources", [])), ts)
        for a in answers
    ]
    con.executemany(
        "INSERT INTO answers (run_id, question, answer, cost_usd, retries, "
        "                     model, confidence, sources_json, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    return len(rows)


def write_eval_run(
    con: sqlite3.Connection,
    run_id: int,
    golden_id: str,
    question: str,
    candidate: str,
    scores: dict,
) -> None:
    """Insert one judged answer into `eval_runs` (Week 5)."""
    con.execute(
        "INSERT INTO eval_runs (run_id, golden_id, question, candidate, "
        "                       accuracy, groundedness, format, reasoning, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, golden_id, question, candidate,
         scores.get("accuracy"), scores.get("groundedness"),
         scores.get("format"), scores.get("reasoning"), time.time()),
    )
    con.commit()


def write_rag_run(
    con: sqlite3.Connection,
    run_id: int,
    golden_id: str,
    question: str,
    answer: str,
    sources: list[str],
    hit: int,
    latency_ms: float,
    cost_usd: float,
) -> None:
    """Insert one RAG answer over the golden set into `rag_runs` (Week 6)."""
    import json
    con.execute(
        "INSERT INTO rag_runs (run_id, golden_id, question, answer, sources_json, "
        "                      hit, latency_ms, cost_usd, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, golden_id, question, answer, json.dumps(sources),
         hit, latency_ms, cost_usd, time.time()),
    )
    con.commit()
