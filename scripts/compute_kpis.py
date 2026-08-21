"""Week 6 - compute the weekly KPI snapshot from the latest rag_runs.

Fills the machine-measurable rows (retrieval hit rate, cost/query, latency
p50/p95) and writes docs/kpi/wk6-snapshot.md. Task success (eyeball) and
grounded rate (LLM-as-judge) are filled in by you / scripts/run_eval.py.

Usage:  python scripts/compute_kpis.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import statistics

from src.pipeline import store

REPORT = "docs/kpi/wk6-snapshot.md"


def main() -> None:
    con = store.connect()
    row = con.execute("SELECT MAX(run_id) FROM rag_runs").fetchone()
    if not row or row[0] is None:
        print("No rag_runs yet - run scripts/run_rag_eval.py first.")
        return
    run_id = row[0]
    rows = con.execute(
        "SELECT hit, latency_ms, cost_usd FROM rag_runs WHERE run_id=?", (run_id,)
    ).fetchall()

    n = len(rows)
    hits = sum(r[0] for r in rows)
    lat = sorted(r[1] for r in rows)
    cost = statistics.mean(r[2] for r in rows) if n else 0.0
    p = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0
    p50, p95 = p(lat, 0.50) / 1000, p(lat, 0.95) / 1000

    table = f"""# KPI Snapshot - Week 6 - Naive RAG baseline

| KPI                     | Value        | How measured               | Notes                |
|-------------------------|--------------|----------------------------|----------------------|
| Task success rate       | X / {n}       | Eyeball (Y/N/Partial)      | fill in manually     |
| Grounded response rate  | X%           | LLM-as-judge (gpt-4o)      | run scripts/run_eval |
| Retrieval hit rate      | {hits} / {n}      | Expected source in top-3   |                      |
| Cost / query            | ${cost:.4f}    | Sum token costs / {n}       | embedding dominates  |
| Latency p50 / p95       | {p50:.2f}s / {p95:.2f}s | time per /ask_batched | mostly the LLM call  |
| Tool / function success | N/A          | not applicable yet (W11+)  |                      |
| Escalation rate         | N/A          | not applicable yet (W13+)  |                      |
| Stakeholder NPS         | N/A          | not applicable yet         |                      |
"""
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"Wrote {REPORT}  (retrieval hit rate {hits}/{n}, cost ${cost:.4f}, "
          f"p50 {p50:.2f}s / p95 {p95:.2f}s)")


if __name__ == "__main__":
    main()
