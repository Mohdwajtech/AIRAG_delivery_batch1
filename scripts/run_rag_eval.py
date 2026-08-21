"""Week 6 - run the golden set through the RAG app and record RAG metrics.

For each golden question: POST to /ask_batched (now RAG), capture the answer +
retrieved sources, compute a retrieval hit (expected source in the retrieved
chunk ids), time it, and persist to the rag_runs table.

Prereqs:
  * index built:   python scripts/build_index.py
  * API running:   uvicorn api.main:app --port 8000
  * OPENAI_API_KEY set  (or run the whole stack in fake mode: USE_FAKE=1)

Usage:  python scripts/run_rag_eval.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import time

import httpx

from src.eval.golden import load_golden
from src.pipeline import store

API_URL = "http://localhost:8000/ask_batched"


def main() -> None:
    golden = load_golden()
    con = store.connect()                      # creates rag_runs if absent
    run_id = int(time.time())

    hits = 0
    for g in golden:
        t0 = time.time()
        r = httpx.post(API_URL, json={"question": g.question}, timeout=60.0)
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.time() - t0) * 1000

        answer  = data["content"]
        sources = data.get("sources", [])
        # retrieval hit: did a chunk from the expected source doc get retrieved?
        stems = {s.split("#")[0] for s in sources}
        hit = 1 if (g.expected_source and g.expected_source in stems) else 0
        hits += hit

        store.write_rag_run(con, run_id, g.id, g.question, answer, sources,
                            hit, latency_ms, data.get("cost_usd", 0.0))
        print(f"{g.id}: hit={hit} sources={sorted(stems)}  | {answer[:55]}")

    n = len(golden)
    print(f"\nrun_id={run_id}  retrieval hit rate = {hits}/{n}  "
          f"(rows written to rag_runs)")


if __name__ == "__main__":
    main()
