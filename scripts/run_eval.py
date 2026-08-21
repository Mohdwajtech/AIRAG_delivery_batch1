"""Week 5 - run the golden set through the app and judge every answer.

Flow:  load golden set -> POST each question to /ask_batched -> judge the answer
against the ideal -> persist to eval_runs -> write docs/eval-run-001.md.

Prereqs:
  * the API running:   uvicorn api.main:app --port 8000
  * OPENAI_API_KEY set (real judge = gpt-4o), OR run offline: USE_FAKE=1 python scripts/run_eval.py

Usage:  python scripts/run_eval.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import statistics
import time

import httpx

from src.eval.golden import load_golden
from src.eval.judge import judge
from src.pipeline import store

API_URL = "http://localhost:8000/ask_batched"
REPORT  = "docs/eval-run-001.md"


def get_candidate(question: str) -> tuple[str, list[str]]:
    """Ask the running app for an answer."""
    r = httpx.post(API_URL, json={"question": question}, timeout=60.0)
    r.raise_for_status()
    data = r.json()
    return data["content"], data.get("sources", [])


def main() -> None:
    golden = load_golden()
    con = store.connect()                      # creates eval_runs if absent
    run_id = int(time.time())

    rows, accs, grds, fmts = [], [], [], []
    for g in golden:
        candidate, _sources = get_candidate(g.question)
        s = judge(g.question, g.ideal_answer, candidate, g.must_mention)
        store.write_eval_run(con, run_id, g.id, g.question, candidate, s)
        rows.append((g.id, candidate, s))
        accs.append(s["accuracy"]); grds.append(s["groundedness"]); fmts.append(s["format"])
        print(f"{g.id}: acc={s['accuracy']} grnd={s['groundedness']} fmt={s['format']}  "
              f"| {candidate[:60]}")

    n = len(rows)
    avg = lambda xs: round(statistics.mean(xs), 2) if xs else 0.0
    lines = [
        f"# Eval run 001 - baseline ({n} golden questions)",
        "",
        f"- run_id: `{run_id}`",
        f"- avg accuracy:     **{avg(accs)}** / 4",
        f"- avg groundedness: **{avg(grds)}** / 4",
        f"- avg format:       **{avg(fmts)}** / 4",
        "",
        "| id | acc | grnd | fmt | candidate |",
        "|----|-----|------|-----|-----------|",
    ]
    for gid, cand, s in rows:
        c = cand.replace("|", "/")[:70]
        lines.append(f"| {gid} | {s['accuracy']} | {s['groundedness']} | {s['format']} | {c} |")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {REPORT}  (avg accuracy {avg(accs)}/4 over {n} questions)")


if __name__ == "__main__":
    main()
