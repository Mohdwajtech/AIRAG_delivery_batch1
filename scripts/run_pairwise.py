"""Week 5 - pairwise sanity check with position-flip.

For each golden question, compare the app's answer against the golden IDEAL
answer using pairwise_consistent (which always flips position). Reports how
often the app answer wins / ties / loses vs the ideal. Extend this to compare
your own prompt v1 vs v2 answers.

Prereqs: API running + OPENAI_API_KEY (or USE_FAKE=1 for an offline dry run).
Usage:   python scripts/run_pairwise.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx

from src.eval.golden import load_golden
from src.eval.pairwise import pairwise_consistent

API_URL = "http://localhost:8000/ask_batched"


def get_candidate(question: str) -> str:
    r = httpx.post(API_URL, json={"question": question}, timeout=60.0)
    r.raise_for_status()
    return r.json()["content"]


def main() -> None:
    golden = load_golden()
    tally = {"APP": 0, "IDEAL": 0, "TIE (order-dependent)": 0}
    for g in golden:
        app_answer = get_candidate(g.question)
        winner = pairwise_consistent(
            g.question, ("APP", app_answer), ("IDEAL", g.ideal_answer)
        )
        tally[winner] = tally.get(winner, 0) + 1
        print(f"{g.id}: winner = {winner}")
    print("\nTotals:", tally)


if __name__ == "__main__":
    main()
