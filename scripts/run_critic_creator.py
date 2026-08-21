"""Week 5 - run the critic/creator loop on a weak answer and save the trace.

Takes the first golden question, gets the app's current answer as the draft,
then loops critic <-> creator until it satisfies the rubric. Writes the
round-by-round trace to docs/critic-creator-trace.md.

Prereqs: API running + OPENAI_API_KEY (or USE_FAKE=1 for an offline dry run).
Usage:   python scripts/run_critic_creator.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import httpx

from src.eval.golden import load_golden
from src.eval.critic_creator import critic_creator

API_URL = "http://localhost:8000/ask_batched"
REPORT  = "docs/critic-creator-trace.md"


def get_candidate(question: str) -> str:
    r = httpx.post(API_URL, json={"question": question}, timeout=60.0)
    r.raise_for_status()
    return r.json()["content"]


def main() -> None:
    g = load_golden()[0]
    draft = get_candidate(g.question)
    result = critic_creator(g.question, draft, g.must_mention)

    lines = [f"# Critic-Creator trace - {g.id}", "",
             f"**Question:** {g.question}", "",
             f"**Initial draft:** {draft}", ""]
    for step in result["trace"]:
        lines += [f"## Round {step['round']}",
                  f"- answer: {step['answer']}",
                  f"- issues: {step['issues'] or 'none'}", ""]
    lines += [f"**Final ({result['rounds']} rounds):** {result['final']}", ""]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {REPORT}  ({result['rounds']} rounds)")
    print("FINAL:", result["final"])


if __name__ == "__main__":
    main()
