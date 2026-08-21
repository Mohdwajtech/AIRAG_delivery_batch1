"""LLM-as-judge (Week 5).

Score a candidate answer against an ideal answer on a rubric — accuracy,
groundedness, format — each 1-4. The judge is the STRONG model (gpt-4o); the
capstone answers with gpt-4o-mini, but judging is where we don't cut corners.

fake = flow (deterministic stand-in, runs offline) / real = quality (gpt-4o).
Honours Settings.use_fake, and USE_FAKE=1 as an env override for a free run.
"""
from __future__ import annotations
import json
import os

from src.pipeline.settings import Settings

_USE_FAKE = Settings().use_fake or os.getenv("USE_FAKE", "").lower() in ("1", "true", "yes")

JUDGE_MODEL = "gpt-4o"   # strong judge — do NOT fall back to the capstone's gpt-4o-mini

RUBRIC = (
    "You are a strict, fair evaluator of answers from a question-answering assistant.\n"
    "Score the CANDIDATE answer against the IDEAL answer on three dimensions, each 1-4:\n"
    "  - accuracy    : are the facts correct and complete versus the ideal?\n"
    "  - groundedness: is it supported by the ideal/source, nothing invented or contradictory?\n"
    "  - format      : is it clear, appropriately concise, and well-structured?\n"
    "Scale: 1 = Poor, 2 = OK, 3 = Good, 4 = Excellent.\n"
    "Be strict on accuracy: an answer that omits a key fact or contradicts the ideal "
    "cannot score above 2.\n"
    "Return your scores and a one-paragraph reasoning that names specific facts."
)

JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_scores",
        "description": "Submit rubric scores and reasoning for the candidate answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "accuracy":     {"type": "integer", "minimum": 1, "maximum": 4},
                "groundedness": {"type": "integer", "minimum": 1, "maximum": 4},
                "format":       {"type": "integer", "minimum": 1, "maximum": 4},
                "reasoning":    {"type": "string"},
            },
            "required": ["accuracy", "groundedness", "format", "reasoning"],
        },
    },
}


def build_messages(question: str, ideal: str, candidate: str) -> list[dict]:
    user = (f"QUESTION:\n{question}\n\n"
            f"IDEAL ANSWER:\n{ideal}\n\n"
            f"CANDIDATE ANSWER:\n{candidate}")
    return [{"role": "system", "content": RUBRIC},
            {"role": "user",   "content": user}]


def judge(question: str, ideal: str, candidate: str,
          must_mention: list[str] | None = None) -> dict:
    """Return {accuracy, groundedness, format, reasoning}."""
    if _USE_FAKE:
        return _fake_judge(question, ideal, candidate, must_mention or [])
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,                       # a judge should be repeatable
        messages=build_messages(question, ideal, candidate),
        tools=[JUDGE_TOOL],
        tool_choice={"type": "function", "function": {"name": "submit_scores"}},
    )
    return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)


def _fake_judge(question: str, ideal: str, candidate: str,
                must_mention: list[str]) -> dict:
    """Deterministic stand-in so the harness runs offline. Substring-based (dumb
    on purpose) — the real gpt-4o judge reads for meaning."""
    c = candidate.lower()
    hits = [m for m in must_mention if m.lower() in c]
    contradicts = any(p in c for p in ["no limit", "unlimited", "whenever you like", "any day"])
    if must_mention:
        accuracy = 1 if contradicts else (4 if len(hits) == len(must_mention) else 2)
    else:
        accuracy = 3
    grounded = 1 if contradicts else 4
    fmt = 4 if len(candidate.split()) <= 60 else 3
    reasoning = (f"[fake judge] mentions {len(hits)}/{len(must_mention)} required facts {hits}. "
                 + ("Contradicts the ideal. " if contradicts else "")
                 + ("Missing a required fact. " if hits and len(hits) < len(must_mention) else "")).strip()
    return {"accuracy": accuracy, "groundedness": grounded, "format": fmt, "reasoning": reasoning}
