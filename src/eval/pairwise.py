"""Pairwise comparison with position-flip (Week 5).

"Which of these two answers is better?" is more reliable than absolute scoring.
But judges carry position bias, so we ALWAYS flip position and aggregate:
pairwise_consistent() only declares a winner that wins in both orders.

fake = flow / real = quality (gpt-4o).
"""
from __future__ import annotations
import json
import os

from src.pipeline.settings import Settings

_USE_FAKE = Settings().use_fake or os.getenv("USE_FAKE", "").lower() in ("1", "true", "yes")

JUDGE_MODEL = "gpt-4o"

_PICK_TOOL = {
    "type": "function",
    "function": {
        "name": "pick",
        "parameters": {
            "type": "object",
            "properties": {"winner": {"type": "string", "enum": ["first", "second"]}},
            "required": ["winner"],
        },
    },
}


def _judge_pick(question: str, first: str, second: str) -> str:
    """Return 'first' or 'second' — which the judge prefers for this question."""
    if _USE_FAKE:
        # deterministic stand-in: prefer the one that is non-empty and, on a tie,
        # the first (this is where a real judge's position bias would show).
        sf, ss = len(first.strip()), len(second.strip())
        if sf == 0 and ss > 0:
            return "second"
        if ss == 0 and sf > 0:
            return "first"
        return "first"
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "Which answer better answers the question? Use the tool."},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nFIRST:\n{first}\n\nSECOND:\n{second}"},
        ],
        tools=[_PICK_TOOL],
        tool_choice={"type": "function", "function": {"name": "pick"}},
    )
    return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)["winner"]


def pairwise(question: str, first: tuple[str, str], second: tuple[str, str]) -> str:
    """first/second are (label, text). The judge sees `first` then `second`.
    Returns the winning label."""
    pick = _judge_pick(question, first[1], second[1])
    return first[0] if pick == "first" else second[0]


def pairwise_consistent(question: str, x: tuple[str, str], y: tuple[str, str]) -> str:
    """Run BOTH orders. Only a winner that wins both ways is trustworthy;
    otherwise it's an order-dependent tie (position bias)."""
    w1 = pairwise(question, x, y)
    w2 = pairwise(question, y, x)
    return w1 if w1 == w2 else "TIE (order-dependent)"
