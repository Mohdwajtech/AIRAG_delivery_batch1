"""Critic-Creator improvement loop (Week 5).

Creator drafts -> Critic critiques against a rubric -> Creator revises -> repeat
until the critic is satisfied (or max_rounds). Critic = gpt-4o (strong),
Creator = gpt-4o-mini (cheap).

fake = flow (deterministic) / real = quality.
"""
from __future__ import annotations
import json
import os

from src.pipeline.settings import Settings

_USE_FAKE = Settings().use_fake or os.getenv("USE_FAKE", "").lower() in ("1", "true", "yes")

CRITIC_MODEL  = "gpt-4o"
CREATOR_MODEL = "gpt-4o-mini"

_CRITIC_TOOL = {
    "type": "function",
    "function": {
        "name": "critique",
        "parameters": {
            "type": "object",
            "properties": {
                "issues":         {"type": "array", "items": {"type": "string"}},
                "is_good_enough": {"type": "boolean"},
            },
            "required": ["issues", "is_good_enough"],
        },
    },
}


def critic(question: str, answer: str,
           must_mention: list[str] | None = None, max_words: int = 60) -> dict:
    """Return {issues: [...], is_good_enough: bool}."""
    must_mention = must_mention or []
    if _USE_FAKE:
        issues = [f"missing required fact: '{m}'" for m in must_mention
                  if m.lower() not in answer.lower()]
        if len(answer.split()) > max_words:
            issues.append(f"too long ({len(answer.split())} words); tighten to <= {max_words}")
        return {"issues": issues, "is_good_enough": len(issues) == 0}
    from openai import OpenAI
    client = OpenAI()
    rubric = ("You are a strict critic. List concrete, fixable issues with the answer to the "
              f"question: {question!r}. It must state these facts: {must_mention}. "
              f"It must be under {max_words} words. If there are no issues, return an empty "
              "list and is_good_enough=true.")
    resp = client.chat.completions.create(
        model=CRITIC_MODEL, temperature=0,
        messages=[{"role": "system", "content": rubric},
                  {"role": "user", "content": answer}],
        tools=[_CRITIC_TOOL],
        tool_choice={"type": "function", "function": {"name": "critique"}},
    )
    return json.loads(resp.choices[0].message.tool_calls[0].function.arguments)


def creator(question: str, answer: str, critique: dict) -> str:
    """Revise the answer to address the critique's issues."""
    if not critique["issues"]:
        return answer
    if _USE_FAKE:
        issue = critique["issues"][0]                       # fix the top issue per round
        if "missing required fact" in issue:
            fact = issue.split("'")[1] if "'" in issue else ""
            return (answer.rstrip(". ") + f". {fact}.").strip()
        if "too long" in issue:
            return " ".join(answer.split()[:55])
        return answer
    from openai import OpenAI
    client = OpenAI()
    prompt = (f"Question: {question}\n\nCurrent answer:\n{answer}\n\n"
              "Fix these issues, keep it concise:\n- " + "\n- ".join(critique["issues"]))
    resp = client.chat.completions.create(
        model=CREATOR_MODEL, temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def critic_creator(question: str, draft: str,
                   must_mention: list[str] | None = None, max_rounds: int = 5) -> dict:
    """Loop critic <-> creator until good enough or max_rounds. Returns the final
    answer plus the round-by-round trace."""
    answer = draft
    trace = []
    for r in range(1, max_rounds + 1):
        c = critic(question, answer, must_mention)
        trace.append({"round": r, "answer": answer, "issues": c["issues"]})
        if c["is_good_enough"]:
            break
        answer = creator(question, answer, c)
    return {"final": answer, "rounds": len(trace), "trace": trace}
