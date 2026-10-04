"""src/debug/scenarios.py — the W12 debugging drill: six broken RAG queries.

Each scenario breaks exactly one pipeline stage. The learner reads the trace,
classifies into one of four buckets, and the grader confirms or hints.

Buckets: retrieval | generation | prompt | infra
"""
from __future__ import annotations

from dataclasses import dataclass

BUCKETS = ("retrieval", "generation", "prompt", "infra")


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    question_id: str
    symptom: str
    bucket: str
    root_cause: str
    fix: str
    trace_tell: str


SCENARIOS = [
    Scenario(
        id="S1", title="The multi-hop answer is missing half",
        question_id="g015",
        symptom="The answer covers remote-work days but says nothing about the in-person meeting requirement.",
        bucket="retrieval",
        root_cause="Recall miss: the second required chunk was never retrieved.",
        fix="Raise top-K or add query expansion for multi-hop questions.",
        trace_tell="In retrieve, only the '-a' chunk is present; the '-b' chunk is absent.",
    ),
    Scenario(
        id="S2", title="The answer is about the office kitchen",
        question_id="g004",
        symptom="The answer is fluent but talks about kitchens/parking instead of the BYOD policy.",
        bucket="retrieval",
        root_cause="Precision failure: distractors scored highest and pushed the correct chunk out.",
        fix="Retune hybrid weighting / reranker so relevant chunks outrank distractors.",
        trace_tell="In retrieve/rerank, top chunk_ids are 'kb-*' distractors with high scores.",
    ),
    Scenario(
        id="S3", title="A confident fact that isn't in the docs",
        question_id="g001",
        symptom="The answer is mostly right but adds a 150%-of-salary cash-out claim no policy contains.",
        bucket="generation",
        root_cause="Hallucination: retrieval was correct but the model asserted an absent fact.",
        fix="Lower temperature and/or tighten the prompt's grounding constraint.",
        trace_tell="retrieve is healthy (correct chunks, good scores) but the answer has an unsupported claim.",
    ),
    Scenario(
        id="S4", title="An extra step nobody asked for",
        question_id="g007",
        symptom="The security-incident answer tacks on 'post in #general Slack', which is not policy.",
        bucket="prompt",
        root_cause="The prompt template lost its grounding instruction.",
        fix="Restore the grounding instruction in the prompt template.",
        trace_tell="In prompt, has_grounding_instruction is false.",
    ),
    Scenario(
        id="S5", title="A fluent answer to a different question",
        question_id="g010",
        symptom="Asked about WFH, the answer describes standard working hours instead.",
        bucket="infra",
        root_cause="Stale cache: hit returned a cached answer for a different question.",
        fix="Fix the cache key / invalidate stale entries (W10 tombstone + kb_version).",
        trace_tell="cache shows cache_hit=true with cached_for_question=g003; retrieve/generate skipped.",
    ),
    Scenario(
        id="S6", title="No answer at all",
        question_id="g008",
        symptom="The request returns an error instead of an answer, and it was slow.",
        bucket="infra",
        root_cause="Generation model call timed out (30s) with no retry.",
        fix="Add retry with backoff and a timeout budget; check API health.",
        trace_tell="generate stage is marked failed (✗) with error 'timeout after 30000ms'.",
    ),
]

_BY_ID = {s.id: s for s in SCENARIOS}


def get(scenario_id: str) -> Scenario:
    if scenario_id not in _BY_ID:
        raise KeyError(f"Unknown scenario {scenario_id!r}; try one of {list(_BY_ID)}")
    return _BY_ID[scenario_id]


def _hint_stage(bucket: str) -> str:
    return {"retrieval": "retrieve/rerank", "generation": "generate",
            "prompt": "prompt", "infra": "cache/generate"}[bucket]


def grade(scenario_id: str, guess_bucket: str) -> tuple[bool, str]:
    """Check classification. Returns (correct, message)."""
    s = get(scenario_id)
    guess = (guess_bucket or "").strip().lower()
    if guess not in BUCKETS:
        return False, f"'{guess_bucket}' is not a bucket. Pick one of: {', '.join(BUCKETS)}."
    if guess == s.bucket:
        return True, (f"✓ Correct — {s.bucket}.\n"
                      f"  Root cause: {s.root_cause}\n"
                      f"  Fix:        {s.fix}")
    return False, (f"✗ Not {guess}. Re-read the trace — hint: look at the "
                   f"'{_hint_stage(s.bucket)}' stage.")
