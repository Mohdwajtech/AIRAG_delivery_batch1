"""src/eval/ragas_metrics.py — four RAG metrics, live RAGAS or offline proxy (W11).

Metrics: faithfulness, answer_relevance, context_precision, context_recall.
Offline proxy uses token-overlap (deterministic, free, CI-safe).
Live mode uses real RAGAS with LLM judge calls.

A row is: {"id", "question", "answer", "contexts": [str,...], "ground_truth"}
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class MetricResult:
    """One metric's outcome across all rows."""
    name: str
    mode: str                       # "live" or "offline-proxy"
    aggregate: float
    per_row: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name}={self.aggregate:.3f} ({self.mode}, n={len(self.per_row)})"


def offline_mode() -> bool:
    """Offline if explicitly forced, or if ragas / API key unavailable."""
    if os.getenv("EVAL_OFFLINE", "").strip() in {"1", "true", "TRUE", "yes"}:
        return True
    if not os.getenv("OPENAI_API_KEY"):
        return True
    try:
        import ragas  # noqa: F401
    except Exception:
        return True
    return False


# ── Token-overlap helpers ─────────────────────────────────────

_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "as", "at", "by", "it",
    "this", "that", "these", "those", "from", "your", "you", "may", "must",
    "can", "do", "does", "how", "what", "who", "i", "my", "me", "we", "our",
    "per", "up", "any", "all", "via", "not", "no", "they", "their",
}


def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if len(t) > 2 and t not in _STOP}


def _overlap_fraction(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ── Offline proxies ───────────────────────────────────────────

def _proxy(name: str, rows: list[dict], fn) -> MetricResult:
    per_row = {r["id"]: round(fn(r), 4) for r in rows}
    agg = round(sum(per_row.values()) / len(per_row), 4) if per_row else 0.0
    return MetricResult(name=name, mode="offline-proxy", aggregate=agg, per_row=per_row)


def _faithfulness_proxy(r: dict) -> float:
    ans = _tokens(r.get("answer", ""))
    ctx = _tokens(" ".join(r.get("contexts", [])))
    return _overlap_fraction(ans, ctx)


def _answer_relevance_proxy(r: dict) -> float:
    q = _tokens(r.get("question", ""))
    ans = _tokens(r.get("answer", ""))
    if not q:
        return 0.0
    coverage = _overlap_fraction(q, ans)
    touches = 1.0 if (q & ans) else 0.0
    return round(0.5 * coverage + 0.5 * touches, 4)


def _context_precision_proxy(r: dict) -> float:
    gt = _tokens(r.get("ground_truth", ""))
    contexts = r.get("contexts", []) or []
    if not contexts:
        return 0.0
    relevant = sum(1 for c in contexts if _tokens(c) & gt)
    return relevant / len(contexts)


def _context_recall_proxy(r: dict) -> float:
    gt = _tokens(r.get("ground_truth", ""))
    ctx = _tokens(" ".join(r.get("contexts", [])))
    return _overlap_fraction(gt, ctx)


# ── Live RAGAS ────────────────────────────────────────────────

def _ragas_score(name: str, rows: list[dict], metric, column: str) -> MetricResult:
    """Run one RAGAS metric. Targets ragas ~0.1.x."""
    from datasets import Dataset
    from ragas import evaluate

    ds = Dataset.from_dict({
        "question":     [r["question"] for r in rows],
        "answer":       [r["answer"] for r in rows],
        "contexts":     [list(r.get("contexts", [])) for r in rows],
        "ground_truth": [r.get("ground_truth", "") for r in rows],
    })
    result = evaluate(ds, metrics=[metric])
    df = result.to_pandas()
    per_row = {rows[i]["id"]: round(float(df[column].iloc[i]), 4) for i in range(len(rows))}
    agg = round(float(df[column].mean()), 4)
    return MetricResult(name=name, mode="live", aggregate=agg, per_row=per_row)


# ── Public API ────────────────────────────────────────────────

def score_faithfulness(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("faithfulness", rows, _faithfulness_proxy)
    from ragas.metrics import faithfulness
    return _ragas_score("faithfulness", rows, faithfulness, "faithfulness")


def score_answer_relevance(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("answer_relevance", rows, _answer_relevance_proxy)
    from ragas.metrics import answer_relevancy
    return _ragas_score("answer_relevance", rows, answer_relevancy, "answer_relevancy")


def score_context_precision(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("context_precision", rows, _context_precision_proxy)
    from ragas.metrics import context_precision
    return _ragas_score("context_precision", rows, context_precision, "context_precision")


def score_context_recall(rows: list[dict]) -> MetricResult:
    if offline_mode():
        return _proxy("context_recall", rows, _context_recall_proxy)
    from ragas.metrics import context_recall
    return _ragas_score("context_recall", rows, context_recall, "context_recall")


ALL_METRICS = (score_faithfulness, score_answer_relevance,
               score_context_precision, score_context_recall)


def score_all(rows: list[dict]) -> dict[str, MetricResult]:
    results = [fn(rows) for fn in ALL_METRICS]
    return {r.name: r for r in results}
