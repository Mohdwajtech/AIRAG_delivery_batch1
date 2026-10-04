"""src/eval/harness.py — build eval rows, synthetic (offline) or live (W11).

Row shape: {"id", "question", "answer", "contexts": [str,...], "ground_truth",
            "cache_hit", "latency_ms", "cost_usd", "notes"}
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.golden import GoldenEntry, load_golden
from src.eval.ragas_metrics import offline_mode

GOLDEN_PATH = Path("data/golden_set.jsonl")
OFFLINE_KB_VERSION = "v2.w10"

_DISTRACTOR = (
    "The office kitchen is restocked every Monday morning and the building "
    "reception desk is open from eight until six."
)


def _split_two(text: str) -> tuple[str, str]:
    words = text.split()
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]), " ".join(words[mid:])


def _synth_row(entry: GoldenEntry, index: int) -> dict:
    """Build one synthetic row. Most healthy; g019/g015/g005 are degraded."""
    gt = entry.ideal_answer if isinstance(entry.ideal_answer, str) else json.dumps(entry.ideal_answer)
    c1, c2 = _split_two(gt)
    contexts = [c1, c2, _DISTRACTOR]
    answer = gt

    if entry.id == "g019":
        answer = gt + (" The company also matches up to five percent of salary "
                       "into a managed crypto retirement fund.")
    elif entry.id == "g015":
        contexts = [c1, _DISTRACTOR]
    elif entry.id == "g005":
        contexts = [c1[:max(1, len(c1) // 2)], _DISTRACTOR]

    cache_hit = index < 13
    return {
        "id":           entry.id,
        "question":     entry.question,
        "answer":       answer,
        "contexts":     contexts,
        "ground_truth": gt,
        "cache_hit":    cache_hit,
        "latency_ms":   184 if cache_hit else 785,
        "cost_usd":     0.000041 if cache_hit else 0.000412,
        "notes":        entry.notes,
    }


def _live_rows(golden: list[GoldenEntry]) -> list[dict]:  # pragma: no cover
    """Build rows from the real capstone pipeline."""
    try:
        from src.rag.qdrant_store import ask_rag as qdrant_ask, get_qdrant_client
        from src.rag.retrieval import hybrid_retrieve, build_bm25_index
    except Exception as exc:
        raise RuntimeError(
            "Live eval needs capstone code on path + OPENAI_API_KEY + QDRANT_URL. "
            f"Run offline with EVAL_OFFLINE=1. Import error: {exc}"
        ) from exc

    client = get_qdrant_client()
    rows: list[dict] = []
    for e in golden:
        res = qdrant_ask(e.question, client=client, k=3)
        retrieved = hybrid_retrieve(e.question, client, k_final=3)
        rows.append({
            "id":           e.id,
            "question":     e.question,
            "answer":       res.get("answer", ""),
            "contexts":     [c.get("text", "") for c in retrieved],
            "ground_truth": e.ideal_answer if isinstance(e.ideal_answer, str) else json.dumps(e.ideal_answer),
            "cache_hit":    res.get("cache_hit", False),
            "latency_ms":   res.get("latency_ms", 0),
            "cost_usd":     res.get("cost_usd", 0.0),
            "notes":        e.notes,
        })
    return rows


def active_kb_version() -> str:
    if offline_mode():
        return OFFLINE_KB_VERSION
    try:
        from src.rag.cache import SemanticCache
        return OFFLINE_KB_VERSION  # replace with pipeline's KB_VERSION when wired
    except Exception:
        return "unknown"


def build_rows(golden: list[GoldenEntry] | None = None) -> list[dict]:
    """The eval rows under test — synthetic (offline) or live."""
    golden = golden or load_golden(GOLDEN_PATH)
    if offline_mode():
        return [_synth_row(e, i) for i, e in enumerate(golden)]
    return _live_rows(golden)
