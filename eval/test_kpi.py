"""eval/test_kpi.py — threshold assertions for the 8-KPI dashboard (W11).

Runs offline (EVAL_OFFLINE=1 in CI) or live.
Offline thresholds are calibrated lower (proxy is cruder than live RAGAS).

Run: EVAL_OFFLINE=1 python3 -m pytest eval/test_kpi.py -v
"""
from __future__ import annotations

import os
import pytest

# Force offline for CI
os.environ.setdefault("EVAL_OFFLINE", "1")

from src.eval.harness import build_rows, active_kb_version
from src.eval.ragas_metrics import score_all, offline_mode
from src.eval.kpi import compute_kpis, reportable_count


# ── Thresholds (calibrated from baseline runs) ───────────────
# Offline thresholds are lower because the token-overlap proxy is crude.
# Live thresholds come from 3 baseline runs — set below the noise floor.

OFFLINE_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevance": 0.55,       # proxy is crude for relevance
    "context_precision": 0.55,      # capped by distractor in synthetic data
    "context_recall": 0.80,
}

LIVE_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevance": 0.85,
    "context_precision": 0.75,
    "context_recall": 0.80,
}


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rows():
    return build_rows()


@pytest.fixture(scope="module")
def kpi_output(rows):
    return compute_kpis(rows, active_kb_version())


@pytest.fixture(scope="module")
def metrics(kpi_output):
    return kpi_output["metrics"]


# ── Metric threshold tests ───────────────────────────────────

def test_faithfulness(metrics):
    thresholds = OFFLINE_THRESHOLDS if offline_mode() else LIVE_THRESHOLDS
    assert metrics["faithfulness"].aggregate >= thresholds["faithfulness"], (
        f"faithfulness {metrics['faithfulness'].aggregate:.3f} "
        f"< threshold {thresholds['faithfulness']}"
    )


def test_answer_relevance(metrics):
    thresholds = OFFLINE_THRESHOLDS if offline_mode() else LIVE_THRESHOLDS
    assert metrics["answer_relevance"].aggregate >= thresholds["answer_relevance"], (
        f"answer_relevance {metrics['answer_relevance'].aggregate:.3f} "
        f"< threshold {thresholds['answer_relevance']}"
    )


def test_context_precision(metrics):
    thresholds = OFFLINE_THRESHOLDS if offline_mode() else LIVE_THRESHOLDS
    assert metrics["context_precision"].aggregate >= thresholds["context_precision"], (
        f"context_precision {metrics['context_precision'].aggregate:.3f} "
        f"< threshold {thresholds['context_precision']}"
    )


def test_context_recall(metrics):
    thresholds = OFFLINE_THRESHOLDS if offline_mode() else LIVE_THRESHOLDS
    assert metrics["context_recall"].aggregate >= thresholds["context_recall"], (
        f"context_recall {metrics['context_recall'].aggregate:.3f} "
        f"< threshold {thresholds['context_recall']}"
    )


# ── KPI structure tests ──────────────────────────────────────

def test_reportable_count(kpi_output):
    numeric, na = reportable_count(kpi_output["kpis"])
    assert numeric == 6 and na == 2, f"Expected 6+2, got {numeric}+{na}"


def test_task_success_rate_in_range(kpi_output):
    ts = kpi_output["kpis"]["1_task_success_rate"]
    assert 0.0 <= ts <= 1.0, f"Task success rate {ts} out of range"


def test_hallucination_plus_grounded_equals_one(kpi_output):
    kpis = kpi_output["kpis"]
    total = kpis["2_grounded_response_rate"] + kpis["3_hallucination_rate"]
    assert abs(total - 1.0) < 0.001, f"KPI2 + KPI3 = {total}, expected 1.0"


def test_latency_p50_less_than_p95(kpi_output):
    kpis = kpi_output["kpis"]
    assert kpis["7_latency_p50_ms"] <= kpis["7_latency_p95_ms"], (
        f"p50 ({kpis['7_latency_p50_ms']}) > p95 ({kpis['7_latency_p95_ms']})"
    )
