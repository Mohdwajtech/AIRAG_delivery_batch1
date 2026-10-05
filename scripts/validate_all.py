#!/usr/bin/env python3
"""scripts/validate_all.py — Validate ALL W7-W12 concepts run correctly.

Runs fully OFFLINE — no API key, no Qdrant, no cost.
Tests every concept taught from W7 through W12.

Usage:
    cd /path/to/project
    python3 scripts/validate_all.py

Expected: ALL checks pass with ✓. Any ✗ means a gap.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["EVAL_OFFLINE"] = "1"
os.environ["USE_FAKE"] = "1"

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {label}")
    else:
        FAIL += 1
        print(f"  ✗ {label}  — {detail}")


def section(title: str):
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


# ══════════════════════════════════════════════════════════════
# W7 — Qdrant Store + Embeddings
# ══════════════════════════════════════════════════════════════
section("W7: Qdrant Store + Embedding Cache")

try:
    from src.rag.qdrant_store import (
        _cache_key, EMBED_MODEL, EMBED_DIMS, DEFAULT_COLLECTION,
        get_qdrant_client, ensure_collection, embed_one, embed_texts,
        upsert_chunks, query_similar, ask_rag,
    )
    check("qdrant_store.py imports cleanly", True)
    check("EMBED_MODEL is text-embedding-3-small", EMBED_MODEL == "text-embedding-3-small")
    check("EMBED_DIMS has small + large", len(EMBED_DIMS) >= 2)
    check("DEFAULT_COLLECTION defined", DEFAULT_COLLECTION == "capstone_chunks")

    # Cache key is deterministic
    k1 = _cache_key("hello world", "model-a")
    k2 = _cache_key("hello world", "model-a")
    k3 = _cache_key("different", "model-a")
    check("Embedding cache key is deterministic", k1 == k2)
    check("Different text → different cache key", k1 != k3)
except Exception as e:
    check("qdrant_store.py imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W8 — Ingestion Pipeline + PII
# ══════════════════════════════════════════════════════════════
section("W8: Ingestion Pipeline + PII Scrubbing")

try:
    from src.ingest.pipeline import (
        ParsedSection, ParsedDocument, Chunk,
        detect_format, parse_document, chunk_document,
        scrub_chunks, finalize_metadata, PII_PATTERNS,
        _parse_markdown, _infer_doc_type,
    )
    check("ingest/pipeline.py imports cleanly", True)

    # Format detection
    check("detect_format(.md) = md", detect_format(Path("test.md")) == "md")
    check("detect_format(.pdf) = pdf", detect_format(Path("test.pdf")) == "pdf")
    check("detect_format(.html) = html", detect_format(Path("test.html")) == "html")

    # PII patterns
    check("PII patterns include EMAIL", "EMAIL" in PII_PATTERNS)
    check("PII patterns include SSN", "SSN" in PII_PATTERNS)
    check("PII patterns include CREDIT_CARD", "CREDIT_CARD" in PII_PATTERNS)

    # PII scrubbing works
    test_chunk = Chunk(text="Contact john@example.com or call 555-123-4567", metadata={})
    scrubbed = scrub_chunks([test_chunk])
    check("PII scrubbing redacts email", "[EMAIL]" in scrubbed[0].text)
    check("PII scrubbing sets has_pii flag", scrubbed[0].metadata.get("has_pii") == True)

    # Parse a corpus file
    corpus_files = list(Path("data/corpus").glob("*.md"))
    if corpus_files:
        doc = parse_document(corpus_files[0])
        check(f"parse_document({corpus_files[0].name}) works", len(doc.sections) > 0)
        chunks = chunk_document(doc)
        check(f"chunk_document produces chunks", len(chunks) > 0)
        chunks = finalize_metadata(chunks)
        check("finalize_metadata adds chunk_id", "chunk_id" in chunks[0].metadata)
    else:
        check("Corpus files exist", False, "No .md files in data/corpus/")

except Exception as e:
    check("ingest/pipeline.py imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W9 — Hybrid Retrieval + Reranking
# ══════════════════════════════════════════════════════════════
section("W9: Hybrid Retrieval + Reranking")

try:
    from src.rag.retrieval import (
        simple_tokenize, build_bm25_index, bm25_search,
        rrf_fuse, dense_retrieve, hybrid_retrieve,
        rerank, ask_rag as retrieval_ask_rag,
        RERANKER_MODEL,
    )
    check("retrieval.py imports cleanly", True)
    check("Reranker model defined", "ms-marco" in RERANKER_MODEL)

    # Tokenizer
    tokens = simple_tokenize("Error AC-1042 happened today")
    check("simple_tokenize keeps identifiers", "ac-1042" in tokens)

    # BM25 index + search
    test_chunks = [
        {"text": "The leave policy allows 20 days per year", "id": "c1"},
        {"text": "BYOD devices must have VPN installed", "id": "c2"},
        {"text": "The office kitchen is restocked on Monday", "id": "c3"},
    ]
    build_bm25_index(test_chunks)
    results = bm25_search("leave policy days", k=2)
    check("BM25 search returns results", len(results) > 0)
    check("BM25 top result is about leave", "leave" in results[0]["text"].lower() if results else False)

    # RRF fusion
    list_a = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]
    list_b = [{"id": "2", "score": 0.95}, {"id": "3", "score": 0.7}]
    fused = rrf_fuse([list_a, list_b], k=60, top_n=3)
    check("RRF fusion produces results", len(fused) > 0)
    check("RRF: doc in both lists ranks higher", fused[0]["id"] == "2",
          f"Expected '2' first, got '{fused[0]['id']}'")

except Exception as e:
    check("retrieval.py imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W10 — Semantic Cache
# ══════════════════════════════════════════════════════════════
section("W10: Semantic Cache + KB Lifecycle")

try:
    from src.rag.cache import SemanticCache
    check("cache.py imports cleanly", True)

    import tempfile, numpy as np
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        test_db = f.name

    cache = SemanticCache(test_db)
    check("SemanticCache initializes", True)

    # Store + lookup
    vec1 = list(np.random.randn(10).astype(float))
    cache.store(vec1, "What is the leave policy?", "20 days per year", kb_version="v1")
    check("cache.store() works", True)

    # Exact match lookup
    hit = cache.lookup(vec1, kb_version="v1", threshold=0.95)
    check("cache.lookup() finds exact match", hit is not None and hit["cache_hit"] == True)
    check("cache.lookup() returns correct answer", hit["answer"] == "20 days per year" if hit else False)

    # Wrong version → miss
    miss = cache.lookup(vec1, kb_version="v2", threshold=0.95)
    check("cache.lookup() misses on wrong kb_version", miss is None)

    # Different vector → miss
    vec2 = list(np.random.randn(10).astype(float))
    miss2 = cache.lookup(vec2, kb_version="v1", threshold=0.95)
    check("cache.lookup() misses on different vector", miss2 is None)

    # Clear
    cleared = cache.clear(kb_version="v1")
    check("cache.clear() removes entries", cleared >= 1)
    miss3 = cache.lookup(vec1, kb_version="v1", threshold=0.95)
    check("After clear, lookup returns None", miss3 is None)

    # Stats
    stats = cache.stats()
    check("cache.stats() returns dict", isinstance(stats, dict))

    os.unlink(test_db)

except Exception as e:
    check("cache.py imports", False, str(e))


# ── W10 Tombstones + KB Lifecycle ─────────────────────────────

section("W10: Tombstones + KB Lifecycle")

try:
    from src.ingest.pipeline import (
        tombstone_source, ingest_or_update_source, KB_VERSION,
    )
    check("tombstone_source() defined", callable(tombstone_source))
    check("ingest_or_update_source() defined", callable(ingest_or_update_source))
    check(f"KB_VERSION = '{KB_VERSION}'", KB_VERSION == "v2.w10")

    from src.rag.retrieval import _live_filter
    check("_live_filter() defined in retrieval.py", callable(_live_filter))

    # Test _live_filter produces a Filter object
    f = _live_filter()
    check("_live_filter() returns a Filter", f is not None and hasattr(f, "must"))
    check("Filter uses IsEmptyCondition (not IsNull)", "IsEmpty" in str(type(f.must[0])))

except ImportError as e:
    check("tombstone imports", False, f"ImportError: {e} — install qdrant-client for full check")
except Exception as e:
    check("tombstone logic", False, str(e))


# ══════════════════════════════════════════════════════════════
# W11 — RAGAS Metrics (Offline Proxy)
# ══════════════════════════════════════════════════════════════
section("W11: RAGAS Metrics (Offline Proxy)")

try:
    from src.eval.ragas_metrics import (
        offline_mode, MetricResult, score_all,
        score_faithfulness, score_answer_relevance,
        score_context_precision, score_context_recall,
        _tokens, _overlap_fraction,
    )
    check("ragas_metrics.py imports cleanly", True)
    check("offline_mode() returns True (EVAL_OFFLINE=1)", offline_mode() == True)

    # Token extraction
    t = _tokens("Employees get 20 days of paid leave per year")
    check("_tokens extracts content words", "employees" in t and "leave" in t)
    check("_tokens removes stop words", "of" not in t and "the" not in t)

    # Overlap fraction
    a = {"leave", "policy", "days"}
    b = {"leave", "days", "kitchen"}
    frac = _overlap_fraction(a, b)
    check("_overlap_fraction directional", abs(frac - 2/3) < 0.01)

    # Build test rows
    test_rows = [
        {"id": "t1", "question": "leave policy",
         "answer": "Employees get 20 days of paid leave",
         "contexts": ["Employees get 20 days of paid leave per year"],
         "ground_truth": "Employees get 20 days of paid leave per year"},
        {"id": "t2", "question": "BYOD policy",
         "answer": "Bring your device with VPN",
         "contexts": ["BYOD devices must have VPN installed"],
         "ground_truth": "BYOD devices must have VPN installed"},
    ]

    results = score_all(test_rows)
    check("score_all returns 4 metrics", len(results) == 4)
    for name in ["faithfulness", "answer_relevance", "context_precision", "context_recall"]:
        r = results[name]
        check(f"  {name}: mode=offline-proxy, agg={r.aggregate:.3f}",
              r.mode == "offline-proxy" and 0 <= r.aggregate <= 1)

except Exception as e:
    check("ragas_metrics.py imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W11 — Eval Harness + KPI Dashboard
# ══════════════════════════════════════════════════════════════
section("W11: Eval Harness + 8-KPI Dashboard")

try:
    from src.eval.golden import GoldenEntry, load_golden
    check("golden.py imports cleanly", True)

    golden = load_golden("data/golden_set.jsonl")
    check(f"Golden set loads: {len(golden)} entries", len(golden) >= 10)

    from src.eval.harness import build_rows, active_kb_version
    rows = build_rows(golden)
    check(f"build_rows produces {len(rows)} rows", len(rows) == len(golden))
    check("Each row has required fields",
          all(k in rows[0] for k in ["id", "question", "answer", "contexts", "ground_truth"]))
    check("Rows include cache_hit field", "cache_hit" in rows[0])
    check("Rows include latency_ms field", "latency_ms" in rows[0])
    check("Rows include cost_usd field", "cost_usd" in rows[0])
    check(f"active_kb_version = '{active_kb_version()}'", active_kb_version() != "")

    from src.eval.kpi import compute_kpis, reportable_count, TOOL_SUCCESS_NA, ESCALATION_NA
    out = compute_kpis(rows, active_kb_version())
    kpis = out["kpis"]
    extras = out["extras"]

    check("8 KPI keys present", len(kpis) >= 8)
    check("KPI #1 (task success) in [0,1]", 0 <= kpis["1_task_success_rate"] <= 1)
    check("KPI #2 (grounded) in [0,1]", 0 <= kpis["2_grounded_response_rate"] <= 1)
    check("KPI #3 = 1 - KPI #2", abs(kpis["3_hallucination_rate"] - (1 - kpis["2_grounded_response_rate"])) < 0.001)
    check("KPI #5 is N/A", kpis["5_tool_success_rate"] == TOOL_SUCCESS_NA)
    check("KPI #8 is N/A", kpis["8_escalation_rate"] == ESCALATION_NA)
    check("Latency p50 ≤ p95", kpis["7_latency_p50_ms"] <= kpis["7_latency_p95_ms"])

    numeric, na = reportable_count(kpis)
    check(f"Reportable: {numeric} numeric + {na} N/A = 6+2", numeric == 6 and na == 2)

    check(f"eval_mode = '{extras['eval_mode']}'", extras["eval_mode"] == "offline-proxy")
    check(f"cache_hit_rate = {extras['cache_hit_rate']}", extras["cache_hit_rate"] > 0)

    # Print the dashboard
    print("\n  --- 8-KPI Dashboard ---")
    for k, v in kpis.items():
        print(f"    {k:30s} {v}")
    print(f"    {'eval_mode':30s} {extras['eval_mode']}")
    print(f"    {'kb_version':30s} {extras['kb_version']}")

except Exception as e:
    check("harness/kpi imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W11 — Golden Set Validation
# ══════════════════════════════════════════════════════════════
section("W11: Golden Set Integrity")

try:
    ids = [e.id for e in golden]
    check("All IDs unique", len(ids) == len(set(ids)))
    check("All IDs non-empty", all(len(i) > 0 for i in ids))
    check("All questions non-empty", all(len(e.question) > 0 for e in golden))
    check("All ideal_answers non-empty", all(len(str(e.ideal_answer)) > 0 for e in golden))
except Exception as e:
    check("Golden set integrity", False, str(e))


# ══════════════════════════════════════════════════════════════
# W12 — Trace Module
# ══════════════════════════════════════════════════════════════
section("W12: Trace Module")

try:
    from src.rag.trace import Trace, StageRecord
    check("trace.py imports cleanly", True)

    tr = Trace(question="Test query")
    tr.add("embed", "3 tokens", {"tokens": ["test", "query"]}, latency_ms=42)
    tr.add("cache", "miss", {"cache_hit": False}, latency_ms=5)
    tr.add("retrieve", "top-3", {"chunk_ids": ["c1", "c2", "c3"]}, latency_ms=88)
    tr.add("rerank", "reordered", {"top1_score": 0.95}, latency_ms=61)
    tr.add("prompt", "built", {"has_grounding_instruction": True}, latency_ms=3)
    tr.add("generate", "answer", {"tokens": 25, "cost_usd": 0.00021}, latency_ms=430)
    tr.answer = "Test answer"

    check("Trace has 6 stages", len(tr.stages) == 6)
    check(f"Total latency = {tr.total_latency_ms}ms", tr.total_latency_ms == 629)
    check("tr.get('cache') works", tr.get("cache") is not None)
    check("tr.get('cache').detail has cache_hit", "cache_hit" in tr.get("cache").detail)

    rendered = tr.render()
    check("render() produces readable output", "TRACE" in rendered and "ANSWER" in rendered)
    check("render() shows all stages", "EMBED" in rendered and "GENERATE" in rendered)

    j = tr.to_json()
    parsed = json.loads(j)
    check("to_json() produces valid JSON", "stages" in parsed and len(parsed["stages"]) == 6)

    # Test failed stage
    tr_fail = Trace(question="Fail test")
    tr_fail.add("generate", "ERROR", {"error": "timeout"}, ok=False, latency_ms=30000)
    rendered_fail = tr_fail.render()
    check("Failed stage shows ✗ flag", "✗" in rendered_fail)

except Exception as e:
    check("trace.py imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W12 — Debug Scenarios + Grader
# ══════════════════════════════════════════════════════════════
section("W12: Debug Scenarios + Grader")

try:
    from src.debug.scenarios import SCENARIOS, BUCKETS, get, grade
    check("scenarios.py imports cleanly", True)
    check("6 scenarios defined", len(SCENARIOS) == 6)
    check("4 buckets defined", len(BUCKETS) == 4)

    # Each scenario has all required fields
    for s in SCENARIOS:
        check(f"  {s.id}: bucket='{s.bucket}', question={s.question_id}",
              s.bucket in BUCKETS and len(s.trace_tell) > 0)

    # Grader works — correct guess
    ok, msg = grade("S1", "retrieval")
    check("grade(S1, retrieval) = correct", ok == True)
    check("Correct grade reveals root cause", "Root cause" in msg)

    # Grader works — wrong guess
    ok, msg = grade("S1", "generation")
    check("grade(S1, generation) = wrong", ok == False)
    check("Wrong grade gives hint", "hint" in msg.lower())

    # Grader works — invalid bucket
    ok, msg = grade("S1", "data_quality")
    check("grade(S1, invalid) = rejected", ok == False and "not a bucket" in msg.lower())

    # All six answers verified
    answer_key = {s.id: s.bucket for s in SCENARIOS}
    expected = {"S1": "retrieval", "S2": "retrieval", "S3": "generation",
                "S4": "prompt", "S5": "infra", "S6": "infra"}
    check("Answer key matches expected", answer_key == expected)

except Exception as e:
    check("scenarios.py imports", False, str(e))


# ══════════════════════════════════════════════════════════════
# W5 — Judge (existing, verify still works)
# ══════════════════════════════════════════════════════════════
section("W5: LLM Judge (fake mode)")

try:
    from src.eval.judge import judge, RUBRIC, JUDGE_TOOL
    check("judge.py imports cleanly", True)
    check("Rubric defined", len(RUBRIC) > 100)
    check("Judge tool schema defined", JUDGE_TOOL["function"]["name"] == "submit_scores")

    result = judge("What is leave?", "20 days", "20 days of leave per year")
    check("judge() returns scores", "accuracy" in result and "groundedness" in result)
    check("Scores in range [1,4]", all(1 <= result[k] <= 4 for k in ["accuracy", "groundedness", "format"]))

except Exception as e:
    check("judge.py", False, str(e))


# ══════════════════════════════════════════════════════════════
# File structure checks
# ══════════════════════════════════════════════════════════════
section("Project Structure")

expected_files = [
    "src/rag/qdrant_store.py", "src/rag/retrieval.py", "src/rag/cache.py", "src/rag/trace.py",
    "src/ingest/pipeline.py",
    "src/eval/golden.py", "src/eval/judge.py", "src/eval/ragas_metrics.py",
    "src/eval/harness.py", "src/eval/kpi.py",
    "src/debug/scenarios.py",
    "api/main.py",
    "data/golden_set.jsonl", "data/corpus/leave_policy.md",
    "scripts/debug_drill.py", "scripts/m2_check.py", "scripts/smoke_test.sh",
    "eval/test_kpi.py",
    "requirements.txt",
    "docs/kpi/wk12-snapshot.md", "docs/runbooks/document_update.md",
]
for f in expected_files:
    check(f"  {f} exists", Path(f).exists())


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
section("SUMMARY")
total = PASS + FAIL
print(f"\n  ✓ {PASS} passed")
if FAIL:
    print(f"  ✗ {FAIL} FAILED")
print(f"  Total: {total} checks")
print()

if FAIL == 0:
    print("  🎉 ALL CHECKS PASSED — project covers W5-W12 concepts!")
else:
    print(f"  ⚠  {FAIL} checks failed — see ✗ items above.")

sys.exit(0 if FAIL == 0 else 1)
