"""wk09_pipeline.py — shared helpers for W9 notebooks.

Day 1 builds BM25 + RRF cell-by-cell. Day 2 imports them from here so we
don't spend time rebuilding.

The corpus (15 short docs about a fictional "Acme Analytics Platform") and
6 test queries are bundled under sample_docs/. Rerun generate_acme_corpus.py
to regenerate, or substitute your own JSONL with the same schema.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ─── Corpus paths ─────────────────────────────────────────────────────

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"
ACME_CORPUS_PATH = SAMPLE_DOCS_DIR / "acme_docs.jsonl"
ACME_QUERIES_PATH = SAMPLE_DOCS_DIR / "acme_test_queries.jsonl"


def load_acme_corpus() -> list[dict]:
    """Return the 15-doc Acme corpus. Each doc: {id, title, text, category}."""
    assert ACME_CORPUS_PATH.exists(), (
        f"Missing {ACME_CORPUS_PATH.name}. "
        f"Run: python demos/generate_acme_corpus.py"
    )
    return [json.loads(line) for line in ACME_CORPUS_PATH.read_text().splitlines() if line]


def load_test_queries() -> list[dict]:
    """Return the 6 test queries with expected-winner annotations."""
    assert ACME_QUERIES_PATH.exists(), (
        f"Missing {ACME_QUERIES_PATH.name}. "
        f"Run: python demos/generate_acme_corpus.py"
    )
    return [json.loads(line) for line in ACME_QUERIES_PATH.read_text().splitlines() if line]


# ─── BM25 (same code Day 1 builds cell-by-cell) ───────────────────────

def simple_tokenize(text: str) -> list[str]:
    """Lowercase + word-and-alphanumeric-token split.

    Deliberately simple so learners see what tokenization actually does.
    Notice we KEEP hyphens inside alphanumeric tokens (so 'AC-1042' stays
    as one token) — critical for BM25 to catch error codes.
    """
    text = text.lower()
    # Match tokens: sequences of word chars possibly containing hyphens/slashes
    # This keeps 'ac-1042' and '/v2/dashboards' as single tokens
    return re.findall(r'[a-z0-9][a-z0-9\-/_]*', text)


def build_bm25_index(corpus: list[dict]):
    """Build a rank-bm25 BM25Okapi index over the corpus text field."""
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [simple_tokenize(doc["text"] + " " + doc["title"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def bm25_search(index, corpus: list[dict], query: str, k: int = 3) -> list[dict]:
    """Query BM25 index; return top-K docs with scores."""
    tokens = simple_tokenize(query)
    scores = index.get_scores(tokens)
    ranked = sorted(zip(scores, corpus), key=lambda pair: pair[0], reverse=True)
    return [
        {"id": doc["id"], "title": doc["title"], "score": float(score), "doc": doc}
        for score, doc in ranked[:k]
    ]


# ─── RRF (Reciprocal Rank Fusion) ─────────────────────────────────────

def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60, top_n: int = 10) -> list[dict]:
    """Fuse multiple ranked result lists via Reciprocal Rank Fusion.

    RRF formula:  score(doc) = sum_over_lists( 1 / (k + rank_in_list) )

    k=60 is the Cormack et al. (2009) default — high enough to make top-1
    only slightly more valuable than top-2 (prevents any single retriever
    from dominating), low enough that rank still matters.

    Each ranked_list contains dicts with 'id' key. Returns fused ranking.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = hit
    
    fused = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:top_n]
    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in fused
    ]


# ─── Cross-encoder reranker ───────────────────────────────────────────

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None
def load_reranker():
    """Lazy-load the cross-encoder. Downloads ~80MB on first use."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Score each (query, candidate.text) pair, return top_k by score.
    
    Candidates must have a 'doc' key with 'text' inside.
    Adds a 'rerank_score' field to each returned dict.
    """
    reranker = load_reranker()
    pairs = [(query, hit["doc"]["text"]) for hit in candidates]
    scores = reranker.predict(pairs)
    scored = [{**hit, "rerank_score": float(s)} for hit, s in zip(candidates, scores)]
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)
    return scored[:top_k]
