"""Advanced retrieval (Week 9) — hybrid search + cross-encoder reranking.

Three retrieval methods, switchable via retrieval_method parameter:
  dense         : pure cosine via Qdrant (W8 baseline)
  hybrid        : BM25 + dense + Reciprocal Rank Fusion
  hybrid_rerank : hybrid + cross-encoder re-scoring (W9 default)

Components:
  BM25           : in-memory keyword index via rank_bm25 (BM25Okapi)
  RRF            : Reciprocal Rank Fusion (k=60, Cormack et al. 2009)
  Cross-encoder  : ms-marco-MiniLM-L-6-v2 (22.7M params, ~8ms/pair)

Design:
  - Pure functions, not classes (testable, composable)
  - Lazy-loaded models via singletons (_bm25_index, _reranker)
  - Metadata filtering passes through to both BM25 and Qdrant
  - All methods accessible via ask_rag(retrieval_method=...)
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# W10: Tombstone-aware live filter
# ═══════════════════════════════════════════════════════════════

def _live_filter(extra_filter=None):
    """Build a Qdrant filter that excludes tombstoned chunks (deleted_at IS EMPTY).

    Uses IsEmptyCondition (NOT IsNullCondition) — the W10 silent-bug lesson.
    IsNullCondition checks for explicit null values. IsEmptyCondition checks for
    the ABSENCE of the field entirely, which is what un-tombstoned chunks have
    (they never had a deleted_at field set). Using IsNull would miss chunks that
    were never tombstoned (because null ≠ absent).

    Pass extra_filter to combine with additional conditions (e.g., source_id match).
    """
    from qdrant_client.models import Filter, IsEmptyCondition, PayloadField

    tombstone_filter = Filter(
        must=[IsEmptyCondition(is_empty=PayloadField(key="deleted_at"))]
    )

    if extra_filter is None:
        return tombstone_filter

    # Combine: tombstone filter + caller's filter
    combined_must = list(tombstone_filter.must or [])
    combined_must.extend(extra_filter.must or [])
    return Filter(must=combined_must)

import re
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# BM25 — Keyword retrieval
# ═══════════════════════════════════════════════════════════════

def simple_tokenize(text: str) -> list[str]:
    """Lowercase + keep hyphens/slashes for identifiers.

    'Error AC-1042' → ['error', 'ac-1042']    (keeps the code intact)
    'POST /v2/dashboards' → ['post', '/v2/dashboards']

    CRITICAL: use the SAME tokenizer for index build AND query time.
    Mismatch = all-zero BM25 scores.
    """
    return re.findall(r'[a-z0-9][a-z0-9\-/_]*', text.lower())


_bm25_index = None
_bm25_corpus = None


def build_bm25_index(chunks: list[dict]) -> None:
    """Build in-memory BM25 index from chunk texts.

    Call once after loading/ingesting the corpus. The index lives in
    memory for the duration of the session.

    Each chunk dict must have 'text' key. Optionally 'title' — if present,
    title tokens are appended (makes title keywords searchable).
    """
    global _bm25_index, _bm25_corpus
    from rank_bm25 import BM25Okapi

    _bm25_corpus = chunks
    tokenized = [
        simple_tokenize(c.get("text", "") + " " + c.get("title", ""))
        for c in chunks
    ]
    _bm25_index = BM25Okapi(tokenized)


def bm25_search(query: str, k: int = 10, doc_filter: Optional[dict] = None) -> list[dict]:
    """Score all chunks by BM25, return top-k with scores.

    Args:
        query: the user's question
        k: number of results to return
        doc_filter: optional {field: value} dict — only return chunks
                    where chunk[field] == value (pre-filter)

    Returns:
        list of dicts with 'id', 'score', 'doc' keys
    """
    if _bm25_index is None:
        raise RuntimeError("Call build_bm25_index(chunks) first.")

    tokens = simple_tokenize(query)
    scores = _bm25_index.get_scores(tokens)

    # Pair scores with corpus, optionally filter
    candidates = []
    for score, chunk in zip(scores, _bm25_corpus):
        if doc_filter:
            if not all(chunk.get(field) == value for field, value in doc_filter.items()):
                continue
        candidates.append((score, chunk))

    ranked = sorted(candidates, key=lambda x: x[0], reverse=True)[:k]

    return [
        {
            "id": chunk.get("id", chunk.get("chunk_id", "")),
            "score": float(score),
            "text": chunk.get("text", ""),
            "doc": chunk,
        }
        for score, chunk in ranked
        if score > 0
    ]


# ═══════════════════════════════════════════════════════════════
# RRF — Reciprocal Rank Fusion
# ═══════════════════════════════════════════════════════════════

def rrf_fuse(
    ranked_lists: list[list[dict]],
    k: int = 60,
    top_n: int = 10,
    weights: list[float] | None = None,
) -> list[dict]:
    """Fuse multiple ranked result lists via Weighted Reciprocal Rank Fusion.

    RRF formula: score(doc) = Σ  weight_i × 1/(k + rank_in_list_i)
      - k=60 is the Cormack et al. (2009) default
      - Uses RANK position, not raw scores → no scale mismatch
      - weights controls how much each list contributes to the fused score
        Default: equal weight (1.0 each) — standard unweighted RRF
        Example: weights=[0.7, 0.3] → dense=70%, BM25=30%

    Each ranked_list contains dicts with 'id' key. Additional fields
    are preserved from whichever list the doc first appeared in.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists length ({len(ranked_lists)})"
        )

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for weight, ranked in zip(weights, ranked_lists):
        for rank, hit in enumerate(ranked, start=1):
            doc_id = str(hit.get("id", ""))
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = hit

    fused = sorted(scores.items(), key=lambda p: p[1], reverse=True)[:top_n]

    return [
        {**docs[doc_id], "rrf_score": score}
        for doc_id, score in fused
    ]


# ═══════════════════════════════════════════════════════════════
# Dense retrieval (W8 baseline — wraps qdrant_store)
# ═══════════════════════════════════════════════════════════════

def dense_retrieve(
    query: str,
    client,
    collection: str = "capstone_chunks_v2",
    k: int = 3,
    model: str = "text-embedding-3-small",
    query_filter=None,
    include_deleted: bool = False,
) -> list[dict]:
    """Pure dense retrieval via Qdrant. This is the W8 baseline.

    By default, tombstoned chunks (deleted_at set) are excluded via _live_filter().
    Pass include_deleted=True for audit/forensic queries.
    """
    from src.rag.qdrant_store import embed_one

    q_vec = embed_one(query, model=model)

    # W10: apply tombstone filter unless explicitly including deleted chunks
    if not include_deleted:
        query_filter = _live_filter(extra_filter=query_filter)

    hits = client.query_points(
        collection_name=collection,
        query=q_vec,
        query_filter=query_filter,
        limit=k,
    ).points

    return [
        {
            "id": h.payload.get("id", h.payload.get("chunk_id", h.id)),
            "score": h.score,
            "text": h.payload.get("text", ""),
            "doc": h.payload,
        }
        for h in hits
    ]


# ═══════════════════════════════════════════════════════════════
# Hybrid retrieval (W9 Day 1)
# ═══════════════════════════════════════════════════════════════

def hybrid_retrieve(
    query: str,
    client,
    collection: str = "capstone_chunks_v2",
    k_per: int = 10,
    k_final: int = 3,
    model: str = "text-embedding-3-small",
    query_filter=None,
    doc_filter: Optional[dict] = None,
    dense_weight: float = 0.7,
) -> list[dict]:
    """BM25 + Dense + RRF fusion with configurable weighting.

    Args:
        query: user's question
        client: Qdrant client
        collection: Qdrant collection name
        k_per: how many results each retriever returns (wider = better fusion)
        k_final: how many fused results to return
        model: embedding model for dense retrieval
        query_filter: Qdrant filter object (for dense retrieval)
        doc_filter: dict filter (for BM25 pre-filtering)
        dense_weight: how much weight dense retrieval gets (0.0–1.0).
            BM25 weight is (1 - dense_weight). Default 0.7 means dense=70%,
            BM25=30%. At 0.5 both are equal (standard unweighted RRF).
            Tuned against golden set in W9: 0.7 gave best F1.

    Returns:
        top k_final results after weighted RRF fusion
    """
    bm25_weight = round(1.0 - dense_weight, 4)

    bm25_hits = bm25_search(query, k=k_per, doc_filter=doc_filter)
    dense_hits = dense_retrieve(
        query, client, collection=collection,
        k=k_per, model=model, query_filter=query_filter,
    )

    return rrf_fuse(
        [bm25_hits, dense_hits],
        k=60,
        top_n=k_final,
        weights=[bm25_weight, dense_weight],
    )


# ═══════════════════════════════════════════════════════════════
# Cross-encoder reranker (W9 Day 2)
# ═══════════════════════════════════════════════════════════════

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def _load_reranker():
    """Lazy singleton — download ~80MB on first use, reuse after."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Re-score candidates with cross-encoder, return top_k.

    The cross-encoder sees (query, document) together — full cross-attention.
    More accurate than bi-encoder cosine but ~8ms per pair (CPU).

    Candidates must have 'text' or 'doc.text' key.
    Adds 'rerank_score' field to each returned dict.
    """
    reranker = _load_reranker()

    # Extract text — handle both flat and nested schemas
    texts = []
    for c in candidates:
        t = c.get("text", "")
        if not t and "doc" in c:
            t = c["doc"].get("text", "")
        texts.append(t)

    pairs = [(query, t) for t in texts]
    scores = reranker.predict(pairs)

    scored = [
        {**c, "rerank_score": float(s)}
        for c, s in zip(candidates, scores)
    ]
    scored.sort(key=lambda h: h["rerank_score"], reverse=True)

    return scored[:top_k]


# ═══════════════════════════════════════════════════════════════
# ask_rag — unified dispatcher
# ═══════════════════════════════════════════════════════════════

def ask_rag(
    question: str,
    client,
    collection: str = "capstone_chunks_v2",
    retrieval_method: str = "hybrid_rerank",
    k_retrieve: int = 20,
    k_final: int = 3,
    model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4o-mini",
    query_filter=None,
    doc_filter: Optional[dict] = None,
) -> dict:
    """End-to-end: retrieve → (optional rerank) → generate answer.

    retrieval_method:
      'dense'         — W8 baseline (cosine via Qdrant)
      'hybrid'        — W9 Day 1 (BM25 + dense + RRF)
      'hybrid_rerank' — W9 Day 2 (hybrid + cross-encoder) ← DEFAULT

    Returns: {answer, citations, chunks_used, retrieval_method}
    """
    # ── Retrieve ──
    if retrieval_method == "dense":
        chunks = dense_retrieve(
            question, client, collection=collection,
            k=k_final, model=model, query_filter=query_filter,
        )

    elif retrieval_method == "hybrid":
        chunks = hybrid_retrieve(
            question, client, collection=collection,
            k_per=k_retrieve, k_final=k_final,
            model=model, query_filter=query_filter, doc_filter=doc_filter,
        )

    elif retrieval_method == "hybrid_rerank":
        # Stage 1: wide hybrid retrieval
        candidates = hybrid_retrieve(
            question, client, collection=collection,
            k_per=k_retrieve, k_final=k_retrieve,
            model=model, query_filter=query_filter, doc_filter=doc_filter,
        )
        # Stage 2: precise cross-encoder reranking
        chunks = rerank(question, candidates, top_k=k_final)

    else:
        raise ValueError(
            f"Unknown retrieval_method: {retrieval_method!r}. "
            f"Choose from: 'dense', 'hybrid', 'hybrid_rerank'"
        )

    if not chunks:
        return {
            "answer": "No relevant documents found.",
            "citations": [],
            "chunks_used": 0,
            "retrieval_method": retrieval_method,
        }

    # ── Generate ──
    from openai import OpenAI

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("doc", {}).get("source", chunk.get("id", f"chunk_{i}"))
        text = chunk.get("text", chunk.get("doc", {}).get("text", ""))
        context_parts.append(f"[{i}] (source: {source})\n{text}")

    context = "\n\n".join(context_parts)

    llm = OpenAI()
    system = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "Cite sources using [1], [2], etc. If the context doesn't contain the answer, say so."
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    resp = llm.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    answer = resp.choices[0].message.content

    return {
        "answer": answer,
        "citations": [
            {
                "source": c.get("doc", {}).get("source", c.get("id", "")),
                "score": c.get("rerank_score", c.get("rrf_score", c.get("score", 0))),
            }
            for c in chunks
        ],
        "chunks_used": len(chunks),
        "retrieval_method": retrieval_method,
    }
