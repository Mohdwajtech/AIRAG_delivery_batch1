"""api/main.py — Production RAG API (upgraded W7→W12).

Wraps the W9 hybrid retrieval + W10 semantic cache behind a stable HTTP contract.

Endpoints:
  POST /ask     → answer a question (cache-aware, hybrid+rerank retrieval)
  GET  /health  → system health (Qdrant, cache, model status)

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

KB_VERSION = os.getenv("KB_VERSION", "v2.w10")
CACHE_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.95"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
COLLECTION = os.getenv("QDRANT_COLLECTION", "capstone_chunks_v2")


# ── Lazy singletons ──────────────────────────────────────────

_qdrant_client = None
_bm25_built = False
_cache = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from src.rag.qdrant_store import get_qdrant_client
            _qdrant_client = get_qdrant_client()
            log.info("Qdrant client connected")
        except Exception as e:
            log.warning("Qdrant unavailable: %s", e)
    return _qdrant_client


def _ensure_bm25():
    global _bm25_built
    if not _bm25_built:
        try:
            from src.rag.retrieval import build_bm25_index
            from src.rag.qdrant_store import get_qdrant_client
            client = _get_qdrant()
            if client:
                # Load chunks from Qdrant for BM25 indexing
                hits = client.scroll(collection_name=COLLECTION, limit=10000)[0]
                chunks = [{"text": h.payload.get("text", ""), "id": h.id, **h.payload} for h in hits]
                build_bm25_index(chunks)
                _bm25_built = True
                log.info("BM25 index built: %d chunks", len(chunks))
        except Exception as e:
            log.warning("BM25 index build failed: %s", e)


def _get_cache():
    global _cache
    if _cache is None:
        try:
            from src.rag.cache import SemanticCache
            _cache = SemanticCache("results.db")
            log.info("Semantic cache initialized")
        except Exception as e:
            log.warning("Cache unavailable: %s", e)
    return _cache


# ── API models ───────────────────────────────────────────────

class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str
    cache_hit: bool = False
    retrieval_method: str = "hybrid_rerank"
    chunks_used: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    kb_version: str = ""
    citations: list[dict] = []


app = FastAPI(
    title="Capstone RAG API",
    description="W9 hybrid retrieval + W10 semantic cache + W12 production API.",
    version="2.0.0",
)


# ── /ask endpoint ────────────────────────────────────────────

@app.post("/ask", response_model=Answer)
async def ask(q: Question):
    """Answer a question with cache-aware hybrid retrieval."""
    t0 = time.time()
    log.info("ask  question=%r", q.question[:80])

    # Ensure BM25 is built
    _ensure_bm25()

    # Stage 1: Check semantic cache
    cache = _get_cache()
    if cache:
        from src.rag.qdrant_store import embed_one
        try:
            q_vec = embed_one(q.question, model=EMBED_MODEL)
            hit = cache.lookup(q_vec, kb_version=KB_VERSION, threshold=CACHE_THRESHOLD)
            if hit:
                latency = int((time.time() - t0) * 1000)
                log.info("cache HIT  similarity=%.3f  latency=%dms", hit["similarity"], latency)
                return Answer(
                    answer=hit["answer"],
                    cache_hit=True,
                    retrieval_method="cache",
                    latency_ms=latency,
                    kb_version=KB_VERSION,
                )
        except Exception as e:
            log.warning("Cache lookup failed: %s", e)

    # Stage 2: Full pipeline — retrieve + rerank + generate
    client = _get_qdrant()
    if client is None:
        latency = int((time.time() - t0) * 1000)
        return Answer(
            answer="Service unavailable: Qdrant not connected.",
            latency_ms=latency,
            kb_version=KB_VERSION,
        )

    try:
        from src.rag.retrieval import ask_rag
        result = ask_rag(
            q.question,
            client,
            collection=COLLECTION,
            retrieval_method="hybrid_rerank",
            k_final=3,
            model=EMBED_MODEL,
            llm_model=LLM_MODEL,
        )
    except Exception as e:
        log.error("Pipeline error: %s", e)
        latency = int((time.time() - t0) * 1000)
        return Answer(
            answer=f"Error: {e}",
            latency_ms=latency,
            kb_version=KB_VERSION,
        )

    latency = int((time.time() - t0) * 1000)

    # Stage 3: Store in cache for next time
    if cache:
        try:
            cache.store(q_vec, q.question, result["answer"], kb_version=KB_VERSION)
        except Exception as e:
            log.warning("Cache store failed: %s", e)

    return Answer(
        answer=result.get("answer", ""),
        cache_hit=False,
        retrieval_method=result.get("retrieval_method", "hybrid_rerank"),
        chunks_used=result.get("chunks_used", 0),
        latency_ms=latency,
        cost_usd=result.get("cost_usd", 0.0),
        kb_version=KB_VERSION,
        citations=result.get("citations", []),
    )


# ── /health endpoint ─────────────────────────────────────────

@app.get("/health")
async def health():
    """System health check — reports all dependency statuses."""
    qdrant_status = "disconnected"
    qdrant_count = 0
    try:
        client = _get_qdrant()
        if client:
            info = client.get_collection(COLLECTION)
            qdrant_count = info.points_count
            qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {e}"

    cache_status = "disabled"
    cache_entries = 0
    try:
        cache = _get_cache()
        if cache:
            stats = cache.stats()
            cache_entries = stats.get("total_entries", 0)
            cache_status = "ready"
    except Exception:
        cache_status = "error"

    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "qdrant_collection": COLLECTION,
        "qdrant_points": qdrant_count,
        "cache": cache_status,
        "cache_entries": cache_entries,
        "model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "kb_version": KB_VERSION,
        "bm25_ready": _bm25_built,
    }
