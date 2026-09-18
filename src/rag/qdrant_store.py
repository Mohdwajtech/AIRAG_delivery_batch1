"""Qdrant vector store wrapper (Week 7).

Replaces the flat JSON index from W6 with a proper vector database.
Adds: embedding cache, Qdrant upsert/query, collection management.

Functions:
  get_qdrant_client : connect to Qdrant (local or cloud)
  ensure_collection : create collection if it doesn't exist
  embed_texts       : embed with cache (don't re-embed identical text)
  upsert_chunks     : push chunks + vectors to Qdrant
  query_similar     : embed query → cosine search → return top-k
  ask_rag           : end-to-end question → answer (retrieve + generate)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
CACHE_DIR = Path("data/embed_cache")
DEFAULT_COLLECTION = "capstone_chunks"
DEFAULT_K = 3


# ── Qdrant client ────────────────────────────────────────────

def get_qdrant_client():
    """Connect to Qdrant. Uses env vars QDRANT_URL + QDRANT_API_KEY,
    or falls back to localhost:6333."""
    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if url and api_key:
        return QdrantClient(url=url, api_key=api_key)
    elif url:
        return QdrantClient(url=url)
    else:
        return QdrantClient(host="localhost", port=6333)


def ensure_collection(
    client,
    name: str = DEFAULT_COLLECTION,
    dims: int | None = None,
) -> None:
    """Create collection if it doesn't exist. Skip if it does."""
    from qdrant_client.models import Distance, VectorParams

    if dims is None:
        dims = EMBED_DIMS.get(EMBED_MODEL, 1536)

    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
        print(f"  ✓ Created collection '{name}' ({dims} dims, cosine)")
    else:
        print(f"  ⊘ Collection '{name}' already exists")


# ── Embedding with cache ─────────────────────────────────────

def _cache_key(text: str, model: str) -> str:
    """Deterministic key from (model + text)."""
    raw = f"{model}::{text}"
    return hashlib.sha256(raw.encode()).hexdigest()


def embed_one(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text with filesystem cache."""
    key = _cache_key(text, model)
    cache_path = CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    client = OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    vector = resp.data[0].embedding

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(vector, f)

    return vector


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed multiple texts. Uses cache per-text, batches cache misses."""
    vectors = [None] * len(texts)
    misses = []  # (index, text) pairs that need API calls

    # Check cache first
    for i, text in enumerate(texts):
        key = _cache_key(text, model)
        cache_path = CACHE_DIR / f"{key}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                vectors[i] = json.load(f)
        else:
            misses.append((i, text))

    # Batch-embed cache misses
    if misses:
        client = OpenAI()
        miss_texts = [t for _, t in misses]
        all_vecs = []
        for batch_start in range(0, len(miss_texts), 100):
            batch = miss_texts[batch_start : batch_start + 100]
            resp = client.embeddings.create(model=model, input=batch)
            all_vecs.extend(item.embedding for item in resp.data)

        # Store in cache and in result
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for (orig_idx, text), vec in zip(misses, all_vecs):
            vectors[orig_idx] = vec
            key = _cache_key(text, model)
            with open(CACHE_DIR / f"{key}.json", "w") as f:
                json.dump(vec, f)

    return vectors


# ── Qdrant operations ────────────────────────────────────────

def upsert_chunks(
    client,
    chunks: list[dict],
    vectors: list[list[float]],
    collection: str = DEFAULT_COLLECTION,
) -> int:
    """Push chunks + vectors to Qdrant. Returns point count."""
    from qdrant_client.models import PointStruct

    points = [
        PointStruct(
            id=idx,
            vector=vec,
            payload={k: v for k, v in chunk.items() if k != "vector"},
        )
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    client.upsert(collection_name=collection, points=points)
    return len(points)


def query_similar(
    client,
    query: str,
    collection: str = DEFAULT_COLLECTION,
    k: int = DEFAULT_K,
    model: str = EMBED_MODEL,
    query_filter=None,
) -> list[dict]:
    """Embed query → Qdrant search → return top-k with scores."""
    q_vec = embed_one(query, model=model)

    hits = client.query_points(
        collection_name=collection,
        query=q_vec,
        query_filter=query_filter,
        limit=k,
    ).points

    return [
        {
            "id": h.id,
            "score": h.score,
            **h.payload,
        }
        for h in hits
    ]


# ── End-to-end ask_rag ────────────────────────────────────────

def ask_rag(
    question: str,
    client=None,
    collection: str = DEFAULT_COLLECTION,
    k: int = DEFAULT_K,
    model: str = EMBED_MODEL,
    llm_model: str = "gpt-4o-mini",
    query_filter=None,
) -> dict:
    """Retrieve context from Qdrant, generate answer with LLM.

    Returns: {answer, citations, chunks_used}
    """
    if client is None:
        client = get_qdrant_client()

    # Retrieve
    chunks = query_similar(
        client, question, collection=collection,
        k=k, model=model, query_filter=query_filter,
    )

    if not chunks:
        return {"answer": "No relevant documents found.", "citations": [], "chunks_used": []}

    # Build context
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_id", chunk.get("source", f"chunk_{i}"))
        context_parts.append(f"[{i}] (source: {source})\n{chunk.get('text', '')}")

    context = "\n\n".join(context_parts)

    # Generate
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
            {"source": c.get("source_id", c.get("source", "")), "score": c.get("score", 0)}
            for c in chunks
        ],
        "chunks_used": len(chunks),
    }
