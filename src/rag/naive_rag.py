"""Naive RAG pipeline (Week 6) — built from scratch, no frameworks.

Functions:
  chunk_text   : sliding window over a document -> list[str]
  load_corpus  : read data/corpus/*.md|*.txt -> list of chunk dicts
  embed_batch  : text -> vector (text-embedding-3-small); batched
  build_index  : load_corpus -> attach a vector to each chunk
  save_index / load_index : JSON persistence (data/embeddings.json)
  cosine       : cosine similarity between two vectors
  retrieve     : embed question -> score vs all chunks -> return top-k

fake = flow (deterministic hashing embedding, runs offline) / real = quality.
Honours Settings.use_fake, and USE_FAKE=1 as an env override.
"""
from __future__ import annotations
import glob
import json
import os
from pathlib import Path

import numpy as np

from src.pipeline.settings import Settings

_USE_FAKE = Settings().use_fake or os.getenv("USE_FAKE", "").lower() in ("1", "true", "yes")

EMBED_MODEL = "text-embedding-3-small"
CORPUS_DIR  = "data/corpus"
INDEX_PATH  = "data/embeddings.json"
DEFAULT_K   = 3


# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Sliding window over characters. Naive on purpose (cuts mid-sentence)."""
    if len(text) <= size:
        return [text]
    chunks, step = [], size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def load_corpus(corpus_dir: str | Path = CORPUS_DIR) -> list[dict]:
    """Read every .md/.txt file, chunk it, return flat list of chunk dicts."""
    out: list[dict] = []
    paths = sorted(glob.glob(str(Path(corpus_dir) / "*.md")) +
                   glob.glob(str(Path(corpus_dir) / "*.txt")))
    for path in paths:
        source_id = Path(path).stem                      # 'leave_policy'
        text = Path(path).read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_text(text)):
            out.append({"chunk_id": f"{source_id}#{idx}", "source_id": source_id, "text": chunk})
    return out


# ── embedding ─────────────────────────────────────────────────────────────────
def _fake_vec(text: str, dim: int = 256) -> list[float]:
    """Deterministic hashing bag-of-words 'embedding' for offline flow.
    Gives lexical similarity (shared words -> higher cosine); NOT semantic.
    Real quality needs the real model."""
    import hashlib
    v = np.zeros(dim)
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(v)
    return (v / n).tolist() if n else v.tolist()


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """One vector per input string. Batched (100 per call) for the real API."""
    if _USE_FAKE:
        return [_fake_vec(t) for t in texts]
    from openai import OpenAI
    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        resp = client.embeddings.create(model=model, input=texts[i:i + 100])
        out.extend(item.embedding for item in resp.data)
    return out


# ── index build / persistence ────────────────────────────────────────────────
def build_index(corpus_dir: str | Path = CORPUS_DIR) -> list[dict]:
    """Load + chunk the corpus, embed every chunk, attach the vector."""
    chunks = load_corpus(corpus_dir)
    vectors = embed_batch([c["text"] for c in chunks])
    for c, v in zip(chunks, vectors):
        c["vector"] = v
    return chunks


def save_index(index: list[dict], path: str | Path = INDEX_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f)


def load_index(path: str | Path = INDEX_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── retrieval ─────────────────────────────────────────────────────────────────
def cosine(a, b) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))


def retrieve(question: str, index: list[dict], k: int = DEFAULT_K) -> list[dict]:
    """Embed the question, score against every chunk, return the top-k rows
    (each with a 'score')."""
    q_vec = np.array(embed_batch([question])[0])
    mat = np.array([row["vector"] for row in index])
    scores = mat @ q_vec                                  # cosine (vectors are normalised)
    top_idx = np.argsort(scores)[::-1][:k]
    return [{**{kk: vv for kk, vv in index[i].items() if kk != "vector"},
             "score": float(scores[i])} for i in top_idx]
