"""RAG package.

Week 6 — naive_rag.py: flat JSON index, sliding-window chunks, cosine retrieval.
Week 7 — qdrant_store.py: Qdrant vector DB, embedding cache, ask_rag() interface.
Week 9 — retrieval.py: hybrid search (BM25 + dense + RRF), cross-encoder reranking.

Usage (W9+):
  from src.rag.retrieval import ask_rag, build_bm25_index
  from src.rag.qdrant_store import get_qdrant_client

  client = get_qdrant_client()
  build_bm25_index(chunks)  # one-time setup for BM25

  # Three retrieval methods:
  result = ask_rag("question", client, retrieval_method='dense')          # W8 baseline
  result = ask_rag("question", client, retrieval_method='hybrid')         # W9 Day 1
  result = ask_rag("question", client, retrieval_method='hybrid_rerank')  # W9 Day 2 (default)
"""
