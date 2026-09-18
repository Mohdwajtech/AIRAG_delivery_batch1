"""RAG package.

Week 6 — naive_rag.py: flat JSON index, sliding-window chunks, cosine retrieval.
Week 7 — qdrant_store.py: Qdrant vector DB, embedding cache, ask_rag() interface.

Usage (W7+):
  from src.rag.qdrant_store import get_qdrant_client, ask_rag
  client = get_qdrant_client()
  result = ask_rag("What is the leave policy?", client)
"""
