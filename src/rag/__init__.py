"""Naive RAG package (Week 6).

Built from scratch — no frameworks. Two passes share one store:
  index-time (once):   load_corpus -> chunk_text -> embed_batch -> build_index -> save_index
  query-time (each Q): retrieve (embed question -> cosine vs all -> top-k)
"""
