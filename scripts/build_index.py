"""Week 6 - build the RAG index once and cache it to data/embeddings.json.

Reads data/corpus/*.md, chunks + embeds every chunk, writes the index. Re-run
only when the corpus changes.

  real:     OPENAI_API_KEY set  ->  text-embedding-3-small
  offline:  USE_FAKE=1 python scripts/build_index.py  (hashing embedding, flow only)

Usage:  python scripts/build_index.py
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.rag.naive_rag import build_index, save_index, INDEX_PATH, CORPUS_DIR


def main() -> None:
    index = build_index(CORPUS_DIR)
    save_index(index, INDEX_PATH)
    sources = sorted({c["source_id"] for c in index})
    print(f"Indexed {len(index)} chunks from {len(sources)} docs -> {INDEX_PATH}")
    print("sources:", ", ".join(sources))


if __name__ == "__main__":
    main()
