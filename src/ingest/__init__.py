"""Document ingestion pipeline — Week 8.

Converts raw documents (PDF, HTML, DOCX, Markdown) into clean,
metadata-rich, PII-scrubbed chunks ready for embedding and storage.

Pipeline: parse → chunk → scrub PII → attach metadata → embed → upsert

Modules:
  pipeline : end-to-end ingestion with 5 composable functions

Usage:
  from src.ingest.pipeline import parse_document, chunk_document, ingest_corpus
"""
