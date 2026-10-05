"""Ingestion pipeline (Week 8) — parse, chunk, scrub, enrich, embed, upsert.

Five composable functions:
  parse_document    : path → ParsedDocument (dispatches by file extension)
  chunk_document    : ParsedDocument → list[Chunk] (structure-aware + recursive fallback)
  scrub_chunks      : list[Chunk] → list[Chunk] (regex + Presidio NER)
  finalize_metadata : list[Chunk] → list[Chunk] (timestamps, chunk IDs)
  ingest_corpus     : directory → IngestionResult (orchestrator)

Parsers (lazy imports — only loaded when the format is needed):
  _parse_pdf  : PyMuPDF — default for PDFs, with scanned-page detection
  _parse_html : BeautifulSoup — strips nav/script/style/aside/footer
  _parse_docx : python-docx — uses para.style.name for heading detection
  _parse_md   : plain text — splits on # headings

PII scrubbing (three layers):
  Layer 1: regex patterns (EMAIL, PHONE, EMP_ID, SSN, CREDIT_CARD)
  Layer 2: Presidio NER (PERSON, LOCATION — graceful fallback if not installed)
  Layer 3: manual audit (protocol in docs/pii-audit.md — not automated)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════

@dataclass
class ParsedSection:
    """One heading-bounded section of a parsed document."""
    text: str
    heading: str = ""
    heading_level: int = 0
    page: Optional[int] = None


@dataclass
class ParsedDocument:
    """Output of parse_document()."""
    source: str          # filename, e.g. 'leave_policy.md'
    doc_title: str       # from first heading or filename
    doc_type: str        # hr / policy / technical / security / general
    sections: list[ParsedSection]
    raw_text: str        # full text fallback


@dataclass
class Chunk:
    """A chunk ready for embedding + Qdrant upsert."""
    text: str
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Format detection + dispatch
# ═══════════════════════════════════════════════════════════════

def detect_format(path: Path) -> str:
    """Normalise file extension to a format key."""
    suffix = path.suffix.lower().lstrip(".")
    return {"htm": "html", "markdown": "md"}.get(suffix, suffix)


def parse_document(path: Path) -> ParsedDocument:
    """Dispatch to format-specific parser based on file extension."""
    fmt = detect_format(path)
    if fmt == "pdf":
        return _parse_pdf(path)
    elif fmt == "html":
        return _parse_html(path)
    elif fmt == "docx":
        return _parse_docx(path)
    elif fmt in {"md", "txt"}:
        return _parse_markdown(path)
    else:
        raise ValueError(f"Unsupported format: {fmt} ({path.name})")


# ═══════════════════════════════════════════════════════════════
# Format-specific parsers
# ═══════════════════════════════════════════════════════════════

def _parse_pdf(path: Path) -> ParsedDocument:
    """Extract text from PDF using PyMuPDF. Skips scanned pages (<50 chars)."""
    import pymupdf

    doc = pymupdf.open(str(path))
    meta = doc.metadata
    doc_title = meta.get("title", "") or path.stem

    sections = []
    full_parts = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if len(text.strip()) < 50:
            # Likely a scanned/image page — skip
            continue
        clean = text.strip()
        sections.append(ParsedSection(
            text=clean,
            heading=f"Page {page_num + 1}",
            heading_level=1,
            page=page_num + 1,
        ))
        full_parts.append(clean)

    doc.close()

    return ParsedDocument(
        source=path.name,
        doc_title=doc_title,
        doc_type=_infer_doc_type(path),
        sections=sections,
        raw_text="\n\n".join(full_parts),
    )


def _parse_html(path: Path) -> ParsedDocument:
    """Extract content from HTML, stripping browser chrome first."""
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Strip non-content elements BEFORE extracting text
    for tag_name in ["nav", "footer", "script", "style", "aside", "header", "noscript", "iframe"]:
        for el in soup.find_all(tag_name):
            el.decompose()

    # Extract heading-based sections
    sections = []
    current_heading = path.stem
    current_level = 0
    current_parts = []

    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        if element.name.startswith("h") and element.name[1:].isdigit():
            if current_parts:
                sections.append(ParsedSection(
                    heading=current_heading,
                    heading_level=current_level,
                    text="\n".join(current_parts),
                ))
                current_parts = []
            current_heading = element.get_text(strip=True)
            current_level = int(element.name[1])
        else:
            t = element.get_text(strip=True)
            if t:
                current_parts.append(t)

    if current_parts:
        sections.append(ParsedSection(
            heading=current_heading,
            heading_level=current_level,
            text="\n".join(current_parts),
        ))

    doc_title = soup.title.string if soup.title else path.stem

    return ParsedDocument(
        source=path.name,
        doc_title=doc_title,
        doc_type=_infer_doc_type(path),
        sections=sections,
        raw_text=soup.get_text(separator="\n"),
    )


def _parse_docx(path: Path) -> ParsedDocument:
    """Extract paragraphs from DOCX using heading styles for structure."""
    from docx import Document

    doc = Document(str(path))
    sections = []
    current_heading = path.stem
    current_level = 0
    current_parts = []

    for para in doc.paragraphs:
        if not para.text.strip():
            continue

        style = para.style.name
        if style.startswith("Heading") or style == "Title":
            if current_parts:
                sections.append(ParsedSection(
                    heading=current_heading,
                    heading_level=current_level,
                    text="\n".join(current_parts),
                ))
                current_parts = []
            current_heading = para.text.strip()
            current_level = 0 if style == "Title" else int(style.split()[-1])
        else:
            current_parts.append(para.text.strip())

    if current_parts:
        sections.append(ParsedSection(
            heading=current_heading,
            heading_level=current_level,
            text="\n".join(current_parts),
        ))

    doc_title = sections[0].heading if sections else path.stem

    return ParsedDocument(
        source=path.name,
        doc_title=doc_title,
        doc_type=_infer_doc_type(path),
        sections=sections,
        raw_text="\n\n".join(p.text for p in doc.paragraphs if p.text.strip()),
    )


def _parse_markdown(path: Path) -> ParsedDocument:
    """Parse Markdown by splitting on # headings."""
    text = path.read_text(encoding="utf-8")
    sections = []
    current_heading = path.stem
    current_level = 0
    current_parts = []

    for line in text.split("\n"):
        if line.startswith("#"):
            if current_parts:
                sections.append(ParsedSection(
                    heading=current_heading,
                    heading_level=current_level,
                    text="\n".join(current_parts),
                ))
                current_parts = []
            hashes = len(line) - len(line.lstrip("#"))
            current_heading = line.lstrip("#").strip()
            current_level = hashes
        else:
            if line.strip():
                current_parts.append(line)

    if current_parts:
        sections.append(ParsedSection(
            heading=current_heading,
            heading_level=current_level,
            text="\n".join(current_parts),
        ))

    return ParsedDocument(
        source=path.name,
        doc_title=current_heading or path.stem,
        doc_type=_infer_doc_type(path),
        sections=sections,
        raw_text=text,
    )


# ═══════════════════════════════════════════════════════════════
# Document type inference
# ═══════════════════════════════════════════════════════════════

def _infer_doc_type(path: Path) -> str:
    """Infer doc_type from filename or parent folder."""
    name = path.stem.lower()
    parent = path.parent.name.lower()

    # Check parent folder first
    if parent in ("hr", "policy", "technical", "security", "legal"):
        return parent

    # Check filename keywords
    type_keywords = {
        "hr":        ["leave", "onboarding", "employee", "handbook", "benefits"],
        "policy":    ["policy", "guideline", "procedure", "compliance"],
        "technical": ["api", "architecture", "deployment", "runbook", "sop"],
        "security":  ["security", "access", "encryption", "audit", "incident"],
    }
    for doc_type, keywords in type_keywords.items():
        if any(kw in name for kw in keywords):
            return doc_type

    return "general"


# ═══════════════════════════════════════════════════════════════
# Chunking — structure-aware with recursive fallback
# ═══════════════════════════════════════════════════════════════

def chunk_document(
    doc: ParsedDocument,
    max_size: int = 400,
    overlap: int = 50,
) -> list[Chunk]:
    """Structure-aware chunking: one chunk per section, sub-chunk if too long."""
    chunks = []

    for section in doc.sections:
        section_text = section.text.strip()
        if not section_text:
            continue

        base_meta = {
            "source": doc.source,
            "doc_title": doc.doc_title,
            "doc_type": doc.doc_type,
            "section_path": section.heading,
            "heading": section.heading,
            "heading_level": section.heading_level,
            "page": section.page,
        }

        if len(section_text) <= max_size:
            chunks.append(Chunk(text=section_text, metadata={**base_meta}))
        else:
            for i, sub_text in enumerate(_split_recursive(section_text, max_size, overlap)):
                meta = {**base_meta}
                meta["section_path"] = f"{section.heading} (part {i + 1})"
                chunks.append(Chunk(text=sub_text, metadata=meta))

    return chunks


def _split_recursive(text: str, max_size: int, overlap: int) -> list[str]:
    """Paragraph → sentence → character fallback."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= max_size:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)

    return chunks


# ═══════════════════════════════════════════════════════════════
# PII scrubbing — regex + Presidio (graceful fallback)
# ═══════════════════════════════════════════════════════════════

PII_PATTERNS = {
    "EMAIL":       re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
    "PHONE":       re.compile(r'\+?\d{0,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3,4}(?:[-.\s]?\d{4})?\b'),
    "EMP_ID":      re.compile(r'\bEMP-\d{4,6}\b'),
    "SSN":         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
}


def scrub_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Scrub PII from all chunks. Sets has_pii and pii_types in metadata."""
    for chunk in chunks:
        scrubbed_text, pii_types = _scrub_text(chunk.text)
        chunk.text = scrubbed_text
        chunk.metadata["has_pii"] = len(pii_types) > 0
        chunk.metadata["pii_types"] = pii_types
        chunk.metadata["char_length"] = len(scrubbed_text)
    return chunks


def _scrub_text(text: str) -> tuple[str, list[str]]:
    """Two-pass scrubber: regex first, then Presidio NER."""
    findings = []

    # Pass 1: Regex (fast, deterministic)
    for label, pattern in PII_PATTERNS.items():
        text, count = pattern.subn(f"[{label}]", text)
        if count:
            findings.append(label)

    # Pass 2: Presidio NER (graceful fallback if not installed)
    try:
        from presidio_anonymizer import AnonymizerEngine

        analyzer = _get_analyzer()
        results = analyzer.analyze(
            text=text,
            entities=["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"],
            language="en",
            score_threshold=0.5,
        )
        if results:
            anonymized = AnonymizerEngine().anonymize(
                text=text, analyzer_results=results,
            )
            text = anonymized.text
            findings.extend(r.entity_type for r in results)
    except ImportError:
        pass  # Presidio not installed — regex-only fallback

    return text, list(set(findings))


_analyzer_instance = None

def _get_analyzer():
    """Lazy singleton — load spaCy model once, reuse across all chunks."""
    global _analyzer_instance
    if _analyzer_instance is None:
        from presidio_analyzer import AnalyzerEngine
        _analyzer_instance = AnalyzerEngine()
    return _analyzer_instance


# ═══════════════════════════════════════════════════════════════
# Metadata finalization
# ═══════════════════════════════════════════════════════════════

def finalize_metadata(chunks: list[Chunk]) -> list[Chunk]:
    """Fill in remaining metadata fields: chunk_id, ingested_at, language."""
    now = datetime.now(timezone.utc).isoformat()
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}#{i}"
        chunk.metadata["ingested_at"] = now
        chunk.metadata["language"] = "en"
    return chunks


# ═══════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════

def ingest_corpus(
    corpus_dir: Path,
    store=None,
    embedding_model: str = "text-embedding-3-small",
) -> dict:
    """End-to-end: Parse → Chunk → Scrub → Enrich → Embed → Upsert.

    Args:
        corpus_dir: directory containing documents to ingest
        store: optional Qdrant store (if None, returns chunks without upserting)
        embedding_model: OpenAI embedding model name

    Returns:
        dict with documents_parsed, chunks_created, pii_chunks, cost_usd
    """
    print(f"Scanning {corpus_dir}...")
    files = [p for p in corpus_dir.iterdir() if p.is_file()]
    print(f"  Found {len(files)} files\n")

    # Step 1: Parse (fail-forward)
    parsed_docs = []
    for path in files:
        try:
            doc = parse_document(path)
            parsed_docs.append(doc)
            print(f"  ✓ Parsed {path.name} → {len(doc.sections)} sections")
        except Exception as e:
            print(f"  ✗ Failed {path.name}: {e}")

    # Step 2: Chunk
    all_chunks = []
    for doc in parsed_docs:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
    print(f"\n  Chunked → {len(all_chunks)} chunks total")

    # Step 3: Scrub PII
    all_chunks = scrub_chunks(all_chunks)
    pii_count = sum(1 for c in all_chunks if c.metadata.get("has_pii"))
    print(f"  PII scrubbed → {pii_count} chunks had PII")

    # Step 4: Finalize metadata
    all_chunks = finalize_metadata(all_chunks)

    # Step 5: Embed + Upsert (if store is provided)
    if store is not None:
        from src.rag.qdrant_store import embed_texts, upsert_chunks
        print(f"\n  Embedding {len(all_chunks)} chunks...")
        texts = [c.text for c in all_chunks]
        vectors = embed_texts(texts, model=embedding_model)

        print(f"  Upserting to Qdrant...")
        chunk_dicts = [{**c.metadata, "text": c.text} for c in all_chunks]
        upsert_chunks(store, chunk_dicts, vectors, collection="capstone_chunks_v2")

    # Summary
    total_chars = sum(len(c.text) for c in all_chunks)
    total_tokens = total_chars / 4
    cost = total_tokens * 0.02 / 1_000_000  # text-embedding-3-small rate

    result = {
        "documents_parsed": len(parsed_docs),
        "chunks_created": len(all_chunks),
        "pii_chunks": pii_count,
        "cost_usd": cost,
    }

    print(f"\n{'=' * 50}")
    print(f"  ✓ Documents parsed: {result['documents_parsed']}")
    print(f"  ✓ Chunks created:   {result['chunks_created']}")
    print(f"  ✓ PII-flagged:      {result['pii_chunks']}")
    print(f"  ✓ Embed cost:       ${result['cost_usd']:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════
# KB Lifecycle — Tombstones + Versioning (W10)
# ═══════════════════════════════════════════════════════════════

KB_VERSION = "v2.w10"


def tombstone_source(
    source_id: str,
    client=None,
    collection: str = "capstone_chunks_v2",
) -> int:
    """Soft-delete all chunks from a source by setting deleted_at timestamp.

    Does NOT remove data — marks it as deleted. Tombstoned chunks:
    - Are filtered out of retrieval by _live_filter()
    - Remain queryable for audit (with include_deleted=True)
    - Can be rolled back by clearing deleted_at

    Call cache.clear() AFTER this — non-negotiable.
    """
    from datetime import datetime, timezone
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if client is None:
        from src.rag.qdrant_store import get_qdrant_client
        client = get_qdrant_client()

    # Find all chunks belonging to this source
    hits, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[
            FieldCondition(key="source_id", match=MatchValue(value=source_id))
        ]),
        limit=10000,
    )

    if not hits:
        print(f"  tombstone: no chunks found for source_id={source_id!r}")
        return 0

    # Set deleted_at on each chunk (soft delete)
    now = datetime.now(timezone.utc).isoformat()
    from qdrant_client.models import SetPayloadOperation, SetPayload, PointIdsList
    client.set_payload(
        collection_name=collection,
        payload={"deleted_at": now},
        points=[h.id for h in hits],
    )

    print(f"  tombstone: marked {len(hits)} chunks as deleted (source={source_id})")
    return len(hits)


def ingest_or_update_source(
    path: Path,
    source_id: str,
    client=None,
    collection: str = "capstone_chunks_v2",
    embedding_model: str = "text-embedding-3-small",
) -> dict:
    """Ingest a document, tombstoning any previous version first.

    This is the SAFE way to update a document:
    1. Tombstone old chunks (soft delete)
    2. Parse + chunk + scrub the new version
    3. Upsert new chunks
    4. Caller must clear the cache after this (non-negotiable)

    On first ingest (no previous chunks), the tombstone step is a no-op.
    """
    if client is None:
        from src.rag.qdrant_store import get_qdrant_client
        client = get_qdrant_client()

    # Step 1: Tombstone old version (no-op if first ingest)
    tombstoned = tombstone_source(source_id, client, collection)

    # Step 2: Parse + chunk + scrub
    doc = parse_document(path)
    chunks = chunk_document(doc)
    chunks = scrub_chunks(chunks)
    chunks = finalize_metadata(chunks)

    # Tag each chunk with source_id for future tombstoning
    for c in chunks:
        c.metadata["source_id"] = source_id

    # Step 3: Embed + upsert
    from src.rag.qdrant_store import embed_texts, upsert_chunks
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, model=embedding_model)
    chunk_dicts = [{**c.metadata, "text": c.text} for c in chunks]
    upsert_chunks(client, chunk_dicts, vectors, collection=collection)

    print(f"  update: tombstoned {tombstoned} old → ingested {len(chunks)} new (source={source_id})")
    return {
        "source_id": source_id,
        "tombstoned": tombstoned,
        "ingested": len(chunks),
        "action": "update" if tombstoned > 0 else "first_ingest",
    }
