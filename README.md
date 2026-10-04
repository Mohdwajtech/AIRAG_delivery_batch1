# Knowledge Assistant — Capstone (W1–W12)

A production-grade RAG knowledge assistant built over 12 weeks as part of the
**Advanced Certificate Programme in Agentic AI & RAG Engineering** (IITM Pravartak).

This repo contains a complete, measured, debuggable RAG system — not a demo.
It includes: hybrid retrieval, semantic caching, PII scrubbing, an 8-KPI
evaluation framework, a debugging drill, and everything needed for the M2
milestone and DR #2 design review.

**Validated:** `python3 scripts/validate_all.py` → 117/117 checks passing.

---

## Quick Start (runs offline — no API key, no Qdrant, no cost)

```bash
git clone https://github.com/Mohdwajtech/AIRAG_delivery_batch1.git
cd AIRAG_delivery_batch1
pip install pydantic openai numpy rank-bm25 --break-system-packages
export EVAL_OFFLINE=1 && export USE_FAKE=1

python3 scripts/validate_all.py          # 117 checks across W5-W12
bash scripts/smoke_test.sh               # W11 CI smoke test
python3 scripts/debug_drill.py list      # W12 debugging drill
python3 -m pytest eval/test_kpi.py -v    # W11 eval test suite
python3 scripts/m2_check.py              # W12 M2 milestone checker
```

---

## Project Structure

```
AIRAG_delivery_batch1/
├── src/                              APPLICATION CODE
│   ├── rag/
│   │   ├── qdrant_store.py           W7:  Qdrant + embedding cache
│   │   ├── retrieval.py              W9:  BM25 + hybrid + cross-encoder rerank
│   │   ├── cache.py                  W10: Semantic cache (SQLite, cosine ≥ 0.95)
│   │   ├── trace.py                  W12: Pipeline trace (6 stages, render)
│   │   └── naive_rag.py              W6:  Naive RAG baseline
│   ├── eval/
│   │   ├── golden.py                 W5:  Golden set loader + validation
│   │   ├── judge.py                  W5:  LLM-as-judge (rubric, fake+real)
│   │   ├── ragas_metrics.py          W11: Four RAGAS metrics (offline+live)
│   │   ├── harness.py                W11: Eval row builder (synthetic+live)
│   │   ├── kpi.py                    W11: 8-KPI dashboard engine
│   │   ├── critic_creator.py         W5:  Critic-creator loop
│   │   └── pairwise.py              W5:  Pairwise comparison
│   ├── debug/
│   │   └── scenarios.py              W12: 6 drill scenarios + grader
│   ├── ingest/
│   │   └── pipeline.py               W8:  Parse→chunk→PII scrub→metadata
│   └── pipeline/                     W2:  Async pipeline foundation
├── api/
│   └── main.py                       W12: FastAPI (hybrid+cache+/health)
├── eval/
│   └── test_kpi.py                   W11: 8 pytest threshold assertions
├── scripts/
│   ├── validate_all.py               117-check master validator
│   ├── smoke_test.sh                 W11 CI smoke test
│   ├── debug_drill.py                W12 drill CLI
│   └── m2_check.py                   W12 M2 milestone checker
├── data/
│   ├── corpus/                       5 policy documents
│   └── golden_set.jsonl              20 entries (g001-g020)
├── docs/
│   ├── adr/decisions.md              ADR v2 — 13 consolidated decisions
│   ├── kpi/wk12-snapshot.md          8-KPI dashboard + W6→W12 trajectory
│   ├── runbooks/document_update.md   Tombstone runbook
│   └── dr2-one-pager.md              DR #2 defense document
├── practice_examples/week01-12/      Weekly learning demos + notebooks
└── requirements.txt
```

---

## What Each Week Built

| Week | Layer | Key addition |
|------|-------|-------------|
| W1-W4 | Foundation | LLM calls, async pipeline, FastAPI, cost tracking |
| W5 | Evaluation v1 | Golden set, LLM judge, critic-creator |
| W6 | RAG v1 | Naive RAG — retrieve + generate |
| W7 | Vector Store | Qdrant, embedding cache, similarity search |
| W8 | Ingestion | Parse (PDF/HTML/DOCX/MD), chunk, PII scrub |
| W9 | Retrieval | BM25 + dense + RRF fusion + cross-encoder rerank |
| W10 | Operations | Semantic cache, tombstones, KB lifecycle, runbooks |
| W11 | Evaluation v2 | RAGAS (4 metrics), 8-KPI dashboard, eval-as-code |
| W12 | Debugging + M2 | Trace, 4-bucket taxonomy, 6-scenario drill, M2 ship |

---

## Concept Demos (copy-paste, all run offline)

### W8 — PII Scrubbing
```bash
python3 -c "
from src.ingest.pipeline import Chunk, scrub_chunks
c = Chunk(text='Email john@acme.com, SSN 123-45-6789, card 4111-1111-1111-1111', metadata={})
s = scrub_chunks([c])
print(f'Before: {c.text}')  # not available after scrub
print(f'After:  {s[0].text}')
print(f'PII:    {s[0].metadata[\"pii_types\"]}')
"
```

### W9 — BM25 + RRF Hybrid Search
```bash
python3 -c "
from src.rag.retrieval import build_bm25_index, bm25_search, rrf_fuse
chunks = [{'text':'20 days annual leave','id':'leave'},{'text':'BYOD requires VPN','id':'byod'},{'text':'Kitchen restocked Monday','id':'kitchen'}]
build_bm25_index(chunks)
hits = bm25_search('leave policy', k=3)
for h in hits: print(f'  {h[\"id\"]}: {h[\"score\"]:.2f}')
"
```

### W10 — Semantic Cache (store → hit → version miss → clear)
```bash
python3 -c "
from src.rag.cache import SemanticCache; import numpy as np, tempfile, os
db=tempfile.mktemp(suffix='.db'); c=SemanticCache(db); v=list(np.random.randn(10).astype(float))
c.store(v,'leave?','20 days',kb_version='v2')
print('Hit:',c.lookup(v,kb_version='v2',threshold=0.95)['answer'])
print('Wrong version:',c.lookup(v,kb_version='v3',threshold=0.95))
c.clear(kb_version='v2'); print('After clear:',c.lookup(v,kb_version='v2',threshold=0.95))
os.unlink(db)
"
```

### W11 — 8-KPI Dashboard
```bash
EVAL_OFFLINE=1 python3 -c "
from src.eval.harness import build_rows, active_kb_version
from src.eval.kpi import compute_kpis
rows = build_rows(); out = compute_kpis(rows, active_kb_version())
for k,v in out['kpis'].items(): print(f'  {k}: {v}')
"
```

### W12 — Trace Reading
```bash
python3 -c "
from src.rag.trace import Trace
tr = Trace(question='Leave policy?')
tr.add('embed','3 tok',latency_ms=42); tr.add('cache','miss',{'cache_hit':False},latency_ms=5)
tr.add('retrieve','top-3',{'chunk_ids':['leave-a','leave-b','kb-441']},latency_ms=88)
tr.add('rerank','done',{'top1_score':0.95},latency_ms=61)
tr.add('prompt','built',{'has_grounding_instruction':True},latency_ms=3)
tr.add('generate','ok',{'tokens':29,'cost_usd':0.00021},latency_ms=430)
tr.answer='20 days paid leave per year.'
print(tr.render())
"
```

### W12 — Debug Drill
```bash
python3 scripts/debug_drill.py list
python3 scripts/debug_drill.py grade S1 retrieval    # ✓
python3 scripts/debug_drill.py grade S4 generation   # ✗ hint
python3 scripts/debug_drill.py answers
```

---

## Four-Bucket Debugging Taxonomy

| Bucket | Trace tell | Fix |
|--------|-----------|-----|
| **Retrieval** | Wrong/missing chunk_ids | Raise k, query expansion, retune reranker |
| **Generation** | Retrieve healthy, answer unsupported | Lower temp, tighten grounding |
| **Prompt** | `has_grounding_instruction: false` | Restore instruction, add CI check |
| **Infra** | `cache_hit: true` wrong question / `✗` error | Clear cache, retry, fix key |

**Scan order:** cache → retrieve → prompt → generate

---

## M2 Milestone

| # | Item | Check |
|---|------|-------|
| 1 | Production API | `curl localhost:8000/health` |
| 2 | Eval suite | `EVAL_OFFLINE=1 pytest eval/ -q` |
| 3 | 8-KPI dashboard | `docs/kpi/wk12-snapshot.md` |
| 4 | ADR v2 | `docs/adr/decisions.md` |
| 5 | DR #2 one-pager | `docs/dr2-one-pager.md` |

```bash
python3 scripts/m2_check.py
```

---

## Setup

```bash
# Vocareum (quick)
pip install pydantic openai numpy rank-bm25 --break-system-packages

# Local (venv)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Live mode (optional — needs API keys)
echo "OPENAI_API_KEY=sk-..." > .env
echo "QDRANT_URL=https://..." >> .env
```

---

## License

Educational project — IITM Pravartak / Vocareum.
