# Auditor Expert

RAG-powered ISO 9001 / IATF 16949 / AS9100 audit knowledge base.

**Pipeline:** BGE embeddings → LLM-enriched chunks → ChromaDB → BGE cross-encoder reranking → GPT-4o-mini generation → Claude Haiku groundedness check  
**Judge model:** claude-sonnet-4-6  
**KB:** 26 documents, ~50k words, 5 categories: standard / procedure / example / reference / general

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Copy KB documents into knowledge-base/markdown/
# (all 26 .md files — README.md is excluded automatically)

# 4. Ingest (first time — full reset)
uv run scripts/ingest.py --reset

# 5. Run diagnostics
uv run scripts/diagnostics/tsne_viz.py
uv run scripts/diagnostics/sc_viz.py

# 6. Run evaluation
uv run evaluation/eval.py --category standard   # fast category check first
uv run evaluation/eval.py                        # full 190-question eval

# 7. Launch app
uv run scripts/app.py
```

---

## Iteration Checklist (mandatory after every KB change)

**Never skip steps. Never run full eval before category eval is clean.**

1. `uv run scripts/ingest.py --reset` (full re-ingest) OR `--upsert` (additions only)
2. `uv run scripts/diagnostics/tsne_viz.py` — check for category bleeding
3. `uv run scripts/diagnostics/sc_viz.py` — check cross-category competition
4. `uv run evaluation/eval.py --category [affected_category]` — fast check
5. `uv run evaluation/eval.py` — full 190-question eval only after step 4 is clean

**Use `--upsert` for additions. Use `--reset` only when full regeneration is intended.**  
**`--reset` destroys all existing embeddings — non-deterministic across sessions.**

---

## Repo Structure

```
auditor-expert/
├── .env.example
├── .gitignore              ← chroma_db/, __pycache__, .env, *.html
├── pyproject.toml
├── README.md
├── knowledge-base/
│   └── markdown/           ← 26 .md documents + README.md placeholder
├── chroma_db/              ← gitignored
├── evaluation/
│   ├── eval.py             ← Judge: claude-sonnet-4-6
│   └── tests_auditor.jsonl ← 190 questions: developer + blind + adversarial
└── scripts/
    ├── ingest.py           ← CHUNK_SIZE=400, temp=0, header-aware chunking
    ├── answer.py           ← K=30 retrieval, BGE rerank, streaming, checker
    ├── app.py              ← Gradio Blocks, streaming, auditor persona
    └── diagnostics/
        ├── tsne_viz.py     ← 2D/3D embedding space visualisation
        └── sc_viz.py       ← cosine similarity violin + heatmap + per-query
```

---

## KB Architecture

### Pipeline decisions (all lessons from CAPA/8D Expert applied from day one)

| Decision | Value | Reason |
|---|---|---|
| CHUNK_SIZE | 400 tokens | BGE 512-token limit — prevents silent truncation |
| Enrichment temperature | 0 | Deterministic embeddings — comparable eval scores |
| Chunking | Markdown-header-aware | Preserves clause sections and numbered lists |
| Embed text | headline + summary + queries + original | Single highest-leverage decision |
| Retrieval K | 30 | More candidates for BGE to rerank |
| Rerank top-N | 5 | Final chunks passed to answer model |
| README exclusion | ✓ | git placeholder never ingested |
| FINDING anchors | ✓ | All worked examples from day one |
| Practitioner scenarios | ✓ | Every procedural document |

### Document categories

| Category | Documents | Purpose |
|---|---|---|
| `standard` | 10 | Formal clause requirements |
| `procedure` | 9 | Audit methodology and procedures |
| `example` | 3 | Worked NCR examples with FINDING anchors |
| `reference` | 3 | Decision tables, checklists, grading criteria |
| `general` | 1 | Practitioner scenarios, edge cases, disputes |

---

## Eval Design

**190 questions across 3 sources — prevents eval overfitting:**

| Source | Range | N | Design |
|---|---|---|---|
| Developer | t001–t060 | 60 | Structured, expected sources named |
| Blind practitioner | t061–t130 | 70 | Conversational, no knowledge of doc structure |
| Adversarial | t131–t190 | 60 | Out-of-scope (t131–150) + edge cases (t151–190) |

**Target scores (minimum acceptable):**

| Category | Target |
|---|---|
| standard | ≥ 7.5 |
| procedure | ≥ 7.0 |
| example | ≥ 7.0 |
| reference | ≥ 7.5 |
| general | ≥ 6.5 |
| OUT_OF_SCOPE | ≥ 8.0 (system must decline correctly) |

---

## Key Lessons Applied (from CAPA/8D Expert)

- **CHUNK_SIZE=400 not 500** — BGE tokenizer ≈ 1.15× GPT tokens; 500 GPT ≈ 575 BGE which truncates
- **temperature=0 on enrichment** — without this eval scores measure enrichment randomness not KB quality
- **`--upsert` not `--reset` for additions** — every `--reset` destroys existing embeddings
- **FINDING anchors** — took example category MRR from 0.383 to 1.000 in CAPA/8D Expert
- **Practitioner scenario sections** — formal SOP vocabulary and practitioner question vocabulary don't overlap; both needed
- **t-SNE before eval** — skipping t-SNE caused the worst regression in CAPA/8D Expert
- **Judge model is claude-sonnet-4-6** — scores NOT comparable to CAPA/8D Expert (different judge, different domain)
