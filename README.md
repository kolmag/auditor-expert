# Auditor Expert

RAG-based expert Q&A assistant for quality engineers conducting or preparing for ISO 9001 / IATF 16949 / AS9100 audits.

**Part of the AI-augmented quality engineering portfolio** by Magdalena Koleva — [CAPA/8D Expert](https://github.com/kolmag/capa-8d-expert) · [8D Expert Workbench](https://github.com/kolmag/8d-expert-workbench)

---

## What it does

Answers practitioner audit questions grounded strictly in a curated knowledge base:

- Interpret a specific clause ("What does IATF 16949 clause 8.4.1 require for supplier control?")
- Grade a finding ("Is this a minor NCR, major NCR, or observation?")
- Write a defensible nonconformance report
- Identify evidence to collect for a given clause
- Prepare for an external audit ("What will an auditor ask about calibration?")
- Structure an audit plan and close an NCR
- Link audit findings to corrective action
- Handle edge cases — remote audits, clause interpretation disputes, semiconductor context

---

## Eval results

### Developer set — 60 structured questions (Run 7)

| Dimension | Score |
|---|---|
| Overall | 8.03 / 10 |
| Correctness | 8.22 / 10 |
| Completeness | 8.05 / 10 |
| Groundedness | 7.82 / 10 |
| MRR | 0.903 |
| NDCG@7 | 2.020 |
| Pass rate (≥7) | 88% |

By category:

| Category | Overall | Pass rate |
|---|---|---|
| Example | 9.00 | 100% |
| General | 8.80 | 100% |
| Reference | 8.00 | 91% |
| Procedure | 7.88 | 88% |
| Standard | 7.88 | 82% |

### Full benchmark — 610 questions, 8 sources

| Dimension | Score |
|---|---|
| Overall | 7.56 / 10 |
| MRR | 0.903 |
| NDCG@7 | 2.042 |

By source:

| Source | Overall | Pass rate | n | Notes |
|---|---|---|---|---|
| Adversarial (strict) | 9.55 | 95% | 20 | Out-of-scope and injection attempts |
| g2 | 7.92 | 83% | 105 | External benchmark set |
| Developer | 7.92 | 88% | 60 | Structured, expected_sources annotated |
| Blind practitioner | 7.76 | 84% | 70 | Practitioner-phrased, no KB knowledge |
| Adversarial edge | 7.35 | 70% | 40 | Boundary and nuance questions |
| forum (g4) | 7.30 | 69% | 105 | Forum-style scenario questions |
| standard (g3) | 7.25 | 67% | 105 | AS9100D and standard-heavy questions |
| g1 | 7.15 | 63% | 105 | Technical depth questions |

By category (610 questions):

| Category | Overall | Pass rate | n |
|---|---|---|---|
| Out of scope | 9.91 | 99% | 78 |
| General | 7.69 | 77% | 94 |
| Procedure | 7.48 | 80% | 187 |
| Example | 7.37 | 63% | 19 |
| Standard | 6.86 | 65% | 140 |
| Reference | 6.71 | 58% | 92 |

**Blind practitioner gap vs developer: 0.16 points** (target <1.0 — indicates minimal overfitting to developer set).

Judge: `claude-sonnet-4-6`. Multi-dimensional scoring: correctness (40%), completeness (35%), groundedness (25%).

---

## Run history (developer set, 60 questions)

| Run | Overall | MRR | NDCG | Key change |
|---|---|---|---|---|
| Run 1 (ablation) | 6.98 | 0.375 | 0.753 | BGE-large + ms-marco — wrong stack, documented as ablation |
| Run 3 | 6.87 | 0.364 | 0.757 | OSS-120B answer + OSS-20B checker, keyword rewriter |
| Run 4 | 7.58 | 0.365 | 0.781 | Multi-dim judge + strict compliance ANSWER_SYSTEM |
| Run 5 (HyDE) | 7.28 | 0.808 | 1.828 | HyDE query rewriting — MRR +0.44, overall -0.3 (reranker input bug) |
| Run 6 (reranker fix) | 7.70 | 0.858 | 1.911 | BGE receives HyDE paragraph, not original question |
| **Run 7 (final)** | **8.03** | **0.903** | **2.020** | Routing rules, FINDING anchor fix, KB patch, checker fix |

Run 1 used a different embedding model and reranker — treated as an ablation study, not a regression.

---

## Architecture

```
Question
  │
  ├─ Pre-flight: prompt injection check (pattern match, 0ms)
  │
  ▼
Query Rewriter (Groq OSS-120B) — HyDE
  Generates a 2-3 sentence hypothetical document excerpt
  that looks like a chunk from an audit procedure manual.
  Routing rules target the correct document vocabulary
  (e.g. NCR closure → corrective_action_audit_closure language).
  │
  ▼
ChromaDB retrieval (text-embedding-3-small, K=30)
  Cosine similarity on enriched embed_text:
  headline + summary + practitioner_queries + original_text
  │
  ▼
BGE Cross-Encoder Reranker (bge-reranker-v2-m3, Top-7)
  Input: HyDE paragraph (not original question)
  Runs on GPU (Colab T4/L4) or CPU fallback
  │
  ▼
Answer Generator (Groq OSS-120B)
  Strict compliance mode — zero outside knowledge
  Every claim cited with [source, clause]
  max_tokens=2500 + truncation guard
  │
  ▼
Groundedness Checker (Groq OSS-20B) — NLI actor/critic
  Strips unsupported claims
  Empty answer guard → clean decline if checker strips everything
  │
  ▼
Answer + sources + confidence + action_required
```

**Judge (eval only):** `claude-sonnet-4-6` — correctness, completeness, groundedness scored independently.

---

## Knowledge base

27 documents, 599 chunks across 5 categories:

| Category | Documents | Description |
|---|---|---|
| `standard` | 8 | ISO 9001, IATF 16949, AS9100D, semiconductor context, calibration, document control, customer satisfaction, continual improvement |
| `procedure` | 9 | Audit planning, NCR writing, evidence collection, reporting, closure, supplier auditing, remote auditing, internal audit programme, corrective action |
| `example` | 3 | Worked findings with FINDING anchors — ISO 9001 (NCR-ISO-001 through NCR-ISO-005), IATF, supplier |
| `reference` | 4 | NCR grading criteria, checklists, process vs system vs product audit, management review linkage, auditor competency, process performance indicators |
| `general` | 3 | Edge cases, clause interpretation disputes, practitioner scenarios |

All documents follow strict conventions: one semantic purpose per document, `FINDING:` anchors on all worked examples, `## Practitioner Scenarios` section in every procedural document, enriched at temperature=0 for deterministic embeddings.

---

## Eval set

610 questions across 8 sources:

| Source | Questions | Type |
|---|---|---|
| Developer (t001–t060) | 60 | Structured, `expected_sources` annotated — MRR/NDCG computed |
| Blind practitioner (t061–t130) | 70 | Practitioner-phrased, no KB knowledge |
| Adversarial strict (t131–t150) | 20 | Hard out-of-scope — system must decline |
| Adversarial edge (t151–t190) | 40 | Boundary questions, nuance, pressure scenarios |
| External g1–g4 | 420 | Independent external sets — technical depth, forum scenarios, AS9100D heavy |

---

## Stack

| Component | Model / Tool |
|---|---|
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Retrieval | ChromaDB, cosine, K=30 |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Query rewriter | `openai/gpt-oss-120b` via Groq (HyDE) |
| Answer generator | `openai/gpt-oss-120b` via Groq |
| Groundedness checker | `openai/gpt-oss-20b` via Groq |
| Judge (eval) | `claude-sonnet-4-6` |
| UI | Gradio Blocks, streaming |
| Observability | Langfuse (`@observe` decorators, score push) |
| Vector store | ChromaDB persistent |
| Enrichment | `claude-haiku-4-5` at temperature=0 |

---

## Project structure

```
auditor-expert/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
├── knowledge-base/
│   └── markdown/           # 27 .md KB documents
├── chroma_db/              # gitignored
├── evaluation/
│   ├── eval.py             # Multi-dimensional: correctness, completeness, groundedness, MRR, NDCG
│   ├── tests_auditor.jsonl # 190 questions (developer + blind + adversarial), expected_sources on t001–t060
│   └── tests_external.jsonl # 420 external questions (g1–g4)
├── scripts/
│   ├── ingest.py           # Header-aware chunking, LLM enrichment at temp=0
│   ├── answer.py           # Full async pipeline — HyDE rewriter, streaming, @observe
│   ├── app.py              # Gradio Blocks UI
│   └── diagnostics/
│       ├── tsne_viz.py
│       └── sc_viz.py
├── results/                # gitignored — eval JSON outputs
└── auditor_expert_colab.ipynb  # Colab eval + benchmark pipeline (T4 GPU)
```

---

## Running locally

```bash
uv sync
cp .env.example .env   # add OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
uv run scripts/ingest.py
uv run scripts/app.py
```

Langfuse keys optional — pipeline degrades gracefully without them.

---

## Running eval on Colab (recommended)

BGE reranker is significantly faster on T4 GPU. Use `auditor_expert_colab.ipynb`.

1. Runtime → Change runtime type → T4 GPU
2. Add secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`
3. Run cells 1–3 (mount Drive, install deps, set keys)
4. Run Cell 6b (inline ingest) or Cell 6 (script-based)
5. Run Cell 7 (pipeline + eval functions)
6. Run Cell 11 with desired test file and `compute_retrieval=True`

Full 60-question developer eval: ~8 minutes on T4. Full 610-question eval: ~80 minutes.

---

## Key design decisions

**HyDE query rewriting** — instead of keyword expansion, the rewriter generates a 2–3 sentence hypothetical document excerpt matching the vocabulary of the target document type. Routing rules in the prompt map question patterns to specific document vocabularies. MRR improved from 0.36 to 0.90 across development runs. The HyDE paragraph is passed to both ChromaDB and BGE — passing the original short question to BGE while using HyDE for retrieval produces a vocabulary mismatch that degrades reranking precision.

**LLM-enriched chunking** — each chunk is embedded as `headline + summary + practitioner_queries + original_text`. The headline anchors BGE reranking. Practitioner queries bridge formal SOP vocabulary to conversational question phrasing. All enrichment at temperature=0 — non-zero temperature produces different headlines on each re-ingest, making eval scores non-comparable across runs.

**FINDING anchors** — every worked example starts with `FINDING: [NCR-ID] [factual sentence with clause, finding type, key evidence]`. The enrichment model reproduces the NCR ID verbatim in the chunk headline. When the HyDE for a worked-example question opens with `FINDING: NCR-ISO-001`, cosine similarity between query and chunk is maximised. Without FINDING anchors, generic headlines compete with every other NCR-related chunk in the index.

**Header-aware chunking** — splits on `##`/`###` headers before token-based fallback. Chunk size 400 GPT tokens (BGE tokenizer ≈ 1.15× GPT — prevents silent truncation at BGE's 512-token limit).

**Actor/critic groundedness** — OSS-20B reviews every answer and strips unsupported claims. Empty-answer guard returns a clean decline rather than an empty string when the checker strips everything. Short-answer protection rule preserves citations on factual answers under 50 words.

**Prompt injection defence** — pre-retrieval pattern match blocks injection attempts before the pipeline runs (0ms latency). Vocabulary-based blocking deliberately avoided — terms like "ignore", "output", and "raw" appear in legitimate audit questions.

**Known limitation** — topic-drift questions where out-of-scope content (business case writing, financial planning) partially overlaps with KB vocabulary may receive a KB-grounded but off-topic response. The correct fix is a pre-retrieval intent classifier. Documented as a future improvement.
