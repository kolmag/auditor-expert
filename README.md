# Auditor Expert

**RAG-based expert Q&A assistant for ISO 9001 / IATF 16949 / AS9100D audit knowledge**

Part of an AI engineering portfolio targeting senior quality + AI leadership roles. Built with eval-driven iteration, multi-model benchmarking, and production-grade RAG architecture.

---

## What it does

Auditor Expert answers audit-domain questions the way a senior auditor would — grounded in specific clauses, evidence standards, and grading criteria. It supports:

- **Clause interpretation** — "What does IATF 16949 clause 8.5.1.1 require for control plans?"
- **NCR grading** — "Is a missing reaction plan for SPC a minor or major finding?"
- **Evidence guidance** — "What evidence do I need to close a major NCR from a supplier audit?"
- **Audit preparation** — "What will an AS9100D auditor ask about first article inspection?"
- **Practitioner scenarios** — "My supplier submitted a CA that addresses the symptom, not the root cause — what do I do?"

The system correctly declines out-of-scope questions (HR, pricing, legal, non-quality topics) and flags when the knowledge base cannot answer a question rather than hallucinating.

---

## Architecture

```
User query
    ↓
Query Rewriter (GPT-OSS-120B via Groq)
    ↓
ChromaDB retrieval — K=30 candidates
text-embedding-3-small
    ↓
BGE reranker (BAAI/bge-reranker-v2-m3) — top 7
    ↓
Groundedness Checker (GPT-OSS-20B via Groq)
    ↓
Answer Generator (GPT-OSS-120B via Groq)
    ↓
Gradio streaming interface
```

**Judge (eval only):** Claude Sonnet 4.6

---

## Knowledge Base

27 documents covering the full audit domain:

| Category | Documents | Topics |
|---|---|---|
| `standard` | 10 | ISO 9001, IATF 16949, AS9100D clause requirements |
| `procedure` | 9 | Audit planning, NCR writing, evidence collection, CA closure |
| `example` | 3 | Worked examples with FINDING anchors (ISO 9001, IATF, supplier) |
| `reference` | 4 | NCR grading criteria, checklists, process KPIs, audit types |
| `general` | 1 | Practitioner scenarios, edge cases, clause interpretation disputes |

**584 chunks** after enrichment (headline + summary + 3 practitioner queries per chunk).

---

## Evaluation

**Eval set: 610 questions across 4 independent sources**

| Source | N | Description |
|---|---|---|
| `developer` | 60 | Structured questions with expected sources and clause references |
| `blind_practitioner` | 70 | Practitioner-phrased questions, no knowledge of document structure |
| `adversarial` | 60 | Out-of-scope questions, edge cases, prompt injection attempts |
| `external` (g1/g2/standard/forum) | 420 | Blind questions from multiple generators + real forum questions |

---

## Results

### Ablation study — three runs

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Embeddings | BGE-large-en-v1.5 | text-embedding-3-small | text-embedding-3-small |
| Reranker | ms-marco-MiniLM | bge-reranker-v2-m3 | bge-reranker-v2-m3 |
| RERANK_TOP_N | 5 | 5 | 7 |
| KB docs | 26 | 26 | 27 |
| Answer model | GPT-4o-mini | GPT-4o-mini | GPT-4o-mini |
| **Overall (190q)** | **6.800** | **7.511** | **7.584** |
| **Overall (610q)** | — | **7.059** | **7.289** |
| Median latency | ~12s (CPU) | 7.88s (GPU) | 9.65s (GPU) |

Run 1 used the wrong embedding model and reranker — reframed as an ablation study. The +0.489 overall improvement from Run 1 → Run 2 is attributable entirely to the correct stack.

### Multi-model benchmark — answer model comparison

All runs: same retrieval stack, same 610-question eval set, same judge (Claude Sonnet 4.6).

| Stack | Score | Latency | Actual cost/610q |
|---|---|---|---|
| **OSS-120B + OSS-120B rw + OSS-20B check** | **7.826** | 19.1s | **$2.46** |
| OSS-120B + Haiku rw/check | 7.836 | 13.7s | $5.10 |
| OSS-120B + DeepSeek Flash rw/check | 7.703 | 26.4s | $2.84 |
| GPT-4o-mini + Haiku (baseline) | 7.289 | 9.7s | $4.77 |
| Qwen3-32B (eliminated) | 6.992 | 13.8s | $5.31 |

**Production recommendation: OSS-120B answer + OSS-120B rewriter + OSS-20B checker**

- Same quality as OSS-120B + Haiku (delta: 0.010 — within judge variance)
- 52% cheaper than the Haiku stack ($2.46 vs $5.10 per 610 queries)
- 48% cheaper than the GPT-4o-mini baseline at +0.537 higher score
- All models on Groq — single provider, no cross-API latency variance

**Qwen3-32B eliminated:** OUT_OF_SCOPE pass rate 74% vs 96% baseline. Answers questions it should decline — disqualifying for a production audit assistant.

### Benchmark scores by category (production stack, 610q)

| Category | Score | Pass rate | vs Baseline |
|---|---|---|---|
| example | 8.474 | 95% | +1.053 |
| general | 8.096 | 89% | +0.830 |
| procedure | 7.717 | 88% | +0.765 |
| reference | 7.283 | 71% | +0.348 |
| standard | 7.164 | 72% | +0.507 |
| OUT_OF_SCOPE | 9.436 | 94% | -0.205 |
| **Overall** | **7.826** | **83%** | **+0.537** |

### Key finding: rewriter leverage > answer model quality

> OSS-20B answering with OSS-120B rewriting scored 8.033 (60q sample).
> OSS-120B answering with Haiku rewriting scored 7.836 (full 610).
>
> A stronger rewriter surfaces better context, which improves answers even from a smaller model.
> The rewriter has higher leverage than the answer model in this RAG architecture.

---

## Stack

```python
EMBED_MODEL    = "text-embedding-3-small"       # OpenAI
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"      # HuggingFace local
REWRITE_MODEL  = "openai/gpt-oss-120b"          # Groq
ANSWER_MODEL   = "openai/gpt-oss-120b"          # Groq
CHECKER_MODEL  = "openai/gpt-oss-20b"           # Groq
JUDGE_MODEL    = "claude-sonnet-4-6"            # Anthropic (eval only)
CHUNK_SIZE     = 400                            # BGE 512-token limit
CHUNK_OVERLAP  = 40
RETRIEVAL_K    = 30
RERANK_TOP_N   = 7
ENRICHMENT_TEMP = 0                             # deterministic embeddings
ANSWER_TEMP    = 0                              # deterministic answers
```

---

## Infrastructure cost analysis

| Scenario | Cost/610q | Notes |
|---|---|---|
| Production stack (Groq only) | $0.40 | OSS-120B + OSS-20B answer+rw+check |
| Judge (eval only) | $2.06 | Claude Sonnet 4.6 |
| Full eval run | $2.46 | Production stack + judge |
| Batch eval with DS Flash | $2.13 | DeepSeek Flash rw/check + judge, no answer model |

**Cost at scale (production, no judge):**

| Volume | Cost |
|---|---|
| 1,000 queries | ~$0.66 |
| 10,000 queries | ~$6.56 |
| 100,000 queries | ~$65.60 |

Async implementation (not yet applied) would reduce latency ~40% with no cost impact.

---

## Project structure

```
auditor-expert/
├── knowledge-base/
│   └── markdown/          # 27 .md documents
├── chroma_db/             # gitignored
├── evaluation/
│   ├── eval.py
│   ├── benchmark.py
│   ├── tests_auditor.jsonl    # 190 questions (developer + blind + adversarial)
│   └── tests_external.jsonl   # 420 questions (g1/g2/standard/forum)
└── scripts/
    ├── ingest.py
    ├── answer.py
    ├── app.py
    └── diagnostics/
        ├── tsne_viz.py
        └── sc_viz.py
```

---

## Setup

```bash
# Install
git clone https://github.com/kolmag/auditor-expert
cd auditor-expert
uv sync

# Configure
cp .env.example .env
# Add: ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# Ingest knowledge base
uv run scripts/ingest.py --reset

# Run diagnostics
uv run scripts/diagnostics/tsne_viz.py
uv run scripts/diagnostics/sc_viz.py

# Launch app
uv run scripts/app.py
```

---

## Evaluation

```bash
# Full 610-question eval (recommended)
uv run evaluation/eval.py --all --out results_run.json

# Developer questions only (60q)
uv run evaluation/eval.py --source developer

# Category-specific
uv run evaluation/eval.py --category standard

# Multi-model benchmark
uv run evaluation/benchmark.py --model gpt_oss_120b
uv run evaluation/benchmark.py --results   # comparison table
```

---

## Documentation

**`PRODUCTION_ARCHITECTURE.md`** — Full stack specification, model selection rationale, async pipeline design, cost analysis, and re-ingest checklist.

**`LESSONS_LEARNED_AUDITOR_EXPERT.md`** — Project retrospective covering KB architecture decisions, evaluation process lessons, benchmark findings, and start-clean checklist for App 4.

**`MODEL_SELECTION_FRAMEWORK.md`** — Reusable 7-step framework for RAG pipeline model selection: judge elimination → component elimination → 60-question sample → full benchmark → Pareto frontier → production recommendation.

---

## Lessons learned

Key findings from this project (full document: `LESSONS_LEARNED_AUDITOR_EXPERT.md`):

1. **Correct stack before first ingest.** Run 1 used the wrong embedding model — cost a full re-ingest and one wasted eval run. Cross-reference the stack spec before writing a single line of `ingest.py`.

2. **Rewriter leverage > answer model.** The query rewriter has higher impact on retrieval quality than the answer model quality. Test rewriter candidates before answer models.

3. **CHUNK_SIZE=400 prevents silent truncation.** BGE tokenizer is ~1.15× GPT token count. CHUNK_SIZE=500 causes silent truncation of enriched chunks at BGE's 512-token limit.

4. **temperature=0 on enrichment is non-negotiable.** Non-zero temperature produces different headlines on every re-ingest — scores become non-comparable across runs.

5. **Langfuse from day one.** Trace-level debugging (which chunks ranked, what BGE scored) saves hours of guesswork. Test the connection before first ingest.

6. **Always run 610 combined.** Never split developer and external eval sets into separate runs. One job, one cost, one comparable result.

7. **Model selection framework.** Run judge elimination → component elimination (10q each) → 60q sample → full 610. Catching Qwen3's OUT_OF_SCOPE failure at question 2 instead of question 610 saves 90 minutes and $5.

---

## From Portfolio to Production

This project is production-quality in architecture and evaluation rigour, but several components would need to change for industrial deployment:

**Security & access control**
- API key management via secrets manager (AWS Secrets Manager, Azure Key Vault) — not `.env` files
- Role-based access control — not all auditors should access all audit programmes
- Audit trail for all queries and answers — regulatory requirement in some industries

**Scalability**
- Move ChromaDB to a managed vector store (Pinecone, Weaviate, Qdrant Cloud) for multi-user concurrent access
- Containerise with Docker, deploy on Kubernetes or managed container service
- Rate limiting and request queuing for high-volume usage

**Knowledge base management**
- Document versioning — when a standard is revised (e.g. ISO 9001:2025), chunks must be updated and re-ingested without losing audit history
- Access controls on KB content — customer-specific requirements may be confidential
- Scheduled re-ingestion when source standards are updated

**Reliability**
- Fallback models if primary provider (Groq) is unavailable
- Response caching for frequently asked questions
- Health checks and uptime monitoring
- Async retry logic with exponential backoff on API failures

**Compliance**
- Data residency requirements — some industries require data to stay within specific regions
- GDPR considerations for any user query logging
- Audit log retention policy aligned with quality system record retention requirements

**Evaluation in production**
- Online evaluation via user feedback (thumbs up/down) rather than offline judge-only scoring
- A/B testing framework for model updates
- Drift detection — monitor if answer quality degrades as KB grows or standards change

---

## About

Built by **Maggie (Magdalena Koleva)** — Quality & HSE Manager, 20 years in supplier quality and auditing across semiconductor, automotive, and medical device industries.

Portfolio: AI-augmented quality leadership tools built for real domain problems.

- App 1: [CAPA/8D Expert](https://github.com/kolmag/capa-8d-expert) — overall 7.121/10, 197-question eval
- App 2: [8D Expert Workbench](https://github.com/kolmag/8d-expert-workbench) — integrated builder + RAG
- App 3: **Auditor Expert** — this repo
