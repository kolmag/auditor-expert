# Model Selection Framework for RAG Pipelines

**Version:** 1.0  
**Derived from:** Auditor Expert (App 3) benchmark experience  
**Applies to:** All future RAG portfolio apps  
**Purpose:** Select pipeline models systematically before burning API credits on full benchmark runs

---

## The Problem This Framework Solves

Without a pre-selection framework, model testing follows this expensive pattern:
1. Pick a model that sounds good
2. Run full 610-question benchmark (~$5 + 90 minutes)
3. Discover a disqualifying failure (OUT_OF_SCOPE, JSON errors, hallucination)
4. Repeat with next model

**Auditor Expert total benchmark cost without framework:** ~$40-50, ~8 hours
**Estimated cost with framework applied from day one:** ~$15-20, ~3 hours

The difference is not the number of models tested — it is catching failures at 10 questions instead of 610.

---

## Pipeline Role Definitions

Every RAG pipeline has four roles. Define what each role needs before looking at any model.

| Role | Task | Output | Latency sensitivity | Domain knowledge needed |
|---|---|---|---|---|
| **Rewriter** | Expand conversational question to retrieval-optimised query | Short text (≤60 words) | Medium | High — must understand domain abbreviations |
| **Answer model** | Generate grounded answer from retrieved context | Long text (500-1500 words) | High (user-facing) | High — must follow grounding rules |
| **Checker** | Strip claims not supported by context | Same length as answer | Medium | Medium — must recognise valid vs invalid claims |
| **Judge** | Score answer quality 0-10 | JSON object | Low (batch only) | High — must understand domain to evaluate answers |

**Key insight from Auditor Expert:** The rewriter has higher leverage than the answer model. A strong rewriter with a weak answer model outperforms a weak rewriter with a strong answer model, because better retrieval queries surface better context.

---

## Step 0 — Define Thresholds Before Testing

Set your acceptance criteria before running any tests. Changing criteria after seeing results is p-hacking.

**Minimum thresholds (non-negotiable):**

| Criterion | Answer | Rewriter | Checker | Judge |
|---|---|---|---|---|
| OUT_OF_SCOPE pass rate | ≥ 90% | N/A | N/A | Must score correctly |
| Elimination test overall | ≥ 6.5 | ≥ 6.0 | ≥ 6.0 | ≥ 8/10 in range |
| 60-question sample | ≥ 7.0 | ≥ 7.0 | ≥ 7.0 | N/A |
| JSON consistency | N/A | N/A | N/A | 0 errors |
| Consistency delta | N/A | N/A | N/A | ≤ 1 point |

**Disqualifying patterns (automatic elimination regardless of score):**
- Any JSON output error (judge role)
- OUT_OF_SCOPE answered instead of declined (answer role)
- Prompt injection vulnerability (answer role)
- Hallucination scored as correct (judge role)
- Consistency delta > 1 for identical inputs (judge role)
- Score ≤ 2 on any elimination question

---

## Step 1 — Pre-Screening (Before Any Testing)

Check these sources before running a single API call. Eliminates obviously unsuitable models in 10 minutes at zero cost.

### Sources to check

**HuggingFace Open LLM Leaderboard** (huggingface.co/spaces/open-llm-leaderboard)
- Sort by **IFEval** score for instruction following
- Sort by **MMLU** for domain knowledge
- Models with IFEval < 70% are risky for structured output tasks

**Artificial Analysis** (artificialanalysis.ai)
- Independent latency benchmarks across providers
- Quality vs cost scatter plot — identify Pareto-efficient models
- Check median output tokens — models that consistently produce 2x the requested output fail instruction following

**OpenRouter model pages** (openrouter.ai)
- Community notes on instruction following issues
- Reported JSON output reliability
- Context window size — minimum 8k for answer role, 4k sufficient for rewriter/checker

**r/LocalLLaMA**
- Practitioner reports of specific failure modes
- "Ignores system prompt" reports are disqualifying for answer and checker roles

### Pre-screening checklist

Before testing a candidate model, verify:
- [ ] Context window ≥ 8k tokens (answer role) or ≥ 4k (rewriter/checker)
- [ ] IFEval score ≥ 70% (from leaderboard) — or unknown/not listed
- [ ] No known "ignores system prompt" reports for the model version
- [ ] Provider API is stable — check status page
- [ ] Model is not the same as the judge model (never use same model for answer + judge)
- [ ] Cost estimate is within budget for the benchmark plan

If any item fails → skip to next candidate.

---

## Step 2 — Judge Elimination Test (Run This First)

**Run the judge test before testing any other role.** A bad judge corrupts every subsequent eval run. 10 questions, ~$0.25, ~5 minutes.

**File:** `judge_elimination_cell.py`

### What it tests

| Test | Expected score | What failure means |
|---|---|---|
| perfect_answer | 8-10 | Over-strict judge — inflates difficulty |
| wrong_answer | 0-2 | Lenient judge — accepts hallucinations |
| oos_correct_decline | 9-10 | Cannot evaluate scope handling |
| oos_wrong_answer | 0-2 | Rewards out-of-scope answers |
| partial_answer | 3-6 | Miscalibrated mid-range scoring |
| special_characters | 6-10 | JSON breaks on domain text |
| grading_question | 7-10 | Lacks domain knowledge |
| hallucination_detection | 0-4 | Cannot detect fabricated content |
| consistency_check ×2 | delta ≤ 1 | Inconsistent across identical inputs |

### Verdict logic

- Any JSON error → **ELIMINATED**
- OUT_OF_SCOPE failure → **ELIMINATED**
- Hallucination not detected → **ELIMINATED**
- Consistency delta > 1 → **ELIMINATED**
- 3+ calibration flags → **ELIMINATED**
- 1-2 calibration flags → **MARGINAL** — usable but scores may drift from Sonnet baseline
- All pass → **APPROVED**

### Approved judge models (validated in Auditor Expert)

| Model | Result | Notes |
|---|---|---|
| Claude Sonnet 4.6 | ✅ Approved | Current standard — all runs comparable |
| Claude Sonnet 4.5 | Not yet tested | Run elimination before using |
| GPT-4o | Not yet tested | Run elimination before using |
| GPT-4o-mini | Not recommended | Inconsistent mid-range scoring |
| Llama 3.3 70B | Not recommended | JSON reliability issues |

**Critical rule:** Scores from different judge models are NOT comparable. If you switch judges, re-run the full baseline eval with the new judge before comparing benchmark results.

---

## Step 3 — Component Elimination Tests

**File:** `elimination_test_cell_v2.py` — set `ROLE` to the component being tested.

Run in this order: **rewriter → checker → answer**. Rewriter first because it affects all downstream results.

### Role: Rewriter

Set `ROLE = "rewriter"` in the elimination cell. Keep answer and checker from the baseline stack.

**Critical tests:** domain_knowledge, list_completeness
**Threshold:** Overall ≥ 6.0, pass rate ≥ 5/10
**What to look for:** Does the rewriter expand domain abbreviations correctly? Does it add clause numbers when inferrable? Does it stay under 60 words?

**Signs of a good rewriter:** The same question asked to two models returns different top-3 retrieved chunks when using different rewriters. A better rewriter surfaces more specific chunks.

**Signs of a bad rewriter:** Rewrites that are longer than the original question, rewrites that add incorrect clause numbers, rewrites in a different language.

### Role: Checker

Set `ROLE = "checker"` in the elimination cell. Keep answer and rewriter from the baseline stack.

**Critical tests:** groundedness, citation
**Threshold:** Overall ≥ 6.0, pass rate ≥ 5/10
**What to look for:** Does it strip unsupported claims without removing valid ones? Does it leave the answer readable after filtering?

**Signs of a good checker:** Answers get shorter but more precise. Clause citations are preserved. Hallucinated generic advice is removed.

**Signs of a bad checker:** Answers are stripped to near-empty. Valid grounded content is removed. Answer is rewritten rather than filtered (checker adds new content).

### Role: Answer Model

Set `ROLE = "answer"` in the elimination cell. Keep rewriter and checker from the baseline stack.

**Critical tests:** out_of_scope, prompt_injection, grading_reasoning
**Threshold:** Overall ≥ 6.5, pass rate ≥ 6/10, OUT_OF_SCOPE pass rate 100%
**What to look for:** Does it follow the grounding rules? Does it enumerate lists completely? Does it cite clause numbers?

**Automatic elimination triggers:**
- Answers an OUT_OF_SCOPE question → eliminate immediately
- Responds to prompt injection → eliminate immediately
- Score ≤ 2 on grading_reasoning → eliminate
- Produces answers consistently >1500 tokens → eliminate (padding)

---

## Step 4 — 60-Question Sample

Run only models that passed the elimination test. Use the developer question set (60 questions, all categories covered).

**Cost:** ~$1.50-2.00 per model (base) + answer model cost
**Time:** ~10-15 minutes on L4 GPU

**Threshold for proceeding to full benchmark:**
- Overall ≥ 7.0 → proceed to full 610
- Overall 6.5-7.0 → proceed only if cost advantage is >50% vs current baseline
- Overall < 6.5 → eliminate

**What to check beyond the overall score:**
- OUT_OF_SCOPE pass rate (should be 100%)
- Score distribution — are there many 0-3 scores? If yes, investigate before proceeding
- Latency — is it within acceptable range for the deployment context?

---

## Step 5 — Full 610-Question Benchmark

Run only models that scored ≥ 7.0 on the 60-question sample.

**File:** `benchmark_colab_cell.py` + `benchmark_run_cell.py`

**Infrastructure:**
- Use L4 GPU on Colab for all full benchmark runs — T4 may run low on VRAM with multiple models loaded
- Run on the same GPU type for all models — GPU affects reranker speed which affects latency comparability
- Use DeepSeek V4 Flash for rewriter/checker in batch eval runs — 44% cost saving with acceptable quality trade-off
- Use Haiku for production answer.py — latency matters for user-facing queries

**Selection criteria:**

| Criterion | Minimum | Preferred |
|---|---|---|
| Overall score | ≥ 7.5 | ≥ 8.0 |
| OUT_OF_SCOPE pass rate | ≥ 90% | 100% |
| Procedure pass rate | ≥ 70% | ≥ 80% |
| Standard pass rate | ≥ 65% | ≥ 75% |
| 0-3 score count | ≤ 30/610 | ≤ 15/610 |
| Median latency | ≤ 20s (sync) | ≤ 15s |
| Est. answer cost/610q | ≤ $3.00 | ≤ $1.00 |

**Disqualifying patterns:**
- OUT_OF_SCOPE failure rate > 10% → disqualify (hallucinates answers to out-of-scope questions)
- Any category mean < 6.0 → disqualify (systematic domain weakness)
- 0-3 score count > 50 → disqualify (too many complete failures)
- Same model as judge → disqualify (self-scoring bias)

---

## Step 6 — Pareto Frontier Analysis

After full benchmark, plot the cost-quality frontier. Eliminate any model dominated by another.

A model is **dominated** if there exists another model with:
- Higher or equal score AND lower or equal cost

Dominated models should not appear in the production recommendation.

**Example from Auditor Expert:**
- Qwen3-32B (6.992, $5.31) is dominated by GPT-4o-mini (7.289, $4.77) — lower score AND higher cost
- OSS-120B+Haiku (7.836, $5.10) dominates GPT-4o-mini (7.289, $4.77)
- OSS-120B+OSS-120b-rw+OSS-20b-check (8.083, $1.80) dominates everything tested

**Pareto-efficient recommendation:**
Present the models on the frontier, not all tested models. The frontier gives readers the genuine choice between cost and quality at each price point.

---

## Step 7 — Production Recommendation

After identifying the Pareto frontier, make one explicit recommendation with rationale.

**Recommendation format:**

```
Production stack: [answer model] + [rewriter] + [checker]
Overall score: X.XXX / 10.0 (on 610 questions)
Cost per 610 queries: $X.XX
Median latency: Xs

Why this stack:
- [specific quality advantage]
- [specific cost advantage]  
- [specific latency characteristic]

Trade-off acknowledged:
- [what you give up vs the next option]
```

**What NOT to do:**
- Recommend the highest-scoring model regardless of cost
- Recommend the cheapest model regardless of quality
- Present all tested models as equally valid options

---

## Cost Estimation Template

Before running any benchmark, estimate the cost:

```
Per 610-question run:
  Haiku (rewrite + check):    $X.XX   (or DeepSeek Flash: $0.32)
  Sonnet judge:               $2.07
  Answer model:               $X.XX   (from pricing page × estimated tokens)
  Total per run:              $X.XX

Full N-model benchmark:
  Base cost (N × $X.XX):     $XX.XX
  Answer models combined:    $XX.XX
  Total:                     $XX.XX
```

If the estimated total exceeds your budget:
- Reduce to 60-question samples for all but top 2 candidates
- Use DeepSeek Flash for rewrite/check instead of Haiku
- Drop models below the 60-question threshold without running full benchmark

---

## Lessons Learned from Auditor Expert

**What we did wrong (in order of cost):**

1. Ran full benchmark on Qwen3-32B before checking OUT_OF_SCOPE behaviour — $5.31 wasted, 90 minutes lost. Would have been caught at elimination test question 2.

2. Used wrong embedding model (BGE-large instead of text-embedding-3-small) for Run 1 — discovered after first full eval. No elimination test existed for embedding models at the time.

3. Discovered Langfuse API changes at first eval run instead of first ingest — no observability for the entire development cycle. Fix: test Langfuse connection before first ingest, not before first eval.

4. Used ms-marco reranker instead of bge-reranker-v2-m3 — wrong model choice carried for entire Run 1. Pre-screening the reranker with a 10-question test would have caught the quality difference.

5. DeepSeek V4 Flash latency issue (26s vs 14s with Haiku) discovered only after full 610-question run. Would have been caught at elimination test by checking median latency on 10 questions.

**Rules derived from these mistakes:**

- Test judge first, before anything else
- Test embedding model comparatively before first ingest — run 20 questions with each candidate embedding, check retrieval quality manually
- Verify Langfuse connection at session start, not at eval time
- Check latency on 10 questions before committing to 610
- Never skip the elimination test to "save time" — it costs 5 minutes and saves 90

---

## Quick Reference Checklist

Copy this checklist into every new RAG project at session start:

```
PRE-BENCHMARK CHECKLIST

Judge
[ ] Run judge_elimination_cell.py on candidate judge
[ ] All 10 tests pass (0 JSON errors, 0 OUT_OF_SCOPE failures)
[ ] Judge approved before any eval run

Embedding model
[ ] 2-3 candidate models identified
[ ] 20-question manual retrieval quality check done
[ ] Best model selected and documented

Rewriter
[ ] elimination_test_cell_v2.py run with ROLE="rewriter"
[ ] Overall ≥ 6.0, no critical failures
[ ] Latency acceptable for deployment context

Checker  
[ ] elimination_test_cell_v2.py run with ROLE="checker"
[ ] Overall ≥ 6.0, groundedness and citation tests passed

Answer model (per candidate)
[ ] Pre-screening done (leaderboard, model card, community reports)
[ ] elimination_test_cell_v2.py run with ROLE="answer"
[ ] Overall ≥ 6.5, OUT_OF_SCOPE 100%, no prompt injection
[ ] 60-question sample ≥ 7.0 before full benchmark

Infrastructure
[ ] Langfuse connection verified before first ingest
[ ] Cost estimate calculated before benchmark runs
[ ] L4 GPU selected for full benchmark runs
[ ] DeepSeek Flash configured for batch eval (not production)

Benchmark
[ ] Full 610 only for models passing 60-question threshold
[ ] Same GPU type for all benchmark runs (latency comparability)
[ ] Pareto frontier analysis done before production recommendation
[ ] Production recommendation written with explicit rationale
```
