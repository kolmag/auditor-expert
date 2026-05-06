"""
eval.py — Auditor Expert evaluation pipeline

Judge model: claude-sonnet-4-6 (per starter prompt — NOT 4.5)

Multi-dimensional scoring (per CAPA/8D Expert standard):
  - correctness   0-10  factual accuracy of answer
  - completeness  0-10  coverage of all required elements
  - groundedness  0-10  answer supported by KB context, no hallucination
  - overall       0-10  composite judge score

Retrieval metrics (requires expected_sources in question):
  - MRR    Mean Reciprocal Rank — position of first correct source in top-K
  - NDCG   Normalized Discounted Cumulative Gain — ranking quality

Latency:
  - median_latency_s  per-question wall time

Usage:
    uv run evaluation/eval.py                          # tests_auditor.jsonl (default)
    uv run evaluation/eval.py --file tests_external.jsonl
    uv run evaluation/eval.py --all                    # both files combined
    uv run evaluation/eval.py --category standard
    uv run evaluation/eval.py --source developer
    uv run evaluation/eval.py --n 20
    uv run evaluation/eval.py --out results.json
    uv run evaluation/eval.py --mrr                    # include MRR/NDCG (needs expected_sources)
"""

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from answer import answer  # async non-streaming version

JUDGE_MODEL = "claude-sonnet-4-6"  # per starter prompt — NOT 4.5

# ── Multi-dimensional judge prompt ────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator for an ISO 9001 / IATF 16949 / AS9100 audit knowledge base system.

Score the answer on THREE separate dimensions, each 0–10.

DIMENSION 1 — CORRECTNESS
Is the factual content of the answer accurate?
10 = all facts correct, clause numbers accurate, grading decisions defensible
7-9 = mostly correct with minor inaccuracies
4-6 = partially correct, one significant factual error
0-3 = wrong, major hallucination, or completely off-topic

DIMENSION 2 — COMPLETENESS
Does the answer cover all required elements the question asks for?
10 = all elements present, lists fully enumerated, nothing important missing
7-9 = mostly complete, minor omissions
4-6 = correct direction but missing key elements
0-3 = significant gaps, critical elements absent

DIMENSION 3 — GROUNDEDNESS
Is the answer supported by the retrieved context, with no unsupported claims?
10 = every claim traceable to context, sources cited, no general knowledge added
7-9 = well-grounded with minor unsupported claims
4-6 = some claims not supported by context
0-3 = significant hallucination or general knowledge substituted for KB content

Special cases (apply to all three dimensions):
- OUT_OF_SCOPE questions: score all dimensions 10 if system correctly declines, 0 if it answers
- Questions where KB cannot answer: score all 7 if system clearly says so

Return ONLY a JSON object:
{
  "correctness": <0-10>,
  "completeness": <0-10>,
  "groundedness": <0-10>,
  "overall": <0-10>,
  "reasoning": "<one sentence>",
  "key_gap": "<most important gap or 'none'>"
}

overall should reflect the weighted combination: correctness 40%, completeness 35%, groundedness 25%.
Round to nearest integer."""


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def compute_mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """
    Mean Reciprocal Rank for a single question.
    Returns 1/rank of first relevant retrieved source, or 0 if none found.
    retrieved_sources: ordered list from reranker (index 0 = top rank)
    expected_sources: list of relevant source filenames
    """
    if not expected_sources:
        return None  # skip questions without expected_sources
    expected_set = set(s.lower().replace(".md", "") for s in expected_sources)
    for rank, src in enumerate(retrieved_sources, 1):
        src_clean = src.lower().replace(".md", "")
        if src_clean in expected_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_sources: list[str], expected_sources: list[str], k: int = 7) -> float:
    """
    NDCG@k for a single question.
    Binary relevance: 1 if source is in expected_sources, 0 otherwise.
    """
    if not expected_sources:
        return None
    expected_set = set(s.lower().replace(".md", "") for s in expected_sources)

    def dcg(sources):
        score = 0.0
        for i, src in enumerate(sources[:k], 1):
            src_clean = src.lower().replace(".md", "")
            rel = 1.0 if src_clean in expected_set else 0.0
            score += rel / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved_sources)
    # Ideal DCG: all relevant docs ranked first
    ideal_sources = list(expected_sources) + [""] * (k - len(expected_sources))
    ideal_dcg = dcg(ideal_sources[:k])
    if ideal_dcg == 0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 4)


# ── Judge ─────────────────────────────────────────────────────────────────────

def judge_answer(
    client: anthropic.Anthropic,
    question: str,
    answer_text: str,
    expected_category: str
) -> dict:
    """Score one answer on correctness, completeness, groundedness, overall."""
    prompt = f"""Question: {question}
Expected category: {expected_category}
Answer to evaluate: {answer_text}"""
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=400,
            temperature=0,
            stop_sequences=["```", "


"],
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        result = json.loads(match.group() if match else raw)
        # Ensure all dimensions present
        for dim in ["correctness", "completeness", "groundedness", "overall"]:
            if dim not in result:
                result[dim] = 0
        return result
    except Exception as e:
        return {
            "correctness": 0, "completeness": 0, "groundedness": 0, "overall": 0,
            "reasoning": f"Judge error: {e}", "key_gap": "judge_failed"
        }


# ── Load questions ────────────────────────────────────────────────────────────

def load_questions(file_path: Path) -> list[dict]:
    questions = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


# ── Main eval loop ────────────────────────────────────────────────────────────

def run_eval(
    questions: list[dict],
    out_path: str = None,
    verbose: bool = True,
    compute_retrieval: bool = False
) -> dict:
    """
    Run multi-dimensional evaluation on a list of question dicts.
    Returns summary dict with scores by category + retrieval metrics if enabled.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    scores = defaultdict(lambda: defaultdict(list))  # scores[cat][dim]
    source_scores = defaultdict(lambda: defaultdict(list))
    latencies = []
    mrr_scores = []
    ndcg_scores = []
    n_mrr_questions = 0

    print(f"\n{'='*70}")
    print(f"Auditor Expert — Multi-Dimensional Evaluation")
    print(f"Questions: {len(questions)}  |  Judge: {JUDGE_MODEL}")
    print(f"Dimensions: correctness · completeness · groundedness · overall")
    if compute_retrieval:
        n_with_sources = sum(1 for q in questions if q.get("expected_sources"))
        print(f"Retrieval metrics: MRR + NDCG@7  ({n_with_sources} questions with expected_sources)")
    print(f"{'='*70}\n")

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        expected_cat = q.get("expected_category", "unknown")
        source = q.get("source", "unknown")
        expected_sources = q.get("expected_sources", [])

        start = time.time()

        # Get answer
        try:
            result = asyncio.run(answer(question))
            answer_text = result["answer"]
            retrieved_sources = result.get("sources", [])
        except Exception as e:
            answer_text = f"ERROR: {e}"
            retrieved_sources = []

        latency = round(time.time() - start, 2)
        latencies.append(latency)

        # Judge — multi-dimensional
        judgment = judge_answer(client, question, answer_text, expected_cat)
        correctness   = judgment.get("correctness", 0)
        completeness  = judgment.get("completeness", 0)
        groundedness  = judgment.get("groundedness", 0)
        overall       = judgment.get("overall", 0)

        # Retrieval metrics
        mrr = None
        ndcg = None
        if compute_retrieval and expected_sources:
            mrr  = compute_mrr(retrieved_sources, expected_sources)
            ndcg = compute_ndcg(retrieved_sources, expected_sources)
            mrr_scores.append(mrr)
            ndcg_scores.append(ndcg)
            n_mrr_questions += 1

        # Accumulate
        for dim, val in [("correctness", correctness), ("completeness", completeness),
                          ("groundedness", groundedness), ("overall", overall)]:
            scores[expected_cat][dim].append(val)
            source_scores[source][dim].append(val)

        row = {
            "id": qid,
            "question": question,
            "expected_category": expected_cat,
            "source": source,
            "answer": answer_text,
            "retrieved_sources": retrieved_sources,
            "correctness": correctness,
            "completeness": completeness,
            "groundedness": groundedness,
            "overall": overall,
            "reasoning": judgment.get("reasoning", ""),
            "key_gap": judgment.get("key_gap", ""),
            "latency": latency,
        }
        if compute_retrieval and expected_sources:
            row["expected_sources"] = expected_sources
            row["mrr"] = mrr
            row["ndcg"] = ndcg

        results.append(row)

        # Progress line
        if verbose:
            status = "✓" if overall >= 7 else ("△" if overall >= 4 else "✗")
            mrr_str = f" | MRR={mrr:.2f}" if mrr is not None else ""
            print(f"  {status} [{i:03d}/{len(questions)}] {qid} | {expected_cat:<14} "
                  f"| C={correctness:2d} Co={completeness:2d} G={groundedness:2d} Ov={overall:2d}"
                  f"{mrr_str} | {judgment.get('reasoning','')[:45]}")

        time.sleep(0.3)

    # ── Summary ────────────────────────────────────────────────────────────────

    all_overall = [r["overall"] for r in results]
    overall_mean = sum(all_overall) / len(all_overall) if all_overall else 0
    median_latency = statistics.median(latencies) if latencies else 0
    mean_mrr  = sum(mrr_scores)  / len(mrr_scores)  if mrr_scores  else None
    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else None

    def dim_stats(dim_scores: list) -> dict:
        return {
            "mean": round(sum(dim_scores) / len(dim_scores), 3),
            "n": len(dim_scores),
            "pass_rate": round(sum(1 for s in dim_scores if s >= 7) / len(dim_scores), 3)
        }

    by_category = {}
    for cat, dims in sorted(scores.items()):
        by_category[cat] = {
            "correctness":  dim_stats(dims["correctness"]),
            "completeness": dim_stats(dims["completeness"]),
            "groundedness": dim_stats(dims["groundedness"]),
            "overall":      dim_stats(dims["overall"]),
        }

    by_source = {}
    for src, dims in sorted(source_scores.items()):
        by_source[src] = {
            "overall": round(sum(dims["overall"]) / len(dims["overall"]), 3),
            "n": len(dims["overall"]),
        }

    summary = {
        "overall": round(overall_mean, 3),
        "correctness":  round(sum(r["correctness"]  for r in results) / len(results), 3),
        "completeness": round(sum(r["completeness"] for r in results) / len(results), 3),
        "groundedness": round(sum(r["groundedness"] for r in results) / len(results), 3),
        "n_questions": len(questions),
        "judge_model": JUDGE_MODEL,
        "median_latency_s": round(median_latency, 2),
        "mrr":  round(mean_mrr, 4)  if mean_mrr  is not None else None,
        "ndcg": round(mean_ndcg, 4) if mean_ndcg is not None else None,
        "n_mrr_questions": n_mrr_questions,
        "by_category": by_category,
        "by_source": by_source,
        "score_distribution": {
            "10":  sum(1 for s in all_overall if s == 10),
            "8-9": sum(1 for s in all_overall if 8 <= s <= 9),
            "6-7": sum(1 for s in all_overall if 6 <= s <= 7),
            "4-5": sum(1 for s in all_overall if 4 <= s <= 5),
            "0-3": sum(1 for s in all_overall if s <= 3),
        }
    }

    # ── Print summary ──────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Overall:       {summary['overall']:.3f} / 10.0")
    print(f"  Correctness:   {summary['correctness']:.3f}")
    print(f"  Completeness:  {summary['completeness']:.3f}")
    print(f"  Groundedness:  {summary['groundedness']:.3f}")
    print(f"  Median latency: {median_latency:.2f}s")
    if mean_mrr is not None:
        print(f"  MRR:           {mean_mrr:.4f}  (n={n_mrr_questions})")
        print(f"  NDCG@7:        {mean_ndcg:.4f}")

    print(f"\nBy category:")
    print(f"  {'Category':<16} {'Overall':>8} {'Correct':>8} {'Complete':>9} {'Ground':>7} {'Pass%':>7}")
    for cat, stats in by_category.items():
        print(f"  {cat:<16} "
              f"{stats['overall']['mean']:>8.3f} "
              f"{stats['correctness']['mean']:>8.3f} "
              f"{stats['completeness']['mean']:>9.3f} "
              f"{stats['groundedness']['mean']:>7.3f} "
              f"{stats['overall']['pass_rate']:>7.0%}")

    print(f"\nBy source:")
    for src, stats in by_source.items():
        print(f"  {src:<25} {stats['overall']:.3f}  (n={stats['n']})")

    print(f"\nDistribution: {summary['score_distribution']}")
    print(f"{'='*70}\n")

    if out_path:
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"✓ Results saved: {out_path}")

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    eval_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Auditor Expert multi-dimensional evaluation")
    parser.add_argument("--file", type=str, default="tests_auditor.jsonl")
    parser.add_argument("--all", action="store_true",
                        help="Run both tests_auditor.jsonl and tests_external.jsonl")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mrr", action="store_true",
                        help="Compute MRR and NDCG (requires expected_sources in questions)")
    args = parser.parse_args()

    if args.all:
        questions = []
        for fname in ["tests_auditor.jsonl", "tests_external.jsonl"]:
            fpath = eval_dir / fname
            if not fpath.exists():
                print(f"⚠ Not found, skipping: {fpath}")
                continue
            batch = load_questions(fpath)
            questions.extend(batch)
            print(f"✓ Loaded {len(batch)} from {fname}")
    else:
        fpath = eval_dir / args.file
        if not fpath.exists():
            print(f"✗ File not found: {fpath}")
            sys.exit(1)
        questions = load_questions(fpath)
        print(f"✓ Loaded {len(questions)} from {args.file}")

    if args.category:
        questions = [q for q in questions if q.get("expected_category") == args.category]
        print(f"Filter: category={args.category} → {len(questions)} questions")
    if args.source:
        questions = [q for q in questions if q.get("source") == args.source]
        print(f"Filter: source={args.source} → {len(questions)} questions")
    if args.n:
        questions = questions[:args.n]
        print(f"Filter: first {args.n}")
    if not questions:
        print("No questions match filters.")
        sys.exit(1)

    if args.mrr:
        n_with = sum(1 for q in questions if q.get("expected_sources"))
        if n_with == 0:
            print("⚠  --mrr flag set but no questions have expected_sources field.")
            print("   Annotate expected_sources in tests_auditor.jsonl first.")
            print("   Continuing without retrieval metrics.")

    run_eval(questions, out_path=args.out,
             verbose=not args.quiet, compute_retrieval=args.mrr)


if __name__ == "__main__":
    main()
