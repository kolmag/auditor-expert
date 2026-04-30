"""
eval.py — Auditor Expert evaluation pipeline

Judge model: claude-sonnet-4-6 (per starter prompt — NOT 4.5)
Scores: 0–10 per question on correctness, completeness, groundedness, no hallucination
Output: per-question scores + category breakdown + overall score

Usage:
    uv run evaluation/eval.py                                   # tests_auditor.jsonl (default)
    uv run evaluation/eval.py --file tests_external.jsonl       # external questions
    uv run evaluation/eval.py --all                             # both files combined
    uv run evaluation/eval.py --category standard               # fast category check
    uv run evaluation/eval.py --source developer                # filter by source
    uv run evaluation/eval.py --n 20                            # first N questions
    uv run evaluation/eval.py --out results.json                # save results
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from answer import answer  # non-streaming version

JUDGE_MODEL = "claude-sonnet-4-6"   # per starter prompt — NOT 4.5

JUDGE_SYSTEM = """You are an expert evaluator for an ISO 9001 / IATF 16949 / AS9100 audit knowledge base system.

Score the answer to the question on a scale of 0–10 using this rubric:

10 — Perfect: correct, complete, grounded in audit standards, no hallucination, cites relevant clause/document
8–9 — Very good: correct and complete with minor omissions, well-grounded
6–7 — Good: mostly correct, may have minor gaps or one unsupported claim
4–5 — Partial: correct direction but missing key information or contains one significant error
2–3 — Poor: partially correct but significant gaps, wrong grade, or unsupported claims
0–1 — Wrong: incorrect answer, major hallucination, or completely off-topic

Special cases:
- OUT_OF_SCOPE questions: score 10 if the system correctly declines to answer, 0 if it answers as if in scope
- Questions where the KB cannot be expected to contain the answer: score 7 if the system says so clearly

Return ONLY a JSON object with:
{
  "score": <integer 0-10>,
  "reasoning": "<one sentence explaining the score>",
  "key_gap": "<most important thing missing or wrong, or 'none' if score >= 8>"
}"""


def load_questions(file_path: Path) -> list[dict]:
    """Load questions from a JSONL file."""
    questions = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def judge_answer(
    client: anthropic.Anthropic,
    question: str,
    answer_text: str,
    expected_category: str
) -> dict:
    """Score one answer using Claude Sonnet 4.6 as judge."""
    prompt = f"""Question: {question}
Expected category: {expected_category}
Answer to evaluate: {answer_text}"""
    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,   # increase from 200 — gives judge more room
            temperature=0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        import re
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Extract just the JSON object — ignore anything after closing brace
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge error: {e}", "key_gap": "judge_failed"}


def run_eval(
    questions: list[dict],
    out_path: str = None,
    verbose: bool = True
) -> dict:
    """
    Run evaluation on a list of question dicts.
    Returns summary dict with scores by category.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    results = []
    scores_by_category = defaultdict(list)
    scores_by_source = defaultdict(list)

    print(f"\n{'='*65}")
    print(f"Auditor Expert — Evaluation Run")
    print(f"Questions: {len(questions)}  |  Judge: {JUDGE_MODEL}")
    print(f"{'='*65}\n")

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        expected_cat = q.get("expected_category", "unknown")
        source = q.get("source", "unknown")

        # Get answer
        try:
            result = answer(question)
            answer_text = result["answer"]
        except Exception as e:
            answer_text = f"ERROR: {e}"

        # Judge
        judgment = judge_answer(client, question, answer_text, expected_cat)
        score = judgment.get("score", 0)

        results.append({
            "id": qid,
            "question": question,
            "expected_category": expected_cat,
            "source": source,
            "answer": answer_text,
            "score": score,
            "reasoning": judgment.get("reasoning", ""),
            "key_gap": judgment.get("key_gap", "")
        })

        scores_by_category[expected_cat].append(score)
        scores_by_source[source].append(score)

        # Progress
        if verbose:
            status = "✓" if score >= 7 else ("△" if score >= 4 else "✗")
            print(f"  {status} [{i:03d}/{len(questions)}] {qid} | {expected_cat:<16} "
                  f"| score={score:2d} | {judgment.get('reasoning', '')[:60]}")

        # Rate limiting
        time.sleep(0.3)

    # Summary
    all_scores = [r["score"] for r in results]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0

    summary = {
        "overall": round(overall, 3),
        "n_questions": len(questions),
        "judge_model": JUDGE_MODEL,
        "by_category": {
            cat: {
                "mean": round(sum(scores) / len(scores), 3),
                "n": len(scores),
                "pass_rate": round(sum(1 for s in scores if s >= 7) / len(scores), 3)
            }
            for cat, scores in sorted(scores_by_category.items())
        },
        "by_source": {
            src: round(sum(scores) / len(scores), 3)
            for src, scores in sorted(scores_by_source.items())
        },
        "score_distribution": {
            "10":  sum(1 for s in all_scores if s == 10),
            "8-9": sum(1 for s in all_scores if 8 <= s <= 9),
            "6-7": sum(1 for s in all_scores if 6 <= s <= 7),
            "4-5": sum(1 for s in all_scores if 4 <= s <= 5),
            "0-3": sum(1 for s in all_scores if s <= 3),
        }
    }

    # Print summary
    print(f"\n{'='*65}")
    print(f"OVERALL SCORE: {overall:.3f} / 10.0")
    print(f"\nBy category:")
    for cat, stats in summary["by_category"].items():
        bar = "█" * int(stats["mean"])
        print(f"  {cat:<16} {stats['mean']:.3f}  {bar}  (n={stats['n']}, "
              f"pass_rate={stats['pass_rate']:.0%})")
    print(f"\nBy source:")
    for src, mean in summary["by_source"].items():
        print(f"  {src:<20} {mean:.3f}")
    print(f"\nScore distribution: {summary['score_distribution']}")
    print(f"{'='*65}\n")

    # Save
    if out_path:
        output = {"summary": summary, "results": results}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"✓ Results saved: {out_path}")

    return summary


def main():
    eval_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="Auditor Expert evaluation")
    parser.add_argument("--file", type=str, default="tests_auditor.jsonl",
                        help="Test file in evaluation/ dir (default: tests_auditor.jsonl)")
    parser.add_argument("--all", action="store_true",
                        help="Run both tests_auditor.jsonl and tests_external.jsonl combined")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by expected_category")
    parser.add_argument("--source", type=str, default=None,
                        help="Filter by source")
    parser.add_argument("--n", type=int, default=None,
                        help="Run only first N questions")
    parser.add_argument("--out", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-question output")
    args = parser.parse_args()

    # Load questions
    if args.all:
        questions = []
        for fname in ["tests_auditor.jsonl", "tests_external.jsonl"]:
            fpath = eval_dir / fname
            if not fpath.exists():
                print(f"⚠ Not found, skipping: {fpath}")
                continue
            batch = load_questions(fpath)
            questions.extend(batch)
            print(f"✓ Loaded {len(batch)} questions from {fname}")
    else:
        fpath = eval_dir / args.file
        if not fpath.exists():
            print(f"✗ File not found: {fpath}")
            sys.exit(1)
        questions = load_questions(fpath)
        print(f"✓ Loaded {len(questions)} questions from {args.file}")

    # Apply filters
    if args.category:
        questions = [q for q in questions if q.get("expected_category") == args.category]
        print(f"Filter: category={args.category} → {len(questions)} questions")

    if args.source:
        questions = [q for q in questions if q.get("source") == args.source]
        print(f"Filter: source={args.source} → {len(questions)} questions")

    if args.n:
        questions = questions[:args.n]
        print(f"Filter: first {args.n} questions")

    if not questions:
        print("No questions match the filter criteria.")
        sys.exit(1)

    run_eval(questions, out_path=args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
