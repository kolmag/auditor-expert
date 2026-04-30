"""
benchmark.py — Auditor Expert Multi-Model Benchmark

Runs the same 610-question eval across multiple answer generation models.
Retrieval, reranking, query rewrite, and groundedness check are fixed.
Only the answer generation model changes between runs.

Usage:
    uv run evaluation/benchmark.py --model deepseek        # single model
    uv run evaluation/benchmark.py --model all             # all models sequentially
    uv run evaluation/benchmark.py --model deepseek --n 60 # quick 60-question test
    uv run evaluation/benchmark.py --results               # print comparison table

Models:
    gpt4o_mini      openai/gpt-4o-mini (baseline)
    gpt4o           openai/gpt-4o (premium ceiling)
    haiku           anthropic/claude-haiku-4-5
    deepseek        deepseek/deepseek-v4-pro (OpenRouter)
    gpt_oss_20b     openai/gpt-oss-20b (Groq)
    gpt_oss_120b    openai/gpt-oss-120b (Groq)
    qwen3_32b       qwen/qwen3-32b (Groq)
    llama70b        llama-3.3-70b-versatile (Groq)
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
from openai import OpenAI

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = {
    "gpt4o_mini": {
        "name": "GPT-4o-mini",
        "model_id": "gpt-4o-mini",
        "provider": "openai",
        "est_cost_per_610": 2.00,
    },
    "gpt4o": {
        "name": "GPT-4o",
        "model_id": "gpt-4o",
        "provider": "openai",
        "est_cost_per_610": 8.00,
    },
    "haiku": {
        "name": "Claude Haiku 4.5",
        "model_id": "claude-haiku-4-5",
        "provider": "anthropic",
        "est_cost_per_610": 1.50,
    },
    "deepseek": {
        "name": "DeepSeek V4 Pro",
        "model_id": "deepseek/deepseek-v4-pro",
        "provider": "openrouter",
        "est_cost_per_610": 0.80,
    },
    "gpt_oss_20b": {
        "name": "GPT-OSS-20B",
        "model_id": "openai/gpt-oss-20b",
        "provider": "groq",
        "est_cost_per_610": 0.18,
    },
    "gpt_oss_120b": {
        "name": "GPT-OSS-120B",
        "model_id": "openai/gpt-oss-120b",
        "provider": "groq",
        "est_cost_per_610": 0.36,
    },
    "qwen3_32b": {
        "name": "Qwen3-32B",
        "model_id": "qwen/qwen3-32b",
        "provider": "groq",
        "est_cost_per_610": 0.54,
    },
    "llama70b": {
        "name": "Llama 3.3 70B",
        "model_id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "est_cost_per_610": 0.97,
    },
}

# ── Provider clients ──────────────────────────────────────────────────────────

def get_client(provider: str) -> OpenAI:
    """Return an OpenAI-compatible client for the given provider."""
    if provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    elif provider == "anthropic_openai":
        # Anthropic via OpenAI-compatible endpoint (not used — Haiku uses native client)
        return OpenAI(api_key=os.environ["ANTHROPIC_API_KEY"],
                      base_url="https://api.anthropic.com/v1")
    elif provider == "openrouter":
        return OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://github.com/kolmag/auditor-expert"}
        )
    elif provider == "groq":
        return OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1"
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ── Answer system (fixed across all models) ───────────────────────────────────

ANSWER_SYSTEM = """You are an expert ISO 9001 / IATF 16949 / AS9100 auditor with 20 years of experience.

Answer the question using ONLY the provided knowledge base context chunks.

CRITICAL — GROUNDEDNESS RULE:
- Base your answer ONLY on the provided knowledge base context chunks
- Do NOT add introductory phrases like "Great question" or "In quality management..."
- Do NOT add concluding summaries or transitional filler
- Do NOT add generic audit advice unless explicitly in the retrieved context
- Do NOT substitute from general knowledge when a chunk is incomplete — omit entirely
- If a sequential process or numbered list is in the context, reproduce it in full and in order
- If the question asks "what are the X types/elements/domains/steps", enumerate ALL of them from the context — do not summarise or truncate the list
- Shorter, fully-grounded answers are better than longer mixed answers
- If the question cannot be answered from context: say so explicitly
- ALWAYS cite the source document name AND clause number for every specific requirement, threshold, or process step you state
- If a clause number appears anywhere in the retrieved context, it MUST appear in your answer
- Never state a requirement, timeline, or threshold without a citation
- When asked to grade a finding, classify a nonconformance, or assess NCR severity: always state the grade (observation/minor/major/critical), the specific clause violated, and the reasoning"""

CHECKER_SYSTEM = """You are a groundedness checker for an audit expert RAG system.
Review the answer and remove any claims NOT supported by the provided context.
Return ONLY the corrected answer text — no preamble, no explanation."""

REWRITE_SYSTEM = """You are a query rewriter for an audit knowledge base retrieval system.
Rewrite the user question to maximise retrieval of relevant chunks.
Expand abbreviations, add synonyms, include clause numbers if inferable.
Return ONLY the rewritten query — under 60 words."""

JUDGE_SYSTEM = """You are an expert evaluator for an ISO 9001 / IATF 16949 / AS9100 audit knowledge base system.

Score the answer on a scale of 0–10:
10 — Perfect: correct, complete, grounded, no hallucination, cites relevant clause/document
8–9 — Very good: correct and complete with minor omissions
6–7 — Good: mostly correct, minor gaps or one unsupported claim
4–5 — Partial: correct direction but missing key information or one significant error
2–3 — Poor: significant gaps, wrong grade, or unsupported claims
0–1 — Wrong: incorrect, major hallucination, or off-topic

Special: OUT_OF_SCOPE questions score 10 if system correctly declines, 0 if it answers.

Return ONLY JSON:
{"score": <0-10>, "reasoning": "<one sentence>", "key_gap": "<gap or none>"}"""

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_MODEL = "text-embedding-3-small"
RETRIEVAL_K = 30
RERANK_TOP_N = 7
JUDGE_MODEL = "claude-sonnet-4-6"


# ── Pipeline functions ────────────────────────────────────────────────────────

_reranker = None
_collection = None
_anthropic_client = None
_embed_fn = None


def init_pipeline(chroma_dir: str):
    """Initialise retrieval pipeline — called once per benchmark run."""
    global _reranker, _collection, _anthropic_client, _embed_fn

    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    from sentence_transformers import CrossEncoder

    print("Initialising pipeline...")
    _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    if _reranker is None:
        print(f"  Loading reranker: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL)
        print(f"  ✓ Reranker loaded")

    _embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL
    )
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    _collection = chroma_client.get_collection(
        name="auditor_expert",
        embedding_function=_embed_fn
    )
    print(f"  ✓ Collection ready: {_collection.count()} chunks")


def rewrite_query(question: str) -> str:
    try:
        groq_client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1"
        )
        r = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            max_tokens=120,
            temperature=0,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": question}
            ]
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return question


def retrieve_and_rerank(query: str) -> tuple[list[dict], str]:
    """Returns (top_chunks, context_string)."""
    results = _collection.query(
        query_texts=[query], n_results=RETRIEVAL_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = [{
        "original_text": m.get("original_text", d),
        "source": m.get("source", ""),
        "distance": dist
    } for d, m, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )]

    pairs = [(query, c["original_text"]) for c in chunks]
    scores = _reranker.predict(pairs)
    top = [c for _, c in sorted(zip(scores, chunks), key=lambda x: -x[0])][:RERANK_TOP_N]

    context = "\n\n---\n\n".join(
        f"[{c['source'].replace('.md', '')}]\n{c['original_text']}" for c in top
    )
    return top, context


def generate_answer_openai(client: OpenAI, model_id: str, question: str, context: str) -> str:
    """Generate answer using OpenAI-compatible API."""
    response = client.chat.completions.create(
        model=model_id,
        temperature=0,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Context from audit knowledge base:\n\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


def generate_answer_anthropic(model_id: str, question: str, context: str) -> str:
    """Generate answer using Anthropic native API."""
    response = _anthropic_client.messages.create(
        model=model_id,
        max_tokens=1500,
        temperature=0,
        system=ANSWER_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Context from audit knowledge base:\n\n{context}\n\nQuestion: {question}"
        }]
    )
    return response.content[0].text


def check_groundedness(answer_text: str, context: str) -> str:
    try:
        groq_client = OpenAI(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1"
        )
        r = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": CHECKER_SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER TO CHECK:\n{answer_text}"}
            ]
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return answer_text


def judge_answer(question: str, answer_text: str, expected_category: str) -> dict:
    prompt = f"Question: {question}\nExpected category: {expected_category}\nAnswer: {answer_text}"
    try:
        r = _anthropic_client.messages.create(
            model=JUDGE_MODEL, max_tokens=300, temperature=0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        import re
        raw = r.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        return json.loads(match.group() if match else raw)
    except Exception as e:
        return {"score": 0, "reasoning": f"Judge error: {e}", "key_gap": "judge_failed"}


# ── Benchmark run ─────────────────────────────────────────────────────────────

def run_benchmark(
    model_key: str,
    questions: list[dict],
    verbose: bool = True
) -> dict:
    """
    Run a full benchmark eval for one model.
    Returns summary dict with scores, latency, and cost estimate.
    """
    model_cfg = MODELS[model_key]
    model_name = model_cfg["name"]
    model_id = model_cfg["model_id"]
    provider = model_cfg["provider"]

    # Get answer client
    if provider == "anthropic":
        answer_client = None  # uses _anthropic_client directly
    else:
        answer_client = get_client(provider)

    results = []
    scores_by_cat = defaultdict(list)
    scores_by_src = defaultdict(list)
    latencies = []
    answer_times = []

    print(f"\n{'='*65}")
    print(f"BENCHMARK: {model_name} ({model_id})")
    print(f"Questions: {len(questions)}  |  Judge: {JUDGE_MODEL}")
    print(f"{'='*65}\n")

    for i, q in enumerate(questions, 1):
        start = time.time()
        question = q["question"]
        expected_cat = q.get("expected_category", "unknown")
        source = q.get("source", "unknown")

        try:
            # Step 1 — rewrite
            retrieval_query = rewrite_query(question)

            # Step 2 — retrieve + rerank
            top_chunks, context = retrieve_and_rerank(retrieval_query)

            if not top_chunks:
                answer_text = "The knowledge base does not contain information relevant to this question."
            else:
                # Step 3 — generate answer
                ans_start = time.time()
                if provider == "anthropic":
                    raw_answer = generate_answer_anthropic(model_id, question, context)
                else:
                    raw_answer = generate_answer_openai(answer_client, model_id, question, context)
                answer_times.append(time.time() - ans_start)

                # Step 4 — groundedness check
                answer_text = check_groundedness(raw_answer, context)

        except Exception as e:
            answer_text = f"ERROR: {e}"
            top_chunks = []

        # Step 5 — judge
        judgment = judge_answer(question, answer_text, expected_cat)
        score = judgment.get("score", 0)
        latency = round(time.time() - start, 2)

        results.append({
            "id": q["id"],
            "question": question,
            "expected_category": expected_cat,
            "source": source,
            "answer": answer_text,
            "score": score,
            "reasoning": judgment.get("reasoning", ""),
            "key_gap": judgment.get("key_gap", ""),
            "latency": latency,
            "sources": [c["source"] for c in top_chunks]
        })

        scores_by_cat[expected_cat].append(score)
        scores_by_src[source].append(score)
        latencies.append(latency)

        if verbose:
            status = "✓" if score >= 7 else ("△" if score >= 4 else "✗")
            print(f"  {status} [{i:03d}/{len(questions)}] {q['id']} | {expected_cat:<14} "
                  f"| score={score:2d} | {judgment.get('reasoning', '')[:52]}")

        time.sleep(0.2)

    # Summary
    import statistics
    all_scores = [r["score"] for r in results]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    median_latency = statistics.median(latencies) if latencies else 0
    median_answer_time = statistics.median(answer_times) if answer_times else 0

    summary = {
        "model_key": model_key,
        "model_name": model_name,
        "model_id": model_id,
        "provider": provider,
        "overall": round(overall, 3),
        "n_questions": len(questions),
        "judge_model": JUDGE_MODEL,
        "median_latency_s": round(median_latency, 2),
        "median_answer_time_s": round(median_answer_time, 2),
        "est_answer_cost_usd": model_cfg["est_cost_per_610"] * len(questions) / 610,
        "by_category": {
            cat: {
                "mean": round(sum(s) / len(s), 3),
                "n": len(s),
                "pass_rate": round(sum(1 for x in s if x >= 7) / len(s), 3)
            }
            for cat, s in sorted(scores_by_cat.items())
        },
        "by_source": {
            src: round(sum(s) / len(s), 3)
            for src, s in sorted(scores_by_src.items())
        },
        "score_distribution": {
            "10": sum(1 for s in all_scores if s == 10),
            "8-9": sum(1 for s in all_scores if 8 <= s <= 9),
            "6-7": sum(1 for s in all_scores if 6 <= s <= 7),
            "4-5": sum(1 for s in all_scores if 4 <= s <= 5),
            "0-3": sum(1 for s in all_scores if s <= 3),
        }
    }

    # Print summary
    print(f"\n{'='*65}")
    print(f"{model_name}: {overall:.3f} / 10.0  |  Latency: {median_latency:.1f}s  "
          f"|  Answer time: {median_answer_time:.1f}s")
    print(f"\nBy category:")
    for cat, stats in summary["by_category"].items():
        bar = "█" * int(stats["mean"])
        print(f"  {cat:<16} {stats['mean']:.3f}  {bar}  (pass={stats['pass_rate']:.0%})")
    print(f"{'='*65}\n")

    return summary, results


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison_table(results_dir: Path):
    """Load all benchmark results and print comparison table."""
    files = sorted(results_dir.glob("benchmark_*.json"))
    if not files:
        print("No benchmark results found.")
        return

    summaries = []
    for f in files:
        data = json.loads(f.read_text())
        summaries.append(data["summary"])

    # Sort by overall score descending
    summaries.sort(key=lambda x: x["overall"], reverse=True)

    print(f"\n{'='*80}")
    print(f"MULTI-MODEL BENCHMARK — Auditor Expert")
    print(f"{'='*80}")
    print(f"{'Model':<22} {'Overall':>8} {'Procedure':>10} {'Standard':>10} "
          f"{'Example':>8} {'Latency':>9} {'Est. Cost':>10}")
    print(f"{'-'*80}")
    for s in summaries:
        cats = s["by_category"]
        print(f"  {s['model_name']:<20} {s['overall']:>8.3f} "
              f"{cats.get('procedure', {}).get('mean', 0):>10.3f} "
              f"{cats.get('standard', {}).get('mean', 0):>10.3f} "
              f"{cats.get('example', {}).get('mean', 0):>8.3f} "
              f"{s['median_latency_s']:>8.1f}s "
              f"${s['est_answer_cost_usd']:>8.2f}")
    print(f"{'='*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auditor Expert multi-model benchmark")
    parser.add_argument("--model", type=str, required=False, default=None,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Model to benchmark, or 'all' for sequential run")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of questions (default: all 610)")
    parser.add_argument("--results", action="store_true",
                        help="Print comparison table from saved results")
    parser.add_argument("--chroma-dir", type=str,
                        default=str(Path(__file__).parent.parent / "chroma_db"),
                        help="Path to ChromaDB directory")
    args = parser.parse_args()

    if args.results:
        print_comparison_table(RESULTS_DIR)
        return

    if not args.model:
        parser.print_help()
        return

    # Load questions
    questions = []
    for fname in ["tests_auditor.jsonl", "tests_external.jsonl"]:
        fpath = EVAL_DIR / fname
        if fpath.exists():
            batch = [json.loads(l) for l in fpath.read_text().splitlines() if l.strip()]
            questions.extend(batch)

    if args.n:
        questions = questions[:args.n]
        print(f"Using first {args.n} questions")

    print(f"Total questions: {len(questions)}")

    # Init pipeline once
    init_pipeline(args.chroma_dir)

    # Run benchmark(s)
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        summary, results = run_benchmark(model_key, questions)

        # Save results
        out_path = RESULTS_DIR / f"benchmark_{model_key}.json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"✓ Saved: {out_path}")

    # Print comparison if multiple models run
    if len(models_to_run) > 1:
        print_comparison_table(RESULTS_DIR)


if __name__ == "__main__":
    main()
