"""
ingest.py — Auditor Expert KB ingestion pipeline

Lessons applied from CAPA/8D Expert:
- CHUNK_SIZE=400 (not 500) — prevents BGE 512-token silent truncation
- temperature=0 on ALL enrichment calls — deterministic embeddings, comparable evals
- Markdown-header-aware chunking — respects semantic boundaries
- LLM-enriched embed_text: headline + summary + practitioner_queries + original
- README.md excluded from ingestion
- --reset flag: only use when full regeneration is intended
- --upsert flag: safe for additions without destroying existing embeddings

Usage:
    uv run scripts/ingest.py --reset     # full re-ingest (destroys existing embeddings)
    uv run scripts/ingest.py --upsert    # add/update only (safe for additions)
    uv run scripts/ingest.py             # defaults to --upsert
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import anthropic
import chromadb
import tiktoken
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

KB_DIR = Path(__file__).parent.parent / "knowledge-base" / "markdown"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "auditor_expert"

CHUNK_SIZE = 400          # GPT tokens — keeps BGE enriched chunk under 512 tokens
CHUNK_OVERLAP = 40        # 10% overlap
HEADER_MAX_TOKENS = 800   # sections larger than this fall back to token splitting

ENRICHMENT_MODEL = "claude-haiku-4-5"   # fast, cheap, deterministic at temp=0
EMBED_MODEL = "text-embedding-3-small"  # OpenAI

ENRICHMENT_PROMPT = """You are an expert ISO 9001 / IATF 16949 / AS9100 auditor with 20 years of experience.

Analyse the following chunk from an audit knowledge base document and return a JSON object.

For each chunk, return JSON with exactly these fields:
- "headline": single precise sentence (max 20 words) capturing the main concept — include clause numbers, standard names, NCR grades, and specific thresholds where present
- "summary": 2-3 sentences of contextual explanation including clause numbers, standard names, decision thresholds, and audit terminology
- "practitioner_queries": list of exactly 3 questions an auditor or auditee would ask in real life — conversational, urgent, practitioner phrasing (e.g. "what evidence do I need for 8.4?", "is this a minor or major finding?", "can I grade this as observation instead of minor?")

Return ONLY valid JSON. No preamble, no markdown fences, no explanation outside the JSON."""


# ── Chunking ──────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Count GPT tokens using cl100k_base (proxy for chunk size decisions)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def split_by_headers(text: str) -> list[str]:
    """
    Split markdown on ## and ### headers first.
    Sections exceeding HEADER_MAX_TOKENS fall back to token-based splitting.
    Preserves semantic boundaries — clause sections, numbered requirements,
    decision trees stay intact.
    """
    # Split on ## or ### headers (keep the header with the section)
    pattern = r"(?=^#{2,3} )"
    raw_sections = re.split(pattern, text, flags=re.MULTILINE)

    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        if count_tokens(section) <= HEADER_MAX_TOKENS:
            chunks.append(section)
        else:
            # Fall back to token-based splitting for large sections
            chunks.extend(token_split(section))
    return chunks


def token_split(text: str) -> list[str]:
    """
    Fallback token-based splitter for sections exceeding HEADER_MAX_TOKENS.
    Uses simple word-boundary splitting to stay near CHUNK_SIZE.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    result = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        result.append(chunk_text.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in result if c]


def split_into_chunks(text: str) -> list[str]:
    """Main chunking entry point — header-aware with token fallback."""
    return split_by_headers(text)


# ── Enrichment ────────────────────────────────────────────────────────────────

def enrich_chunk(client: anthropic.Anthropic, chunk_text: str, doc_category: str) -> dict:
    """
    Call Claude Haiku at temperature=0 to generate headline, summary,
    and practitioner_queries for a chunk.

    Returns dict with keys: headline, summary, practitioner_queries
    Falls back to minimal metadata on any error.
    """
    try:
        response = client.messages.create(
            model=ENRICHMENT_MODEL,
            max_tokens=600,
            temperature=0,   # CRITICAL — deterministic embeddings, comparable evals
            system=ENRICHMENT_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"doc_category: {doc_category}\n\n---\n\n{chunk_text}"
                }
            ]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if model adds them despite instruction
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return {
            "headline": data.get("headline", ""),
            "summary": data.get("summary", ""),
            "practitioner_queries": data.get("practitioner_queries", [])
        }
    except Exception as e:
        print(f"  ⚠ Enrichment failed: {e} — using minimal metadata")
        return {
            "headline": chunk_text[:80].replace("\n", " "),
            "summary": "",
            "practitioner_queries": []
        }


def build_embed_text(headline: str, summary: str, queries: list[str], original: str) -> str:
    """
    Construct the text that gets embedded.
    Structure: headline + summary + practitioner_queries + original_text
    This is the single highest-leverage architectural decision from CAPA/8D Expert.
    """
    queries_text = "\n".join(f"Q: {q}" for q in queries)
    return f"{headline}\n\n{summary}\n\n{queries_text}\n\n{original}"


# ── Ingestion ─────────────────────────────────────────────────────────────────

def extract_doc_category(text: str, filename: str) -> str:
    """
    Extract doc_category from frontmatter metadata line.
    Falls back to filename-based inference if not found.
    """
    match = re.search(r"\*\*doc_category:\*\*\s*(\w+)", text)
    if match:
        return match.group(1)
    # Filename-based fallback
    if "worked_example" in filename:
        return "example"
    if "checklist" in filename or "grading" in filename:
        return "reference"
    if "scenario" in filename or "dispute" in filename or "edge_case" in filename:
        return "general"
    if "iso9001" in filename or "iatf" in filename or "as9100" in filename:
        return "standard"
    return "procedure"


def ingest(reset: bool = False) -> None:
    """
    Main ingestion function.

    reset=True:  deletes and recreates the collection (--reset flag)
    reset=False: upserts only — safe for additions without destroying embeddings
    """
    print(f"\n{'='*60}")
    print(f"Auditor Expert — KB Ingestion")
    print(f"Mode: {'RESET (full re-ingest)' if reset else 'UPSERT (additive)'}")
    print(f"{'='*60}\n")

    # ── Clients ───────────────────────────────────────────────────────────────
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # ── Embedding function ────────────────────────────────────────────────────
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL)

    # ── Collection ────────────────────────────────────────────────────────────
    if reset:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print(f"✓ Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✓ Collection ready: {COLLECTION_NAME}")

    # ── Load documents ────────────────────────────────────────────────────────
    md_files = sorted(
        f for f in KB_DIR.glob("*.md")
        if f.name != "README.md"   # git placeholder — never ingest
    )

    if not md_files:
        print(f"✗ No .md files found in {KB_DIR}")
        print("  Copy your 26 KB documents into knowledge-base/markdown/")
        sys.exit(1)

    print(f"✓ Found {len(md_files)} documents\n")

    total_chunks = 0

    for doc_path in md_files:
        text = doc_path.read_text(encoding="utf-8")
        doc_category = extract_doc_category(text, doc_path.stem)
        chunks = split_into_chunks(text)

        print(f"  {doc_path.name} [{doc_category}] → {len(chunks)} chunks")

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_path.stem}_{i:04d}"

            # Skip if already exists in upsert mode
            if not reset:
                existing = collection.get(ids=[chunk_id])
                if existing["ids"]:
                    continue

            # Enrich
            enrichment = enrich_chunk(anthropic_client, chunk_text, doc_category)
            embed_text = build_embed_text(
                enrichment["headline"],
                enrichment["summary"],
                enrichment["practitioner_queries"],
                chunk_text
            )

            # Store
            collection.upsert(
                ids=[chunk_id],
                documents=[embed_text],
                metadatas=[{
                    "source": doc_path.name,
                    "doc_category": doc_category,
                    "chunk_index": i,
                    "headline": enrichment["headline"],
                    "original_text": chunk_text,
                    "practitioner_queries": json.dumps(enrichment["practitioner_queries"])
                }]
            )
            total_chunks += 1

    try:
        langfuse.flush()
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"✓ Ingestion complete")
    print(f"  Documents: {len(md_files)}")
    print(f"  Chunks ingested: {total_chunks}")
    print(f"  Collection size: {collection.count()}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditor Expert KB ingestion")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--reset", action="store_true",
                       help="Delete and recreate collection (full re-ingest)")
    group.add_argument("--upsert", action="store_true",
                       help="Add/update only — preserves existing embeddings (default)")
    args = parser.parse_args()

    ingest(reset=args.reset)
