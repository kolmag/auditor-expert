"""
answer.py — Auditor Expert retrieval and answer generation pipeline

Architecture:
1. Query rewrite (Claude Haiku) — expands conversational question to retrieval-optimised form
2. ChromaDB retrieval (BGE embeddings, K=30)
3. BGE cross-encoder reranking — local, free, ~2s after warmup
4. Answer generation (GPT-4o-mini) — streaming, grounded in retrieved context
5. Groundedness check (Claude Haiku) — strips hallucinated claims

All LLM calls at temperature=0 for consistency.
Judge model for eval: claude-sonnet-4-6 (per starter prompt).
"""

import json
import os
from typing import Generator, Optional

import anthropic
import chromadb
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import OpenAI

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

COLLECTION_NAME = "auditor_expert"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

RETRIEVAL_K = 30          # candidates for BGE reranker — more = better final ranking
RERANK_TOP_N = 5          # final chunks passed to answer model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

REWRITE_MODEL = "claude-haiku-4-5"
ANSWER_MODEL = "gpt-4o-mini"
CHECKER_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-sonnet-4-6"   # per starter prompt — NOT 4.5

ANSWER_SYSTEM = """You are an expert ISO 9001 / IATF 16949 / AS9100 auditor with 20 years of experience in manufacturing quality management, supplier auditing, and NCR writing.

Answer the question using ONLY the provided knowledge base context chunks.

CRITICAL — GROUNDEDNESS RULE:
- Base your answer ONLY on the provided knowledge base context chunks
- Do NOT add introductory phrases like "Great question" or "In quality management..." or "Certainly!"
- Do NOT add concluding summaries or transitional filler
- Do NOT add generic audit advice unless it is explicitly stated in the retrieved context
- Do NOT substitute from general knowledge when a chunk is incomplete — omit the claim entirely
- If a sequential process or numbered list is in the context, reproduce it in full and in order
- Shorter, fully-grounded answers are better than longer mixed answers
- If the question cannot be answered from the provided context, say so explicitly: "The knowledge base does not contain information on this specific question."

FORMAT:
- Answer in clear prose unless the context contains a numbered list or table — then reproduce that structure
- Cite the source document name in brackets when referencing a specific requirement, e.g. [audit_reporting_communication.md]
- Do not use headers unless the answer covers multiple distinct topics"""

CHECKER_SYSTEM = """You are a groundedness checker for an audit expert RAG system.

Review the answer below and remove any claims that are NOT supported by the provided context chunks.

Rules:
- Keep all claims that are directly stated or clearly implied in the context
- Remove claims that add generic audit advice not present in context
- Remove introductory or concluding filler
- Do not add any new information
- If the answer is already fully grounded, return it unchanged
- Return ONLY the corrected answer text — no preamble, no explanation"""

REWRITE_SYSTEM = """You are a query rewriter for an audit knowledge base retrieval system.

Rewrite the user question to maximise retrieval of relevant chunks. 

Rules:
- Expand abbreviations: NCR → nonconformity report, IATF → IATF 16949, SC → special characteristic
- Add relevant synonyms: "grade" → "grade severity major minor observation"
- Include standard clause numbers if inferable from context
- Keep the rewritten query concise — under 60 words
- Return ONLY the rewritten query — no explanation"""


# ── BGE Reranker ──────────────────────────────────────────────────────────────

_reranker = None

def get_reranker():
    """Lazy-load BGE cross-encoder reranker. Falls back to None if torch unavailable."""
    global _reranker
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
        print("✓ BGE cross-encoder reranker loaded")
        return _reranker
    except Exception as e:
        print(f"⚠ BGE reranker unavailable ({e}) — using embedding scores only")
        return None


def rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Rerank retrieved chunks using BGE cross-encoder.
    Falls back to original embedding order if reranker unavailable.
    """
    reranker = get_reranker()
    if reranker is None or not chunks:
        return chunks[:RERANK_TOP_N]

    pairs = [(query, c["original_text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:RERANK_TOP_N]]


# ── Query rewrite ─────────────────────────────────────────────────────────────

def rewrite_query(client: anthropic.Anthropic, question: str) -> str:
    """Expand conversational question to retrieval-optimised form."""
    try:
        response = client.messages.create(
            model=REWRITE_MODEL,
            max_tokens=120,
            temperature=0,
            system=REWRITE_SYSTEM,
            messages=[{"role": "user", "content": question}]
        )
        return response.content[0].text.strip()
    except Exception:
        return question  # graceful fallback to original


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_chunks(collection, query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Query ChromaDB and return top-K chunks with metadata."""
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "embed_text": doc,
            "original_text": meta.get("original_text", doc),
            "source": meta.get("source", ""),
            "doc_category": meta.get("doc_category", ""),
            "headline": meta.get("headline", ""),
            "distance": dist
        })
    return chunks


# ── Groundedness checker ──────────────────────────────────────────────────────

def check_groundedness(
    client: anthropic.Anthropic,
    answer: str,
    context: str
) -> str:
    """
    Claude Haiku actor/critic — strips claims not supported by retrieved context.
    Returns cleaned answer.
    """
    try:
        response = client.messages.create(
            model=CHECKER_MODEL,
            max_tokens=2000,
            temperature=0,
            system=CHECKER_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nANSWER TO CHECK:\n{answer}"
            }]
        )
        return response.content[0].text.strip()
    except Exception:
        return answer  # graceful fallback


# ── Answer generation (streaming) ─────────────────────────────────────────────

def answer_stream(
    question: str,
    history: Optional[list] = None,
    langfuse_trace=None
) -> Generator[str, None, None]:
    """
    Full RAG pipeline with streaming.
    Yields partial answer strings for Gradio streaming UI.

    Pipeline:
    1. Query rewrite
    2. Retrieve K=30 chunks
    3. BGE rerank → top 5
    4. Stream answer (GPT-4o-mini)
    5. Groundedness check (Claude Haiku)
    """
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # Step 1 — query rewrite
    retrieval_query = rewrite_query(anthropic_client, question)
    if langfuse_trace:
        langfuse_trace.span(name="query_rewrite", output={"rewritten": retrieval_query})

    # Step 2 — retrieve
    raw_chunks = retrieve_chunks(collection, retrieval_query)

    # Step 3 — rerank
    top_chunks = rerank_chunks(retrieval_query, raw_chunks)

    if not top_chunks:
        yield "The knowledge base does not contain information relevant to this question."
        return

    # Build context string
    context_parts = []
    for c in top_chunks:
        source = c["source"].replace(".md", "")
        context_parts.append(f"[{source}]\n{c['original_text']}")
    context = "\n\n---\n\n".join(context_parts)

    if langfuse_trace:
        langfuse_trace.span(name="retrieval", output={
            "top_chunks": [c["source"] for c in top_chunks],
            "reranked": True
        })

    # Build messages with optional conversation history
    messages = []
    if history:
        for h in history[-4:]:  # last 4 turns for context window management
            messages.append({"role": "user", "content": h[0]})
            messages.append({"role": "assistant", "content": h[1]})

    messages.append({
        "role": "user",
        "content": f"Context from audit knowledge base:\n\n{context}\n\nQuestion: {question}"
    })

    # Step 4 — stream answer
    stream = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}] + messages,
        temperature=0,
        stream=True,
        max_tokens=1500
    )

    full_answer = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_answer += delta
        yield full_answer  # Gradio streaming — yield cumulative text

    # Step 5 — groundedness check (post-stream, applies to final answer)
    checked_answer = check_groundedness(anthropic_client, full_answer, context)

    if langfuse_trace:
        langfuse_trace.span(name="groundedness_check", output={
            "original_length": len(full_answer),
            "checked_length": len(checked_answer),
            "changed": checked_answer != full_answer
        })

    # If checker modified the answer, yield the cleaned version
    if checked_answer != full_answer:
        yield checked_answer


# ── Non-streaming version (for eval) ──────────────────────────────────────────

def answer(question: str) -> dict:
    """
    Non-streaming answer for evaluation pipeline.
    Returns dict with answer text and retrieved sources.
    """
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    retrieval_query = rewrite_query(anthropic_client, question)
    raw_chunks = retrieve_chunks(collection, retrieval_query)
    top_chunks = rerank_chunks(retrieval_query, raw_chunks)

    if not top_chunks:
        return {"answer": "The knowledge base does not contain information relevant to this question.", "sources": []}

    context_parts = []
    for c in top_chunks:
        source = c["source"].replace(".md", "")
        context_parts.append(f"[{source}]\n{c['original_text']}")
    context = "\n\n---\n\n".join(context_parts)

    messages = [{
        "role": "user",
        "content": f"Context from audit knowledge base:\n\n{context}\n\nQuestion: {question}"
    }]

    response = openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}] + messages,
        temperature=0,
        max_tokens=1500
    )
    raw_answer = response.choices[0].message.content

    final_answer = check_groundedness(anthropic_client, raw_answer, context)

    return {
        "answer": final_answer,
        "sources": [c["source"] for c in top_chunks],
        "retrieval_query": retrieval_query
    }
