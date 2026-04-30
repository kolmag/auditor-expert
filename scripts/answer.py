"""
answer.py — Auditor Expert retrieval and answer generation pipeline

Architecture (Async):
1. Query rewrite (GPT-OSS) — expands question (Cached for repeats)
2. ChromaDB retrieval (text-embedding-3-small, K=30)
3. BGE cross-encoder reranking — BAAI/bge-reranker-v2-m3
4. Answer generation (GPT-OSS) — streaming, grounded in context
5. Groundedness check & Judge Eval — parallelized where applicable

All LLM calls at temperature=0 for consistency.
"""

import os
import time
import asyncio
from typing import AsyncGenerator, Optional

from anthropic import AsyncAnthropic
import chromadb
from dotenv import load_dotenv
from langfuse import Langfuse
from openai import AsyncOpenAI

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

COLLECTION_NAME = "auditor_expert"
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBED_MODEL = "text-embedding-3-small"

RETRIEVAL_K = 30          
RERANK_TOP_N = 7          
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

REWRITE_MODEL = "openai/gpt-oss-120b"      
ANSWER_MODEL  = "openai/gpt-oss-120b"      
CHECKER_MODEL = "openai/gpt-oss-20b"       
JUDGE_MODEL = "claude-sonnet-4-6"   

ANSWER_SYSTEM = """You are an expert ISO 9001 / IATF 16949 / AS9100 auditor with 20 years of experience in manufacturing quality management, supplier auditing, and NCR writing.

Answer the question using ONLY the provided knowledge base context chunks.

CRITICAL — GROUNDEDNESS RULE:
- Base your answer ONLY on the provided knowledge base context chunks
- Do NOT add introductory phrases like "Great question" or "In quality management..." or "Certainly!"
- Do NOT add concluding summaries or transitional filler
- Do NOT add generic audit advice unless it is explicitly stated in the retrieved context
- Do NOT substitute from general knowledge when a chunk is incomplete — omit the claim entirely
- If a sequential process or numbered list is in the context, reproduce it in full and in order
- If the question asks "what are the X types/elements/domains/steps", enumerate ALL of them from the context — do not summarise or truncate the list
- Always include clause numbers and document references when they appear in the retrieved context
- Shorter, fully-grounded answers are better than longer mixed answers
- If the question cannot be answered from the provided context, say so explicitly: "The knowledge base does not contain information on this specific question."

FORMAT:
- Answer in clear prose unless the context contains a numbered list or table — then reproduce that structure
- ALWAYS cite the source document name AND clause number for every specific requirement, threshold, or process step you state, e.g. [audit_reporting_communication.md, ISO 19011 clause 6.6]
- If a clause number appears anywhere in the retrieved context, it MUST appear in your answer
- Never state a requirement, timeline, or threshold without a citation — if you cannot find a citation in the context, omit the claim entirely
- When asked to grade a finding, classify a nonconformance, or assess NCR severity: always state the grade (observation/minor/major/critical), the specific clause violated, and the reasoning based on the grading criteria in the retrieved context
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

JUDGE_SYSTEM = """You are an expert evaluator grading an AI's response to an audit question.
Evaluate if the answer is accurate, helpful, and directly addresses the user's question based strictly on the provided context.
Return a brief 1-2 sentence evaluation summary and a score out of 10."""


# ── In-Memory Cache ───────────────────────────────────────────────────────────

_QUERY_CACHE = {}


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
    """Rerank retrieved chunks using BGE cross-encoder."""
    reranker = get_reranker()
    if reranker is None or not chunks:
        return chunks[:RERANK_TOP_N]

    pairs = [(query, c["original_text"]) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:RERANK_TOP_N]]


# ── Async Pipeline Components ─────────────────────────────────────────────────

async def rewrite_query(client: AsyncOpenAI, question: str) -> str:
    """Expand conversational question to retrieval-optimised form (with caching)."""
    if question in _QUERY_CACHE:
        return _QUERY_CACHE[question]

    try:
        response = await client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=120,
            temperature=0,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": question}
            ]
        )
        content = response.choices[0].message.content
        result = content.strip() if content and content.strip() else question
    except Exception:
        result = question

    _QUERY_CACHE[question] = result
    return result

def retrieve_chunks_sync(collection, query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Query ChromaDB (synchronous wrapped for threadpool)."""
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    if not results["documents"] or not results["documents"][0]:
        return chunks
        
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "embed_text": doc,
            "original_text": meta.get("original_text", doc) if meta else doc,
            "source": meta.get("source", "") if meta else "",
            "distance": dist
        })
    return chunks

async def check_groundedness(client: AsyncOpenAI, answer: str, context: str) -> str:
    """Actor/critic — strips claims not supported by retrieved context."""
    try:
        response = await client.chat.completions.create(
            model=CHECKER_MODEL,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": CHECKER_SYSTEM},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER TO CHECK:\n{answer}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return answer

async def run_judge(client: AsyncAnthropic, answer: str, question: str, context: str) -> str:
    """Evaluate the generated response (For Evaluation Pipeline)."""
    try:
        response = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            temperature=0,
            system=JUDGE_SYSTEM,
            messages=[
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER TO EVALUATE:\n{answer}"}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Judge evaluation failed: {str(e)}"


# ── Answer generation (Production Streaming) ──────────────────────────────────

async def answer_stream(
    question: str,
    history: Optional[list] = None,
    langfuse_trace=None
) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline with streaming (Fully Async).
    Note: Judge model is omitted here to maximize production UI speed.
    """
    start_time = time.time()

    if not question or not question.strip():
        yield "Please provide a valid question."
        return

    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    groq_client = AsyncOpenAI(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL
    )
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    # Step 1 — query rewrite
    retrieval_query = await rewrite_query(groq_client, question)
    
    if langfuse_trace:
        langfuse_trace.span(name="query_rewrite", output={"rewritten": retrieval_query})

    # Step 2 & 3 — retrieve & rerank (non-blocking threadpool)
    raw_chunks = await asyncio.to_thread(retrieve_chunks_sync, collection, retrieval_query)
    top_chunks = await asyncio.to_thread(rerank_chunks, retrieval_query, raw_chunks)

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

    # Build messages
    messages = []
    if history:
        for h in history[-4:]:
            if isinstance(h, dict):
                messages.append({"role": h["role"], "content": h["content"]})
            else:
                if h[0]: messages.append({"role": "user", "content": h[0]})
                if h[1]: messages.append({"role": "assistant", "content": h[1]})

    messages.append({
        "role": "user",
        "content": f"Context from audit knowledge base:\n\n{context}\n\nQuestion: {question}"
    })

    # Step 4 — stream answer
    stream = await groq_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}] + messages,
        temperature=0,
        stream=True,
        max_tokens=1500
    )

    full_answer = ""
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_answer += delta
        yield full_answer

    # Step 5 — Groundedness check (post-stream validation)
    checked_answer = await check_groundedness(groq_client, full_answer, context)

    try:
        lf = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        lf.create_event(
            name="answer_stream",
            input={"question": question, "rewritten_query": retrieval_query},
            output={
                "answer": checked_answer[:500],
                "sources": [c["source"] for c in top_chunks],
                "latency": round(time.time() - start_time, 2)
            }
        )
        lf.flush()
    except Exception:
        pass

    if checked_answer != full_answer:
        yield checked_answer


# ── Answer generation (Evaluation Pipeline) ───────────────────────────────────

async def answer(question: str) -> dict:
    """
    Non-streaming answer for evaluation pipeline.
    Runs checker and judge strictly in parallel for maximum speed.
    """
    start_time = time.time()

    if not question or not question.strip():
        return {"answer": "Please provide a valid question.", "sources": []}
        
    anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    groq_client = AsyncOpenAI(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    embed_fn = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=EMBED_MODEL
    )
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )

    retrieval_query = await rewrite_query(groq_client, question)
    
    raw_chunks = await asyncio.to_thread(retrieve_chunks_sync, collection, retrieval_query)
    top_chunks = await asyncio.to_thread(rerank_chunks, retrieval_query, raw_chunks)

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

    # Await standard LLM generation
    response = await openai_client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "system", "content": ANSWER_SYSTEM}] + messages,
        temperature=0,
        max_tokens=1500
    )
    raw_answer = response.choices[0].message.content

    # Run check and judge in parallel
    final_answer, judge_feedback = await asyncio.gather(
        check_groundedness(groq_client, raw_answer, context),
        run_judge(anthropic_client, raw_answer, question, context)
    )

    latency = round(time.time() - start_time, 2)

    try:
        lf = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        lf.create_event(
            name="answer",
            input={"question": question, "rewritten_query": retrieval_query},
            output={
                "answer": final_answer[:500],
                "judge_feedback": judge_feedback,
                "sources": [c["source"] for c in top_chunks],
                "latency": latency
            }
        )
        lf.flush()
    except Exception:
        pass

    return {
        "answer": final_answer,
        "judge_feedback": judge_feedback,
        "sources": [c["source"] for c in top_chunks],
        "retrieval_query": retrieval_query,
        "latency": latency
    }