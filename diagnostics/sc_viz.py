"""
sc_viz.py — Cosine similarity (Sc) analysis

Three modes:
1. Default: intra-category violin + cross-category heatmap
2. --queries: per-query Sc — which categories appear in top-K for specific questions
3. --pre-ingest: run on raw document text before ingest (catches overlap at design stage)

Usage:
    uv run scripts/diagnostics/sc_viz.py                    # violin + heatmap
    uv run scripts/diagnostics/sc_viz.py --queries          # per-query analysis
    uv run scripts/diagnostics/sc_viz.py --save sc_out.html # save to file

Run heatmap BEFORE first ingest on designed KB documents to catch overlap early.
"""

import argparse
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import chromadb
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dotenv import load_dotenv
from plotly.subplots import make_subplots

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
COLLECTION_NAME = "auditor_expert"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
KB_DIR = Path(__file__).parent.parent.parent / "knowledge-base" / "markdown"

CATEGORY_COLORS = {
    "standard":  "#2563eb",
    "procedure": "#16a34a",
    "example":   "#dc2626",
    "reference": "#9333ea",
    "general":   "#ea580c",
}

# Representative queries for per-query Sc analysis
PROBE_QUERIES = [
    "what evidence do I need for a major NCR",
    "is this a minor or major finding",
    "what does IATF 16949 require for process audits",
    "how do I close a nonconformance report",
    "what are special characteristics in automotive",
    "difference between process audit and system audit",
    "what is the corrective action timeline for major NCR",
    "GR&R measurement system analysis acceptance criteria",
    "AS9100 first article inspection requirements",
    "how to grade a supplier audit finding",
]


def load_from_chroma() -> tuple[np.ndarray, list[dict]]:
    """Load embeddings and metadata from ChromaDB."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        print(f"✗ Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        sys.exit(1)

    results = collection.get(include=["embeddings", "metadatas"])
    return np.array(results["embeddings"]), results["metadatas"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_intra_category_sc(
    embeddings: np.ndarray,
    metadatas: list[dict]
) -> dict[str, list[float]]:
    """
    Compute pairwise cosine similarities within each category.
    Returns dict: category → list of similarity scores.
    """
    cat_embeddings = defaultdict(list)
    for emb, meta in zip(embeddings, metadatas):
        cat = meta.get("doc_category", "unknown")
        cat_embeddings[cat].append(emb)

    sc_by_category = {}
    for cat, embs in cat_embeddings.items():
        scores = []
        embs = np.array(embs)
        for i, j in combinations(range(len(embs)), 2):
            scores.append(cosine_similarity(embs[i], embs[j]))
        sc_by_category[cat] = scores
        print(f"  {cat:<12} n={len(embs):>4}  mean_Sc={np.mean(scores):.3f}  "
              f"std={np.std(scores):.3f}")

    return sc_by_category


def compute_cross_category_sc(
    embeddings: np.ndarray,
    metadatas: list[dict]
) -> dict[tuple, float]:
    """
    Compute mean cosine similarity between all category pairs.
    Returns dict: (cat_a, cat_b) → mean similarity.
    High cross-category Sc = retrieval competition risk.
    """
    cat_embeddings = defaultdict(list)
    for emb, meta in zip(embeddings, metadatas):
        cat = meta.get("doc_category", "unknown")
        cat_embeddings[cat].append(np.array(emb))

    categories = sorted(cat_embeddings.keys())
    cross_sc = {}

    for cat_a, cat_b in combinations(categories, 2):
        embs_a = cat_embeddings[cat_a]
        embs_b = cat_embeddings[cat_b]
        # Sample for efficiency if large
        sample_a = embs_a[:50]
        sample_b = embs_b[:50]
        scores = []
        for a in sample_a:
            for b in sample_b:
                scores.append(cosine_similarity(a, b))
        mean_sc = float(np.mean(scores))
        cross_sc[(cat_a, cat_b)] = mean_sc

    return cross_sc, categories


def plot_violin(sc_by_category: dict[str, list[float]], save_path: str = None):
    """Violin plot of intra-category Sc distributions."""
    fig = go.Figure()
    for cat, scores in sorted(sc_by_category.items()):
        fig.add_trace(go.Violin(
            y=scores,
            name=cat,
            box_visible=True,
            meanline_visible=True,
            fillcolor=CATEGORY_COLORS.get(cat, "#666"),
            opacity=0.7,
            line_color=CATEGORY_COLORS.get(cat, "#666"),
        ))

    fig.update_layout(
        title="Intra-Category Cosine Similarity (Sc) — Auditor Expert",
        yaxis_title="Cosine Similarity",
        xaxis_title="doc_category",
        height=500,
        showlegend=False,
        yaxis=dict(range=[0, 1])
    )

    if save_path:
        fig.write_html(save_path.replace(".html", "_violin.html"))
        print(f"✓ Saved violin: {save_path.replace('.html', '_violin.html')}")
    else:
        fig.show()


def plot_heatmap(cross_sc: dict, categories: list, save_path: str = None):
    """Cross-category Sc heatmap — high values = retrieval competition risk."""
    n = len(categories)
    matrix = np.zeros((n, n))

    for i, cat_a in enumerate(categories):
        matrix[i][i] = 1.0  # self-similarity
        for j, cat_b in enumerate(categories):
            if i != j:
                key = (cat_a, cat_b) if (cat_a, cat_b) in cross_sc else (cat_b, cat_a)
                matrix[i][j] = cross_sc.get(key, 0.0)

    fig = ff.create_annotated_heatmap(
        z=matrix.tolist(),
        x=categories,
        y=categories,
        annotation_text=[[f"{v:.2f}" for v in row] for row in matrix],
        colorscale="RdYlGn_r",
        zmin=0.0, zmax=1.0,
        showscale=True
    )
    fig.update_layout(
        title="Cross-Category Cosine Similarity Heatmap — High = Retrieval Competition Risk",
        height=450,
        width=600
    )

    if save_path:
        fig.write_html(save_path.replace(".html", "_heatmap.html"))
        print(f"✓ Saved heatmap: {save_path.replace('.html', '_heatmap.html')}")
    else:
        fig.show()


def per_query_sc(save_path: str = None):
    """
    For each probe query, retrieve top-K chunks and show which categories appear.
    Identifies which categories compete for retrieval on specific question types.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    print("\nPer-query category distribution (top-10 chunks):\n")
    rows = []

    for query in PROBE_QUERIES:
        results = collection.query(query_texts=[query], n_results=10, include=["metadatas"])
        cats = [m.get("doc_category", "?") for m in results["metadatas"][0]]
        cat_counts = {c: cats.count(c) for c in set(cats)}
        top_cats = ", ".join(f"{c}({n})" for c, n in sorted(
            cat_counts.items(), key=lambda x: -x[1]))
        print(f"  Q: {query[:55]:<55}  → {top_cats}")
        rows.append({"query": query, "categories": top_cats})

    if save_path:
        import json
        out_path = save_path.replace(".html", "_per_query.json")
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n✓ Saved per-query results: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Sc cosine similarity analysis")
    parser.add_argument("--queries", action="store_true",
                        help="Run per-query Sc analysis")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plots to HTML files")
    args = parser.parse_args()

    if args.queries:
        per_query_sc(save_path=args.save)
        return

    print("Loading embeddings from ChromaDB...")
    embeddings, metadatas = load_from_chroma()
    print(f"✓ {len(embeddings)} chunks loaded\n")

    print("Intra-category Sc:")
    sc_by_category = compute_intra_category_sc(embeddings, metadatas)

    print("\nCross-category Sc:")
    cross_sc, categories = compute_cross_category_sc(embeddings, metadatas)
    for (a, b), sc in sorted(cross_sc.items(), key=lambda x: -x[1]):
        risk = "⚠ HIGH" if sc > 0.85 else ("→ medium" if sc > 0.75 else "  ok")
        print(f"  {a:<12} ↔ {b:<12}  Sc={sc:.3f}  {risk}")

    print("\nGenerating plots...")
    plot_violin(sc_by_category, save_path=args.save)
    plot_heatmap(cross_sc, categories, save_path=args.save)


if __name__ == "__main__":
    main()
