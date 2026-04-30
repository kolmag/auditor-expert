"""
tsne_viz.py — t-SNE embedding space visualisation

Run after every re-ingest to check for category bleeding.
Produces both 2D and 3D plots coloured by doc_category.

Usage:
    uv run scripts/diagnostics/tsne_viz.py
    uv run scripts/diagnostics/tsne_viz.py --3d
    uv run scripts/diagnostics/tsne_viz.py --save tsne_output.html
"""

import argparse
import os
import sys

import chromadb
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
from sklearn.manifold import TSNE

load_dotenv()

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
COLLECTION_NAME = "auditor_expert"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

CATEGORY_COLORS = {
    "standard":  "#2563eb",   # blue
    "procedure": "#16a34a",   # green
    "example":   "#dc2626",   # red
    "reference": "#9333ea",   # purple
    "general":   "#ea580c",   # orange
}


def load_embeddings() -> tuple[np.ndarray, list[dict]]:
    """Load all embeddings and metadata from ChromaDB."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn
        )
    except Exception:
        print(f"✗ Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        sys.exit(1)

    print(f"✓ Collection loaded: {collection.count()} chunks")

    results = collection.get(include=["embeddings", "metadatas"])
    embeddings = np.array(results["embeddings"])
    metadatas = results["metadatas"]
    return embeddings, metadatas


def run_tsne(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Run t-SNE dimensionality reduction."""
    print(f"Running t-SNE ({n_components}D) on {len(embeddings)} chunks...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(30, len(embeddings) // 4),
        random_state=42,
        max_iter=1000,
        verbose=1
    )
    return tsne.fit_transform(embeddings)


def plot_2d(coords: np.ndarray, metadatas: list[dict], save_path: str = None):
    """2D t-SNE plot coloured by doc_category."""
    categories = [m.get("doc_category", "unknown") for m in metadatas]
    sources = [m.get("source", "") for m in metadatas]
    headlines = [m.get("headline", "")[:60] for m in metadatas]

    fig = px.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        color=categories,
        color_discrete_map=CATEGORY_COLORS,
        hover_data={"source": sources, "headline": headlines},
        title="Auditor Expert — t-SNE Embedding Space (2D)",
        labels={"x": "t-SNE dim 1", "y": "t-SNE dim 2", "color": "Category"},
        width=900, height=650
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(legend_title_text="doc_category")

    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    else:
        fig.show()


def plot_3d(coords: np.ndarray, metadatas: list[dict], save_path: str = None):
    """3D t-SNE plot coloured by doc_category."""
    categories = [m.get("doc_category", "unknown") for m in metadatas]
    sources = [m.get("source", "") for m in metadatas]

    fig = px.scatter_3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        color=categories,
        color_discrete_map=CATEGORY_COLORS,
        hover_data={"source": sources},
        title="Auditor Expert — t-SNE Embedding Space (3D)",
        labels={"color": "Category"},
        width=900, height=700
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))

    if save_path:
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")
    else:
        fig.show()


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualisation for Auditor Expert")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="Generate 3D plot instead of 2D")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to HTML file instead of displaying")
    args = parser.parse_args()

    embeddings, metadatas = load_embeddings()

    # Category distribution summary
    from collections import Counter
    cat_counts = Counter(m.get("doc_category", "unknown") for m in metadatas)
    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat:<12} {count:>4} chunks")
    print()

    n_components = 3 if args.three_d else 2
    coords = run_tsne(embeddings, n_components=n_components)

    if args.three_d:
        plot_3d(coords, metadatas, save_path=args.save)
    else:
        plot_2d(coords, metadatas, save_path=args.save)


if __name__ == "__main__":
    main()
