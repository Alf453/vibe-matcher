"""
Core retrieval logic: cosine similarity ranking + friendly fallback.
Assumes the input DataFrame has at least columns: ['name', 'desc', 'vibes'].
"""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .embedding import Embedder


def rank_top_k(
    query: str,
    df: pd.DataFrame,
    embedder: Embedder,
    top_k: int = 3,
    good_threshold: float = 0.7,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Returns a DataFrame of the top-k items with a 'similarity' column
    and an optional fallback message.
    """
    if "desc" not in df.columns:
        raise ValueError("DataFrame must contain a 'desc' column.")

    # Precompute product embeddings
    product_embs = embedder.embed_texts(df["desc"].tolist())
    q_emb = embedder.embed_texts([query])[0].reshape(1, -1)

    # Compute cosine similarity
    sims = cosine_similarity(q_emb, product_embs)[0]
    idx = np.argsort(-sims)[:top_k]

    # Prepare output
    out = df.iloc[idx].copy().reset_index(drop=True)
    out["similarity"] = sims[idx]

    # Fallback message if no good match
    fallback = None
    if float(out["similarity"].max()) < good_threshold:
        fallback = (
            "No strong match. Try refining with vibe tags like 'boho', 'cozy', 'urban', "
            "'athleisure', 'beachy', 'chic', or combine mood + setting "
            "(e.g., 'cozy urban coffee')."
        )

    return out, fallback
