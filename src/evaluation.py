"""
Simple test & evaluation utilities:
- run_queries: executes multiple queries, returns (results_df, summary_df, latencies_ms)
- plot_latency: matplotlib plot of latencies
"""
from __future__ import annotations
from typing import Iterable, Tuple, List
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .embedding import Embedder
from .vibe_matcher import rank_top_k


def run_queries(
    queries: Iterable[str],
    df: pd.DataFrame,
    embedder: Embedder,
    top_k: int = 3,
    good_threshold: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    records = []
    latencies = []

    for q in queries:
        t0 = time.perf_counter()
        topk_df, fallback = rank_top_k(
            q,
            df,
            embedder,
            top_k=top_k,
            good_threshold=good_threshold
        )
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        tmp = topk_df.copy()
        tmp["query"] = q
        tmp["fallback"] = fallback if fallback else ""
        tmp["good"] = (tmp["similarity"] >= good_threshold).astype(int)
        records.append(tmp)

    results_df = pd.concat(records, ignore_index=True)

    summary_df = (
        results_df.groupby("query")
        .agg(
            avg_sim=("similarity", "mean"),
            max_sim=("similarity", "max"),
            top_hit_good=("good", "max"),
        )
        .reset_index()
    )

    return results_df, summary_df, latencies


def plot_latency(latencies_ms: List[float]) -> None:
    plt.figure()
    plt.plot(range(1, len(latencies_ms) + 1), latencies_ms, marker="o")
    plt.title("Query Latency (ms)")
    plt.xlabel("Query #")
    plt.ylabel("Latency (ms)")
    plt.tight_layout()
    plt.show()
