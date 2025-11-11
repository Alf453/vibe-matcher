from src.embedding import Embedder
from src.data_loader import load_products_json
from src.vibe_matcher import rank_top_k
from src.evaluation import run_queries, plot_latency

# Load product data
df = load_products_json("data/sample_products.json")

# Initialize embedder (set use_openai=True if you want real embeddings)
embedder = Embedder(use_openai=False)

# Run a single query
query = "energetic urban chic"
results, fallback = rank_top_k(query, df, embedder)

print("===Results===")
print(results)
print("Fallback:", fallback)

# Run evaluation over multiple queries
queries = [
    "energetic urban chic",
    "cozy cabin weekend",
    "beachy minimal summer"
]

res_df, summary_df, latencies = run_queries(queries, df, embedder)

print("\n===Summary===")
print(summary_df)

plot_latency(latencies)
