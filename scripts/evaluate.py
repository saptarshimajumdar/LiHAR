import argparse
import os
import random
import time
import pickle
import networkx as nx
import osmnx as ox
import pandas as pd
from tqdm import tqdm
from query_engine import query_lihar


def sample_queries(G, n=1000):
    """Sample n random node pairs from graph."""
    nodes = list(G.nodes())
    pairs = []
    tries = 0
    while len(pairs) < n and tries < n * 20:
        s, t = random.sample(nodes, 2)
        pairs.append((s, t))
        tries += 1
    return pairs


def run_benchmark(graphfile, backbonefile, queries=500, radius_hops=1):
    """Benchmark LiHAR vs baseline Dijkstra."""
    G = ox.load_graphml(graphfile)

    # Load precomputed backbone paths
    with open(backbonefile.replace("_backbone.gpickle", "_backbone_paths.pkl"), "rb") as f:
        backbone_data = pickle.load(f)
    backbone_paths = backbone_data["paths"]
    backbone_lengths = backbone_data["lengths"]

    qs = sample_queries(G, n=queries)
    results = []
    fallback_count = 0

    for s, t in tqdm(qs, desc="Running queries"):
        # Baseline Dijkstra
        t0 = time.time()
        try:
            p0 = nx.shortest_path(G, s, t, weight="length")
            dt0 = time.time() - t0
            dist0 = nx.shortest_path_length(G, s, t, weight="length")
        except Exception:
            continue

        # LiHAR
        t1 = time.time()
        try:
            p1 = query_lihar(G, backbone_paths, backbone_lengths, s, t, radius_hops=radius_hops)
            dt1 = time.time() - t1
            dist1 = sum(G[u][v][0].get("length", 0.0) for u, v in zip(p1[:-1], p1[1:]))
        except Exception:
            fallback_count += 1
            continue

        results.append((s, t, dt0, dt1, dist0, dist1))

    return results, fallback_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--radius_hops", type=int, default=1)
    args = parser.parse_args()

    res, fallback_count = run_benchmark(
        args.graph, args.backbone, queries=args.queries, radius_hops=args.radius_hops
    )

    df = pd.DataFrame(
        res,
        columns=["s", "t", "dijkstra_time", "lihar_time", "dijkstra_len", "lihar_len"],
    )
    out_csv = args.backbone.replace(".gpickle", "_eval.csv")
    df.to_csv(out_csv, index=False)

    # ---- Summary stats ----
    avg_dj_time = df["dijkstra_time"].mean()
    avg_lihar_time = df["lihar_time"].mean()
    avg_speedup = (df["dijkstra_time"] / df["lihar_time"]).mean()
    avg_path_ratio = (df["lihar_len"] / df["dijkstra_len"]).mean()

    print("\n========== Evaluation Summary ==========")
    print(f"Total queries: {len(df)}")
    print(f"Fallbacks: {fallback_count}")
    print(f"Average Dijkstra time: {avg_dj_time:.6f} s")
    print(f"Average LiHAR time:    {avg_lihar_time:.6f} s")
    print(f"Average speedup:       {avg_speedup:.2f}x")
    print(f"Avg path length ratio (LiHAR/Dijkstra): {avg_path_ratio:.3f}")
    print("Results saved to", out_csv)
    print("========================================\n")


if __name__ == "__main__":
    main()
