import argparse
import os
import pickle
import osmnx as ox
import networkx as nx
from tqdm import tqdm
import random
import numpy as np

def compute_importance(G, sample_pairs=500, alpha=0.6, beta=0.2, gamma=0.2):
    """Compute node importance from degree, road type, and sampled betweenness."""
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    # degree score
    degs = np.array([G.degree(v) for v in nodes], dtype=float)
    deg_score = (degs - degs.min()) / (degs.max() - degs.min() + 1e-9)

    # road type score
    rank = {'motorway':5,'trunk':4,'primary':3,'secondary':2,
            'tertiary':1,'unclassified':0,'residential':0}
    road_score = np.zeros(n, dtype=float)
    for i, v in enumerate(nodes):
        maxr = 0
        for _, _, edata in G.edges(v, data=True):
            h = edata.get('highway', None)
            if h is None: continue
            if isinstance(h, list): h = h[0]
            maxr = max(maxr, rank.get(h, 0))
        road_score[i] = maxr
    if road_score.max() > 0:
        road_score /= (road_score.max() + 1e-9)

    # sampled centrality
    sample_counts = np.zeros(n, dtype=float)
    rng = random.Random(42)
    for _ in tqdm(range(sample_pairs), desc="sampling paths"):
        s = rng.choice(nodes)
        t = rng.choice(nodes)
        try:
            path = nx.shortest_path(G, s, t, weight="length")
            for u in path:
                sample_counts[node_to_idx[u]] += 1
        except Exception:
            continue
    if sample_counts.max() > 0:
        sample_counts /= sample_counts.max()

    # aggregate
    imp = alpha*road_score + beta*deg_score + gamma*sample_counts
    return {v: float(imp[node_to_idx[v]]) for v in nodes}


def build_backbone(G, imp_dict, backbone_frac=0.01):
    """Pick top-k nodes and build reduced backbone graph."""
    nodes = list(G.nodes())
    k = max(1, int(len(nodes) * backbone_frac))
    top_nodes = sorted(imp_dict.items(), key=lambda x: -x[1])[:k]
    backbone_nodes = set(v for v, _ in top_nodes)

    B = nx.DiGraph()
    for v in backbone_nodes:
        B.add_node(v)

    cutoff = 5000  # meters
    for u in tqdm(backbone_nodes, desc="building backbone"):
        lengths, _ = nx.single_source_dijkstra(G, u, cutoff=cutoff, weight="length")
        for v, dist in lengths.items():
            if v in backbone_nodes and v != u:
                B.add_edge(u, v, length=float(dist))
    return B


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--place', type=str, help='Place name for OSMnx')
    parser.add_argument('--infile', type=str, help='Existing .graphml file')
    parser.add_argument('--out', type=str, default='data/graph.graphml')
    parser.add_argument('--backbone_frac', type=float, default=0.01)
    parser.add_argument('--sample_pairs', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.infile:
        G = ox.load_graphml(args.infile)
    else:
        print("Downloading graph via OSMnx...")
        G = ox.graph_from_place(args.place, network_type='drive')
        G = ox.distance.add_edge_lengths(G)
        ox.save_graphml(G, args.out)

    print("Computing importance...")
    imp = compute_importance(G, sample_pairs=args.sample_pairs)

    print("Building backbone...")
    B = build_backbone(G, imp, backbone_frac=args.backbone_frac)

    # Save artifacts
    with open(args.out.replace('.graphml','_imp.pkl'), 'wb') as f:
        pickle.dump(imp, f)
    with open(args.out.replace('.graphml','_backbone.gpickle'), 'wb') as f:
        pickle.dump(B, f)

    # NEW: save all-pairs shortest paths on backbone
    print("Precomputing backbone all-pairs shortest paths...")
    all_pairs = dict(nx.all_pairs_dijkstra_path(B, weight="length"))
    all_pairs_len = dict(nx.all_pairs_dijkstra_path_length(B, weight="length"))
    with open(args.out.replace('.graphml','_backbone_paths.pkl'), 'wb') as f:
        pickle.dump({"paths": all_pairs, "lengths": all_pairs_len}, f)

    print("Preprocessing finished. Saved:", args.out)


if __name__ == "__main__":
    main()
