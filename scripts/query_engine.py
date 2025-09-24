import argparse
import networkx as nx
import osmnx as ox
import pickle
import random
import math
from collections import deque


# ============================
# Corridor build
# ============================
def build_corridor(G, backbone_path, radius_hops=1):
    """Build corridor: union of k-hop neighborhoods around backbone path nodes."""
    corridor = set()
    for u in backbone_path:
        q = deque([(u, 0)])
        seen = {u}
        while q:
            v, d = q.popleft()
            corridor.add(v)
            if d < radius_hops:
                for w in G.neighbors(v):
                    if w in seen:
                        continue
                    seen.add(w)
                    q.append((w, d + 1))
    return corridor


# ============================
# Restricted Dijkstra
# ============================
def dijkstra_restricted(G, corridor, s, t):
    """Dijkstra restricted to nodes in corridor set. Returns path or None."""
    import heapq
    dist = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    INF = math.inf

    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > dist.get(u, INF):
            continue
        for v, nbdata in G[u].items():
            if v not in corridor:
                continue
            if isinstance(nbdata, dict):  # MultiDiGraph
                first = next(iter(nbdata.values()))
                w = first.get("length", 1.0)
            else:
                w = nbdata.get("length", 1.0)
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if t not in dist:
        return None

    # reconstruct path
    path = []
    cur = t
    while True:
        path.append(cur)
        if cur == s:
            break
        cur = prev[cur]
    path.reverse()
    return path


# ============================
# Backbone lookup (precomputed paths)
# ============================
def backbone_lookup(backbone_paths, backbone_lengths, sources, targets):
    """Fast lookup for best path between any source-target portal pair."""
    best = None
    for s in sources:
        if s not in backbone_paths:
            continue
        for t in targets:
            if t not in backbone_paths[s]:
                continue
            path = backbone_paths[s][t]
            dist = backbone_lengths[s][t]
            if best is None or dist < best[0]:
                best = (dist, path)
    return None if best is None else best[1]


# ============================
# Main LiHAR query
# ============================
def query_lihar(G, backbone_paths, backbone_lengths, s, t, radius_hops=1):
    """Run LiHAR using precomputed backbone paths. Falls back to Dijkstra if needed."""
    if s == t:
        return [s]

    backbone_nodes = set(backbone_paths.keys())

    # portals: nearest backbone nodes (simple fallback: pick any if missing)
    def nearest_backbone_nodes(node, k=3):
        if node in backbone_nodes:
            return [node]
        # fallback: pick random k backbone nodes (fast stub)
        return random.sample(list(backbone_nodes), min(k, len(backbone_nodes)))

    s_portals = nearest_backbone_nodes(s)
    t_portals = nearest_backbone_nodes(t)

    # coarse path on backbone
    pb = backbone_lookup(backbone_paths, backbone_lengths, s_portals, t_portals)
    if pb is None:
        return nx.shortest_path(G, s, t, weight="length")

    # build corridor + restricted Dijkstra
    corridor = build_corridor(G, pb, radius_hops=radius_hops)
    p = dijkstra_restricted(G, corridor, s, t)
    if p is None:
        return nx.shortest_path(G, s, t, weight="length")
    return p


# ============================
# CLI entry
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, help="full graph .graphml")
    parser.add_argument("--backbone", required=True, help="backbone .gpickle")
    parser.add_argument("--s", type=int, required=True)
    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--radius_hops", type=int, default=1)
    args = parser.parse_args()

    # Load graph
    G = ox.load_graphml(args.graph)

    # Load precomputed backbone paths
    with open(args.backbone.replace("_backbone.gpickle", "_backbone_paths.pkl"), "rb") as f:
        backbone_data = pickle.load(f)
    backbone_paths = backbone_data["paths"]
    backbone_lengths = backbone_data["lengths"]

    # Random nodes if needed
    if args.s < 0 or args.t < 0:
        args.s, args.t = random.sample(list(G.nodes()), 2)
        print("Picked random nodes:", args.s, args.t)

    path = query_lihar(G, backbone_paths, backbone_lengths, args.s, args.t,
                       radius_hops=args.radius_hops)

    print("Path length (nodes):", len(path))
    print("Path:", path)
