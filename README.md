
# 📌 LiHAR: Lightweight Hierarchical Approximate Routing

LiHAR is an optimized hierarchical pathfinding algorithm for large road networks, built on **OpenStreetMap (OSM)** data.  
It improves query speed compared to **Dijkstra’s algorithm** while keeping paths close to optimal.

---

## ✨ Our Contributions (What’s New)

- **Hybrid backbone importance metric**: combines
  - Semantic (road type)  
  - Structural (degree centrality)  
  - Dynamic (sampled betweenness)

- **Precomputed backbone shortest paths**  
  Removes per-query Dijkstra overhead → queries become simple lookup + corridor search.

- **Corridor-restricted Dijkstra**  
  Expands only within a neighborhood around the backbone path → balances speed vs. accuracy.

- **Evaluation framework**  
  Measures runtime, speedup, path quality, and fallback rate.

👉 **Entirely new in this project:**
- Novel importance score formulation.  
- Precompute-at-preprocessing optimization for backbone paths.

---

## 📂 Project Structure

```

LiHAR/
├── data/                  # Graph and results storage
├── scripts/
│   ├── preprocess.py      # Build backbone & precompute paths
│   ├── query\_engine.py    # Run LiHAR queries
│   └── evaluate.py        # Benchmark LiHAR vs Dijkstra
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

````

---

## ⚡ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/saptarshimajumdar/LiHAR.git
cd LiHAR

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
````

---

## 🚀 Usage

### 1. Preprocess a city graph

```bash
python scripts/preprocess.py --place "Bengaluru, India" \
  --out data/bengaluru.graphml \
  --backbone_frac 0.005 \
  --sample_pairs 500
```

**Outputs:**

* `data/bengaluru.graphml` → full graph
* `data/bengaluru_backbone.gpickle` → backbone graph
* `data/bengaluru_backbone_paths.pkl` → precomputed paths

---

### 2. Run a single query

```bash
python scripts/query_engine.py \
  --graph data/bengaluru.graphml \
  --backbone data/bengaluru_backbone.gpickle \
  --s -1 --t -1 \
  --radius_hops 1
```

**Example Output:**

```
Picked random nodes: 7358582746 10094820680
Path length (nodes): 206
Path: [7358582746, ..., 10094820680]
```

---

### 3. Benchmark evaluation

```bash
python -m scripts.evaluate \
  --graph data/bengaluru.graphml \
  --backbone data/bengaluru_backbone.gpickle \
  --queries 50 \
  --radius_hops 1
```

**Example Output:**

```
========== Evaluation Summary ==========
Total queries: 50
Fallbacks: 3
Average Dijkstra time: 0.002145 s
Average LiHAR time:    0.000841 s
Average speedup:       2.55x
Avg path length ratio (LiHAR/Dijkstra): 1.012
Results saved to data/bengaluru_backbone_eval.csv
========================================
```

---

## 📊 Results

* **Speedup:** LiHAR is **2–5× faster** than Dijkstra.
* **Accuracy:** Paths are within \~**1–2%** of exact shortest paths.
* **Fallbacks:** Rare; guarantees correctness when corridor fails.
* **Tradeoff:**

  * Smaller corridor → faster but less accurate.
  * Larger corridor → slower but closer to exact.

---

## 📖 References

* Dijkstra, E. W. (1959). *A note on two problems in connexion with graphs.*
* Sanders, P., & Schultes, D. (2005). *Highway hierarchies hasten exact shortest path queries.*
* Boeing, G. (2017). *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks.*

