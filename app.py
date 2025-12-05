from flask import Flask, request, jsonify, render_template
import pandas as pd
import networkx as nx
import os
app = Flask(__name__)

# =========================================================
# 1. LOAD NODES
# =========================================================
nodes_df = pd.read_csv("nodes.csv")
nodes_df.columns = nodes_df.columns.str.strip()

nodes_df["Name"] = nodes_df["Name"].astype(str).str.strip()
nodes_df["Type"] = nodes_df["Type"].astype(str).str.lower().str.strip()

civilian_nodes = nodes_df[nodes_df["Type"] == "civilians"]["Name"].tolist()
destination_nodes = nodes_df[nodes_df["Type"] == "destination"]["Name"].tolist()

# =========================================================
# 2. LOAD & CLEAN EDGES
# =========================================================
edges_df = pd.read_csv("edges.csv")
edges_df = edges_df.loc[:, ~edges_df.columns.str.contains("^unnamed", case=False, regex=True)]
edges_df.columns = edges_df.columns.str.strip().str.lower()

# Map scenario weights
from_col = "from" if "from" in edges_df.columns else None
to_col = "to" if "to" in edges_df.columns else None
risk_col = aid_col = dist_col = None
for c in edges_df.columns:
    if "danger scenario weights" in c:
        risk_col = c
    if "humanitarian scenario weights" in c:
        aid_col = c
    if "distance scenario weights" in c:
        dist_col = c

if not from_col or not to_col or not risk_col or not aid_col or not dist_col:
    raise ValueError(f"❌ edges.csv missing required columns. Found columns: {edges_df.columns.tolist()}")

edges_df = edges_df.rename(columns={
    from_col: "from",
    to_col: "to",
    risk_col: "risk",
    aid_col: "aid",
    dist_col: "dist"
})

for col in ["risk", "aid", "dist"]:
    edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce")

edges_df["from"] = edges_df["from"].astype(str).str.strip()
edges_df["to"]   = edges_df["to"].astype(str).str.strip()

# =========================================================
# 3. BUILD GRAPH
# =========================================================
GRAPH = {}
for _, row in edges_df.iterrows():
    u = row["from"]
    v = row["to"]
    if u not in GRAPH:
        GRAPH[u] = {}
    GRAPH[u][v] = {
        "risk": row["risk"],
        "aid": row["aid"],
        "dist": row["dist"],
    }


# =========================================================
# 4. DIJKSTRA
# =========================================================
def custom_dijkstra(source, weight_type, absorb_at_dest=True):
    V = set(GRAPH.keys())
    for u in GRAPH:
        for v in GRAPH[u]:
            V.add(v)
    lam = {node: float("inf") for node in V}
    lam[source] = 0.0
    pred = {}
    visited = set()

    while len(visited) < len(V):
        current = None
        best_val = float("inf")
        for n in V:
            if n not in visited and lam[n] < best_val:
                current = n
                best_val = lam[n]
        if current is None:
            break
        visited.add(current)

        if absorb_at_dest and current in destination_nodes:
            continue

        for neigh in GRAPH.get(current, {}):
            w = GRAPH[current][neigh][weight_type]
            new_cost = lam[current] + w
            if new_cost < lam[neigh]:
                lam[neigh] = new_cost
                pred[neigh] = current
    return lam, pred

# =========================================================
# 5. BELLMAN-FORD
# =========================================================
def custom_bellman_ford(source, weight_type, absorb_at_dest=True):
    V = set(GRAPH.keys())
    for u in GRAPH:
        for v in GRAPH[u]:
            V.add(v)
    lam = {node: float("inf") for node in V}
    lam[source] = 0.0
    pred = {}

    edges = []
    for u in GRAPH:
        if absorb_at_dest and u in destination_nodes:
            continue
        for v in GRAPH[u]:
            edges.append((u, v))

    for _ in range(len(V)-1):
        updated = False
        for u, v in edges:
            w = GRAPH[u][v][weight_type]
            if lam[u] + w < lam[v]:
                lam[v] = lam[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break
    return lam, pred

# =========================================================
# 6. RECONSTRUCT PATH
# =========================================================
def reconstruct_path(pred, target):
    path = [target]
    while target in pred:
        target = pred[target]
        path.append(target)
    return list(reversed(path))

# =========================================================
# 7. COMPUTE TOP-3 DESTINATIONS
# =========================================================
def compute_best_three(source, scenario):
    if not GRAPH:
        return []

    # --------------------------------------------------
    # 1. Determine reachable nodes from this source
    # --------------------------------------------------
    reachable = set([source])
    frontier = [source]
    while frontier:
        u = frontier.pop()
        for v in GRAPH.get(u, {}):
            if v not in reachable:
                reachable.add(v)
                frontier.append(v)

    # --------------------------------------------------
    # 2. Gather reachable edge weights for this scenario
    # --------------------------------------------------
    edge_weights = []
    for u in reachable:
        for v in GRAPH.get(u, {}):
            if v in reachable:
                edge_weights.append(GRAPH[u][v][scenario])

    # --------------------------------------------------
    # 3. Choose algorithm dynamically
    # --------------------------------------------------
    use_bellman = any(w < 0 for w in edge_weights)
    absorb = True if scenario != "distance" else False  # distance scenario should not absorb at destination

    if use_bellman:
        lam, pred = custom_bellman_ford(source, scenario, absorb_at_dest=absorb)
        algo = "Bellman-Ford"
    else:
        lam, pred = custom_dijkstra(source, scenario, absorb_at_dest=absorb)
        algo = "Dijkstra"

    # --------------------------------------------------
    # 4. Collect top 3 destinations
    # --------------------------------------------------
    results = []
    for dest in destination_nodes:
        if dest in lam and lam[dest] < float("inf"):
            results.append({
                "destination": dest,
                "cost": float(lam[dest]),
                "path": reconstruct_path(pred, dest),
                "algorithm": algo,
                "fallback": False
            })

    results.sort(key=lambda x: x["cost"])
    return results[:3]


# =========================================================
# 8. FLASK ENDPOINTS
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_nodes")
def get_nodes():
    return jsonify(civilian_nodes)

@app.route("/compute_best_paths", methods=["POST"])
def compute_best_paths():
    data = request.get_json() or {}
    src = data.get("source")
    scenario = data.get("scenario")

    if src not in civilian_nodes:
        return jsonify({"error": f"Unknown source '{src}'"}), 400
    if scenario not in {"risk", "aid", "distance", "dist"}:
        return jsonify({"error": "Scenario must be one of: 'risk', 'aid', 'distance'"}), 400

    scenario = "dist" if scenario == "distance" else scenario
    result = compute_best_three(src, scenario)
    if not result:
        return jsonify({"error": "No reachable destination for this source under the given scenario."}), 200
    return jsonify(result)

# =========================================================
# 7. RUN APP
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
