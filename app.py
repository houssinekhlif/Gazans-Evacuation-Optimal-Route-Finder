from flask import Flask, request, jsonify, render_template
import pandas as pd
import networkx as nx

app = Flask(__name__)

# =========================================================
# 1. LOAD NODES (AUTO-DETECT COLUMNS)
# =========================================================
nodes_df = pd.read_csv("nodes.csv")
nodes_df.columns = nodes_df.columns.str.strip()

# Auto-detect name column
name_candidates = ["Name", "name", "Node", "node", "Label", "label"]
name_col = next((c for c in name_candidates if c in nodes_df.columns), None)
if not name_col:
    raise ValueError("ERROR: Could not find a node NAME column in nodes.csv")

# Auto-detect type column
type_candidates = ["Type", "type", "Category", "category"]
type_col = next((c for c in type_candidates if c in nodes_df.columns), None)
if not type_col:
    raise ValueError("ERROR: Could not find a node TYPE column in nodes.csv")

# Clean fields
nodes_df[name_col] = nodes_df[name_col].astype(str).str.strip()
nodes_df[type_col] = nodes_df[type_col].astype(str).str.lower().str.strip()

# Lists for frontend
civilian_nodes = nodes_df[nodes_df[type_col] == "civilians"][name_col].tolist()
destination_nodes = nodes_df[nodes_df[type_col] == "destination"][name_col].tolist()

# =========================================================
# 2. LOAD EDGES
# =========================================================
edges_df = pd.read_csv("edges.csv")
edges_df = edges_df.loc[:, ~edges_df.columns.str.contains("Unnamed")]

weight_cols = [
    "Humanitarian Scenario Weights",
    "Distance Scenario Weights",
    "Danger Scenario Weights"
]

for col in weight_cols:
    if col not in edges_df.columns:
        raise ValueError(f"Missing edge column: {col}")
    edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce").fillna(0)

edges_df["From"] = edges_df["From"].astype(str).str.strip()
edges_df["To"] = edges_df["To"].astype(str).str.strip()

# =========================================================
# 3. BUILD GRAPH
# =========================================================
G = nx.DiGraph()
for _, row in edges_df.iterrows():
    u = row["From"]
    v = row["To"]
    G.add_edge(
        u, v,
        risk=row["Danger Scenario Weights"],
        aid=row["Humanitarian Scenario Weights"],
        dist=row["Distance Scenario Weights"]
    )
    G.add_edge(
        v, u,
        risk=row["Danger Scenario Weights"],
        aid=row["Humanitarian Scenario Weights"],
        dist=row["Distance Scenario Weights"]
    )

# =========================================================
# 4. CHOOSE ALGORITHM PER SOURCE COMPONENT
# =========================================================
def choose_algorithm_for_source(src, weight_type):
    # Find the weakly connected component containing the source
    for component in nx.weakly_connected_components(G):
        if src in component:
            subgraph = G.subgraph(component)
            break
    else:
        raise ValueError(f"Source node '{src}' is not in any connected component.")

    # If any edge has negative weight, use Bellman-Ford
    values = [d[weight_type] for _, _, d in subgraph.edges(data=True)]
    algorithm = "bellman-ford" if min(values) < 0 else "dijkstra"
    return algorithm, subgraph

# =========================================================
# 5. COMPUTE BEST PATHS
# =========================================================
def compute_paths_from_source(src, weight_type):
    try:
        algorithm, subgraph = choose_algorithm_for_source(src, weight_type)
    except ValueError:
        return []

    # Filter out other civilian nodes from the subgraph (only intersections and destinations as intermediates)
    allowed_nodes = set(destination_nodes + [src])  # source + all destinations
    for node in subgraph.nodes():
        if node not in allowed_nodes and node in civilian_nodes:
            subgraph = subgraph.copy()
            subgraph.remove_node(node)

    reachable_destinations = [dst for dst in destination_nodes if dst in subgraph]
    if not reachable_destinations:
        return []

    results = []
    for dst in reachable_destinations:
        try:
            if algorithm == "dijkstra":
                path = nx.dijkstra_path(subgraph, src, dst, weight=weight_type)
                cost = nx.dijkstra_path_length(subgraph, src, dst, weight=weight_type)
            else:
                path = nx.bellman_ford_path(subgraph, src, dst, weight=weight_type)
                cost = nx.bellman_ford_path_length(subgraph, src, dst, weight=weight_type)

            results.append({
                "destination": dst,
                "path": path,
                "cost": float(cost),
                "algorithm": algorithm
            })
        except nx.NetworkXNoPath:
            continue

    results.sort(key=lambda x: x["cost"])
    return results[:3]


# =========================================================
# 6. API ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_nodes")
def get_nodes():
    return jsonify(civilian_nodes)

@app.route("/compute", methods=["POST"])
def compute():
    data = request.json
    src = data.get("source")
    scenario = data.get("scenario")

    weight_map = {"risk": "risk", "aid": "aid", "distance": "dist"}
    if scenario not in weight_map:
        return jsonify({"error": "Invalid scenario selected"}), 400

    weight_type = weight_map[scenario]
    results = compute_paths_from_source(src, weight_type)

    if not results:
        return jsonify({"error": "No reachable destinations from this source for the selected scenario."})

    return jsonify(results)


# =========================================================
# 7. RUN APP
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
