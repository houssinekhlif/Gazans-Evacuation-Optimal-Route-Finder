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
nodes_df["Type"] = nodes_df["Type"].str.lower().str.strip()
nodes_df["Name"] = nodes_df["Name"].astype(str).str.strip()

# Civilian & Destination nodes
civilian_nodes = nodes_df[nodes_df["Type"] == "civilians"]["Name"].tolist()
destination_nodes = nodes_df[nodes_df["Type"] == "destination"]["Name"].tolist()

# =========================================================
# 2. LOAD EDGES
# =========================================================
edges_df = pd.read_csv("edges.csv")
edges_df = edges_df.loc[:, ~edges_df.columns.str.contains("Unnamed")]

weight_cols = ["Humanitarian Scenario Weights", "Distance Scenario Weights", "Danger Scenario Weights"]

for col in weight_cols:
    edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce").fillna(0)

edges_df["From"] = edges_df["From"].astype(str).str.strip()
edges_df["To"] = edges_df["To"].astype(str).str.strip()

# =========================================================
# 3. BUILD GRAPH
# =========================================================
G = nx.DiGraph()

for _, row in edges_df.iterrows():
    for u, v in [(row["From"], row["To"]), (row["To"], row["From"])]:
        G.add_edge(
            u, v,
            risk=row["Danger Scenario Weights"],
            aid=row["Humanitarian Scenario Weights"],
            dist=row["Distance Scenario Weights"]
        )


# =========================================================
# 4. SELECT ALGORITHM BASED ON WEIGHTS
# =========================================================
def choose_algorithm(weight_type):
    """Returns 'dijkstra' OR 'bellman-ford' based on negative weights."""
    values = [data[weight_type] for _, _, data in G.edges(data=True)]
    return "bellman-ford" if min(values) < 0 else "dijkstra"


# =========================================================
# 5. COMPUTE BEST PATH
# =========================================================
def compute_paths_from_source(src, weight_type):
    algorithm = choose_algorithm(weight_type)
    results = []

    for dst in destination_nodes:
        if src == dst:
            continue
        try:
            if algorithm == "dijkstra":
                path = nx.dijkstra_path(G, src, dst, weight=weight_type)
                cost = nx.dijkstra_path_length(G, src, dst, weight=weight_type)

            else:  # Bellman-Ford
                path = nx.bellman_ford_path(G, src, dst, weight=weight_type)
                cost = nx.bellman_ford_path_length(G, src, dst, weight=weight_type)

            results.append({
                "destination": dst,
                "path": path,
                "cost": cost,
                "algorithm": algorithm
            })

        except Exception:
            continue

    results = sorted(results, key=lambda x: x["cost"])
    return results[:3]  # Return TOP 3


# =========================================================
# 6. API ENDPOINTS
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
    scenario = data.get("scenario")  # "risk", "aid", "distance"

    weight_map = {
        "risk": "risk",
        "aid": "aid",
        "distance": "dist"
    }

    weight_type = weight_map[scenario]

    results = compute_paths_from_source(src, weight_type)
    return jsonify(results)


# =========================================================
# 7. RUN APP
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
