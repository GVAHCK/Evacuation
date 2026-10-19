"""Build Unseen Test Road Networks for Cross-Region Generalization Study:
Region 1: Dense Urban Grid (George Town / Parrys) - orthogonal grid, high intersection density.
Region 2: Sparse Suburban Corridor (OMR / Sholinganallur) - linear spine, sparse bypasses.
Zero data leakage: these regions are strictly unseen and held-out for test evaluation.
"""

import os
import sys
import json
import networkx as nx
import osmnx as ox

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("BuildGeneralizationRegions")

REGIONS = {
    "dense_grid": {
        "name": "George Town (Dense Urban Grid)",
        "bbox": (80.280, 13.085, 80.295, 13.100),
        "desc": "Orthogonal rectilinear downtown grid, high junction density, short link lengths, abundant parallel corridors.",
        "out_json": "data/processed/road_graph_dense_grid.json"
    },
    "sparse_suburban": {
        "name": "OMR Suburban (Sparse Linear Corridor)",
        "bbox": (80.218, 12.875, 80.235, 12.895),
        "desc": "Sparse suburban network dominated by a central north-south arterial spine with dendritic branch roads and limited bypasses.",
        "out_json": "data/processed/road_graph_sparse_suburban.json"
    }
}


def process_osm_to_clean_scc(bbox, region_key: str):
    logger.info(f"Downloading OSM network for {region_key}: bbox={bbox}...")
    G_raw = ox.graph_from_bbox(bbox=bbox, network_type="drive")
    logger.info(f"Raw download for {region_key}: {len(G_raw.nodes)} nodes, {len(G_raw.edges)} edges")

    di_graph = nx.DiGraph()

    for node_id, data in G_raw.nodes(data=True):
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        # Elevation estimate
        elev = max(2.0, min(20.0, (y - 12.8) * 50.0 + 4.0))
        di_graph.add_node(
            str(node_id),
            id=str(node_id),
            x=x,
            y=y,
            lon=x,
            lat=y,
            elevation=round(elev, 2)
        )

    for u, v, k, data in G_raw.edges(keys=True, data=True):
        u_str, v_str = str(u), str(v)
        length = float(data.get("length", 50.0))
        maxspeed = data.get("maxspeed", 13.89)
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
        try:
            speed_val = float(str(maxspeed).replace("km/h", "").strip()) / 3.6
        except (ValueError, TypeError):
            speed_val = 13.89

        lanes = data.get("lanes", 1)
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            lane_count = int(lanes)
        except (ValueError, TypeError):
            lane_count = 1

        oneway = data.get("oneway", False)
        if isinstance(oneway, list):
            oneway = oneway[0]
        is_oneway = oneway in (True, "yes", "1")

        edge_attrs = dict(
            id=f"{u_str}_{v_str}",
            length=max(length, 10.0),
            lanes=max(lane_count, 1),
            speed_limit=max(speed_val, 5.0),
            free_flow_time=max(length, 10.0) / max(speed_val, 5.0),
            hazard_intensity=0.0,
            hazard_state=0
        )
        di_graph.add_edge(u_str, v_str, **edge_attrs)

        # Restore two-way street connectivity
        if not is_oneway and not di_graph.has_edge(v_str, u_str):
            rev_attrs = dict(edge_attrs)
            rev_attrs["id"] = f"{v_str}_{u_str}"
            di_graph.add_edge(v_str, u_str, **rev_attrs)

    # Extract largest SCC
    sccs = list(nx.strongly_connected_components(di_graph))
    largest_scc = max(sccs, key=len)
    proc_G = di_graph.subgraph(largest_scc).copy()

    n_nodes = proc_G.number_of_nodes()
    n_edges = proc_G.number_of_edges()
    tot_length_km = sum(d.get("length", 50.0) for u, v, d in proc_G.edges(data=True)) / 1000.0
    avg_edge_len = (tot_length_km * 1000.0) / max(n_edges, 1)
    density = nx.density(proc_G)
    avg_out_deg = sum(d for n, d in proc_G.out_degree()) / n_nodes
    dead_ends = sum(1 for n, d in proc_G.out_degree() if d == 0)

    # Save to JSON
    out_json = REGIONS[region_key]["out_json"]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    graph_data = {
        "nodes": {n: d for n, d in proc_G.nodes(data=True)},
        "edges": [{"source": u, "target": v, **d} for u, v, d in proc_G.edges(data=True)]
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    sz_kb = os.path.getsize(out_json) / 1024.0

    stats = {
        "region_key": region_key,
        "name": REGIONS[region_key]["name"],
        "description": REGIONS[region_key]["desc"],
        "nodes": n_nodes,
        "edges": n_edges,
        "total_length_km": round(tot_length_km, 2),
        "avg_edge_length_m": round(avg_edge_len, 1),
        "density": round(density, 6),
        "avg_out_degree": round(avg_out_deg, 2),
        "dead_ends": dead_ends,
        "is_strongly_connected": nx.is_strongly_connected(proc_G),
        "file_size_kb": round(sz_kb, 1),
        "json_path": out_json
    }

    logger.info(f"Successfully processed {region_key}: {n_nodes} nodes, {n_edges} edges, {tot_length_km:.2f} km, {sz_kb:.1f} KB")
    return stats


def main():
    logger.info("=== STARTING PHASE A: SELECT & BUILD UNSEEN TEST REGIONS ===")
    all_stats = {}
    for r_key, r_info in REGIONS.items():
        st = process_osm_to_clean_scc(r_info["bbox"], r_key)
        all_stats[r_key] = st

    out_summary = "results/tables/generalization_regions_topology.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Saved region topology summary to {out_summary}")

    print("\n" + "=" * 90)
    print("PHASE A: UNSEEN GENERALIZATION REGION TOPOLOGY AUDIT")
    print("=" * 90)
    for k, v in all_stats.items():
        print(f"Region: {v['name']} ({k})")
        print(f"  Nodes: {v['nodes']} | Edges: {v['edges']} | Road Length: {v['total_length_km']} km")
        print(f"  Avg Edge Length: {v['avg_edge_length_m']} m | Density: {v['density']} | Avg Out-Degree: {v['avg_out_degree']}")
        print(f"  Single SCC: {v['is_strongly_connected']} | Dead-Ends: {v['dead_ends']} | File Size: {v['file_size_kb']} KB")
        print(f"  Topology Archetype: {v['description']}\n")


if __name__ == "__main__":
    main()
