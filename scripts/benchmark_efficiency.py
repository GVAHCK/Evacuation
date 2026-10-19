"""Benchmarking computational efficiency, per-step latency distribution (mean/median/p95/p99),
scalability across synthetic graph scales (1x, 2x, 4x, 8x), model parameter accounting,
and training cost accounting.
Generates results/tables/efficiency_profile.csv and results/figures/fig11_scalability_latency.png.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN

from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from src.baselines.astar import AStarBaseline
from src.baselines.random_routing import RandomValidRoutingBaseline


def benchmark_per_step_latency():
    """Measures fine-grained per-step inference latency across all steps in test scenarios."""
    print("\n--- 1. BENCHMARKING FINE-GRAINED PER-STEP INFERENCE LATENCY ---")
    test_scenarios = json.load(open("data/generated/test_scenarios.json"))
    env = EvacuationEnv(scenarios=test_scenarios, use_traci=False, curriculum_stage=4)

    ppo = MaskablePPO.load("models/ppo/seed_42/best_model.zip")
    dqn = DQN.load("models/dqn/seed_42/best_model.zip")
    dijkstra = DijkstraBaseline()
    astar = AStarBaseline()
    rand_agent = RandomValidRoutingBaseline()

    # Pre-generate observations and states from test scenarios
    # To isolate pure algorithm inference time from simulation step dynamics
    decision_states = []
    for scen in test_scenarios[:20]:  # Sample 20 representative scenarios
        obs, _ = env.reset(options={"scenario": scen})
        done = False
        while not done:
            mask = env.action_masks()
            decision_states.append((obs, mask, env.current_node, env.destination_node))
            act, _ = ppo.predict(obs, action_masks=mask, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(act))
            done = term or trunc

    print(f"Collected {len(decision_states)} unique decision states for latency profiling.")

    models = [
        ("MaskablePPO", ppo, "ppo"),
        ("Masked DQN", dqn, "dqn"),
        ("Dynamic Dijkstra", dijkstra, "dijkstra"),
        ("Dynamic A*", astar, "astar"),
        ("Random Valid", rand_agent, "random")
    ]

    latency_results = {}

    # Warmup runs
    for m_name, m_obj, m_type in models:
        for obs, mask, c_node, d_node in decision_states[:10]:
            if m_type == "ppo":
                _ = m_obj.predict(obs, action_masks=mask, deterministic=True)
            elif m_type == "dqn":
                obs_t = torch.as_tensor(obs, device=m_obj.device).unsqueeze(0)
                with torch.no_grad():
                    q = m_obj.q_net(obs_t).squeeze(0).cpu().numpy()
                q[~mask] = -1e9
                _ = np.argmax(q)
            else:
                _ = m_obj.select_action(env)

    # Timed runs across all decision states (repeated 5 times for statistical stability)
    N_REPEATS = 5
    for m_name, m_obj, m_type in models:
        latencies_ms = []
        for _ in range(N_REPEATS):
            for obs, mask, c_node, d_node in decision_states:
                t0 = time.perf_counter()
                if m_type == "ppo":
                    _ = m_obj.predict(obs, action_masks=mask, deterministic=True)
                elif m_type == "dqn":
                    obs_t = torch.as_tensor(obs, device=m_obj.device).unsqueeze(0)
                    with torch.no_grad():
                        q = m_obj.q_net(obs_t).squeeze(0).cpu().numpy()
                    q[~mask] = -1e9
                    _ = np.argmax(q)
                else:
                    _ = m_obj.select_action(env)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)

        lat_arr = np.array(latencies_ms)
        latency_results[m_name] = {
            "mean_ms": float(np.mean(lat_arr)),
            "median_ms": float(np.median(lat_arr)),
            "p95_ms": float(np.percentile(lat_arr, 95)),
            "p99_ms": float(np.percentile(lat_arr, 99)),
            "std_ms": float(np.std(lat_arr))
        }
        print(f"{m_name:18s} | Mean: {latency_results[m_name]['mean_ms']:.3f} ms | Median: {latency_results[m_name]['median_ms']:.3f} ms | p95: {latency_results[m_name]['p95_ms']:.3f} ms | p99: {latency_results[m_name]['p99_ms']:.3f} ms")

    env.close()
    return latency_results


def benchmark_scalability():
    """Characterizes how inference latency scales with road network size (1x, 2x, 4x, 8x nodes)."""
    print("\n--- 2. BENCHMARKING SCALABILITY ACROSS GRAPH SIZES (1x to 8x) ---")
    ppo = MaskablePPO.load("models/ppo/seed_42/best_model.zip")
    dqn = DQN.load("models/dqn/seed_42/best_model.zip")

    # Base graph stats: 84 nodes, 183 edges
    scale_factors = [1, 2, 4, 8]
    node_counts = [84, 168, 336, 672]
    
    scalability_data = []

    # Prepare synthetic road graphs of increasing size
    for scale, n_nodes in zip(scale_factors, node_counts):
        # Generate connected geometric graph mimicking urban street grid
        n_side = int(np.ceil(np.sqrt(n_nodes)))
        G = nx.grid_2d_graph(n_side, n_side, create_using=nx.DiGraph)
        # Relabel nodes as integers and retain exactly n_nodes
        nodes_subset = list(G.nodes)[:n_nodes]
        sub_G = G.subgraph(nodes_subset).copy()
        
        # Add bidirectional edge weights
        for u, v in sub_G.edges():
            sub_G[u][v]["weight"] = np.random.uniform(2.0, 15.0)

        source = list(sub_G.nodes)[0]
        target = list(sub_G.nodes)[-1]
        n_edges = sub_G.number_of_edges()

        # 1. Measure Dijkstra Latency on this graph
        dijk_times = []
        for _ in range(200):
            t0 = time.perf_counter()
            try:
                _ = nx.shortest_path(sub_G, source=source, target=target, weight="weight")
            except nx.NetworkXNoPath:
                pass
            dt_ms = (time.perf_counter() - t0) * 1000.0
            dijk_times.append(dt_ms)

        # 2. Measure A* Latency on this graph
        def heuristic(u, v):
            return np.hypot(u[0] - v[0], u[1] - v[1]) * 2.0

        astar_times = []
        for _ in range(200):
            t0 = time.perf_counter()
            try:
                _ = nx.astar_path(sub_G, source=source, target=target, heuristic=heuristic, weight="weight")
            except nx.NetworkXNoPath:
                pass
            dt_ms = (time.perf_counter() - t0) * 1000.0
            astar_times.append(dt_ms)

        # 3. Measure PPO Forward Pass Latency (34-dim input is constant O(1))
        dummy_obs = np.random.randn(34).astype(np.float32)
        dummy_mask = np.array([True, True, True, False])
        ppo_times = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = ppo.predict(dummy_obs, action_masks=dummy_mask, deterministic=True)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            ppo_times.append(dt_ms)

        # 4. Measure DQN Forward Pass Latency
        dqn_times = []
        obs_t = torch.as_tensor(dummy_obs, device=dqn.device).unsqueeze(0)
        for _ in range(200):
            t0 = time.perf_counter()
            with torch.no_grad():
                q = dqn.q_net(obs_t).squeeze(0).cpu().numpy()
            q[~dummy_mask] = -1e9
            _ = np.argmax(q)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            dqn_times.append(dt_ms)

        scalability_data.append({
            "scale": f"{scale}x",
            "nodes": n_nodes,
            "edges": n_edges,
            "PPO_ms": float(np.mean(ppo_times)),
            "DQN_ms": float(np.mean(dqn_times)),
            "Dijkstra_ms": float(np.mean(dijk_times)),
            "AStar_ms": float(np.mean(astar_times))
        })
        print(f"Scale {scale}x ({n_nodes:3d} nodes, {n_edges:4d} edges) -> Dijkstra: {np.mean(dijk_times):.3f} ms | A*: {np.mean(astar_times):.3f} ms | PPO: {np.mean(ppo_times):.3f} ms | DQN: {np.mean(dqn_times):.3f} ms")

    df_scale = pd.DataFrame(scalability_data)

    # Plot scalability curve
    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(df_scale["nodes"], df_scale["Dijkstra_ms"], marker="s", linewidth=2, color="#e74c3c", label="Dynamic Dijkstra $O(E + V \\log V)$")
    ax.plot(df_scale["nodes"], df_scale["AStar_ms"], marker="^", linewidth=2, color="#e67e22", label="Dynamic A* $O(E + V \\log V)$")
    ax.plot(df_scale["nodes"], df_scale["PPO_ms"], marker="o", linewidth=2, color="#2980b9", label="MaskablePPO $O(1)$ [Proposed]")
    ax.plot(df_scale["nodes"], df_scale["DQN_ms"], marker="d", linewidth=2, color="#27ae60", label="Masked DQN $O(1)$")

    ax.set_title("Inference Scalability: Decision Latency vs. Road Network Size", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Network Nodes (|V|)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Per-Decision Inference Latency (ms)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9)
    plt.tight_layout()
    fig11_path = "results/figures/fig11_scalability_latency.png"
    plt.savefig(fig11_path)
    plt.close()
    print(f"Saved: {fig11_path}")

    return df_scale


def compute_efficiency_profile(latency_res):
    """Assembles the final comprehensive efficiency profile table."""
    print("\n--- 3. ASSEMBLING EFFICIENCY PROFILE TABLE ---")

    ppo_path = "models/ppo/seed_42/best_model.zip"
    dqn_path = "models/dqn/seed_42/best_model.zip"
    graph_path = "data/processed/road_graph.json"

    # Model sizes
    ppo_sz_kb = round(os.path.getsize(ppo_path) / 1024.0, 1)
    dqn_sz_kb = round(os.path.getsize(dqn_path) / 1024.0, 1)
    graph_sz_kb = round(os.path.getsize(graph_path) / 1024.0, 1)

    # Parameter counts
    ppo_model = MaskablePPO.load(ppo_path)
    dqn_model = DQN.load(dqn_path)
    ppo_params = sum(p.numel() for p in ppo_model.policy.parameters())
    dqn_params = sum(p.numel() for p in dqn_model.q_net.parameters())

    # Training cost accounting:
    # 3 seeds PPO: 76,800 steps. 3 seeds DQN: 75,000 steps.
    # 8 ablation models: 49,152 steps. Validation / initial runs: ~40,000 steps.
    # Grand total steps across project: ~241,000 steps.
    # Hardware throughput: ~412 steps/sec on RTX 3050.
    ppo_train_sec = round(76800 / 412.0, 1)
    ppo_gpu_hrs = round(ppo_train_sec / 3600.0, 4)
    # Energy: 45W TGP * hours
    ppo_energy_kwh = round(45.0 * (ppo_train_sec / 3600.0) / 1000.0, 5)

    dqn_train_sec = round(75000 / 412.0, 1)
    dqn_gpu_hrs = round(dqn_train_sec / 3600.0, 4)
    dqn_energy_kwh = round(45.0 * (dqn_train_sec / 3600.0) / 1000.0, 5)

    profile_rows = [
        {
            "Model": "MaskablePPO (Proposed)",
            "Per_Step_Mean_ms": f"{latency_res['MaskablePPO']['mean_ms']:.3f}",
            "Per_Step_Median_ms": f"{latency_res['MaskablePPO']['median_ms']:.3f}",
            "Per_Step_p95_ms": f"{latency_res['MaskablePPO']['p95_ms']:.3f}",
            "Per_Step_p99_ms": f"{latency_res['MaskablePPO']['p99_ms']:.3f}",
            "Scalability_Complexity": "O(1) [Constant MLP]",
            "Parameter_Count": f"{ppo_params:,}",
            "Disk_Footprint_KB": ppo_sz_kb,
            "Training_Steps": "76,800 (3 seeds)",
            "Training_Time_s": ppo_train_sec,
            "GPU_Hours": ppo_gpu_hrs,
            "Energy_kWh": ppo_energy_kwh
        },
        {
            "Model": "Masked DQN",
            "Per_Step_Mean_ms": f"{latency_res['Masked DQN']['mean_ms']:.3f}",
            "Per_Step_Median_ms": f"{latency_res['Masked DQN']['median_ms']:.3f}",
            "Per_Step_p95_ms": f"{latency_res['Masked DQN']['p95_ms']:.3f}",
            "Per_Step_p99_ms": f"{latency_res['Masked DQN']['p99_ms']:.3f}",
            "Scalability_Complexity": "O(1) [Constant Q-Net]",
            "Parameter_Count": f"{dqn_params:,}",
            "Disk_Footprint_KB": dqn_sz_kb,
            "Training_Steps": "75,000 (3 seeds)",
            "Training_Time_s": dqn_train_sec,
            "GPU_Hours": dqn_gpu_hrs,
            "Energy_kWh": dqn_energy_kwh
        },
        {
            "Model": "Dynamic Dijkstra",
            "Per_Step_Mean_ms": f"{latency_res['Dynamic Dijkstra']['mean_ms']:.3f}",
            "Per_Step_Median_ms": f"{latency_res['Dynamic Dijkstra']['median_ms']:.3f}",
            "Per_Step_p95_ms": f"{latency_res['Dynamic Dijkstra']['p95_ms']:.3f}",
            "Per_Step_p99_ms": f"{latency_res['Dynamic Dijkstra']['p99_ms']:.3f}",
            "Scalability_Complexity": "O(E + V log V)",
            "Parameter_Count": "0 (Non-parametric)",
            "Disk_Footprint_KB": graph_sz_kb,
            "Training_Steps": "0 (Zero-shot)",
            "Training_Time_s": 0.0,
            "GPU_Hours": 0.0,
            "Energy_kWh": 0.0
        },
        {
            "Model": "Dynamic A*",
            "Per_Step_Mean_ms": f"{latency_res['Dynamic A*']['mean_ms']:.3f}",
            "Per_Step_Median_ms": f"{latency_res['Dynamic A*']['median_ms']:.3f}",
            "Per_Step_p95_ms": f"{latency_res['Dynamic A*']['p95_ms']:.3f}",
            "Per_Step_p99_ms": f"{latency_res['Dynamic A*']['p99_ms']:.3f}",
            "Scalability_Complexity": "O(E + V log V)",
            "Parameter_Count": "0 (Non-parametric)",
            "Disk_Footprint_KB": graph_sz_kb,
            "Training_Steps": "0 (Zero-shot)",
            "Training_Time_s": 0.0,
            "GPU_Hours": 0.0,
            "Energy_kWh": 0.0
        },
        {
            "Model": "Random Valid",
            "Per_Step_Mean_ms": f"{latency_res['Random Valid']['mean_ms']:.3f}",
            "Per_Step_Median_ms": f"{latency_res['Random Valid']['median_ms']:.3f}",
            "Per_Step_p95_ms": f"{latency_res['Random Valid']['p95_ms']:.3f}",
            "Per_Step_p99_ms": f"{latency_res['Random Valid']['p99_ms']:.3f}",
            "Scalability_Complexity": "O(deg(u))",
            "Parameter_Count": "0 (Non-parametric)",
            "Disk_Footprint_KB": 0.0,
            "Training_Steps": "0 (Zero-shot)",
            "Training_Time_s": 0.0,
            "GPU_Hours": 0.0,
            "Energy_kWh": 0.0
        }
    ]

    df_profile = pd.DataFrame(profile_rows)
    out_profile = "results/tables/efficiency_profile.csv"
    df_profile.to_csv(out_profile, index=False)
    print(f"Saved: {out_profile}")
    print("\n" + df_profile.to_string(index=False))


def main():
    print("=" * 80)
    print("PHASE B: COMPUTATIONAL EFFICIENCY & SCALABILITY PROFILING")
    print("=" * 80)

    latency_res = benchmark_per_step_latency()
    df_scale = benchmark_scalability()
    compute_efficiency_profile(latency_res)
    print("\nPhase B efficiency profiling complete!")


if __name__ == "__main__":
    main()
