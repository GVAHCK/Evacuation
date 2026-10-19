# Comprehensive System Audit: Multi-Agent Reinforcement Learning (MARL) Extension

**Project**: Adaptive RL Traffic Digital Twin for Multi-Hazard Evacuation Route Optimization  
**Document**: Repository Architecture, Multi-Agent Formulation, and Interface Audit  
**Date**: September 2026  
**Status**: Pre-Implementation MARL Architectural Audit Completed  

---

## 1. Executive Summary & Audit Purpose

This audit inspects the current repository state to establish a rigorous, mathematically sound foundation for extending the existing single-agent evacuation routing digital twin into a **six-model Multi-Agent Reinforcement Learning (MARL)** research framework:
1. **MADDPG**: Multi-Agent Deep Deterministic Policy Gradient (Centralized Critic, Discrete Action Representation)
2. **RS-MADDPG**: Refined MADDPG (Prioritized Replay, Graph-Aware Critic, Hazard Barrier Shaping)
3. **Offline-to-Online MARL**: Conservative Pretraining from Trajectory Datasets + Online Multi-Agent Fine-Tuning
4. **MA-GRL**: Multi-Agent Graph Reinforcement Learning (Ego-Graph Message Passing over Variable Topologies)
5. **GraphSAGE-MAPPO**: Inductive GraphSAGE Neighborhood Aggregation + Centralized Actor-Critic MAPPO
6. **HARL (HAPPO)**: Heterogeneous-Agent Reinforcement Learning (Heterogeneous Policy Gradient Monotonic Updates)

All existing baselines, checkpoints, configurations, and historical results in `models/`, `results/tables/`, and `configs/` remain **strictly frozen and immutable**.

---

## 2. Current Single-Agent System Architecture

### 2.1 Environmental Core
- **Simulator**: `src/environment/evacuation_env.py` (`EvacuationEnv`).
- **Road Network Representation**: `networkx.DiGraph` loaded via `src/network/osm_loader.py` from OpenStreetMap JSON exports.
  - Small network: 84 nodes, 183 edges (`data/processed/road_graph.json`).
  - Large network: 336 nodes, 736 edges (`data/processed/road_graph_large.json`).
  - George Town: 366 nodes, 994 edges (`data/processed/road_graph_dense_grid.json`).
  - OMR Suburban: 336 nodes, 870 edges (`data/processed/road_graph_sparse_suburban.json`).
- **Microscopic Traffic Digital Twin**: Eclipse SUMO 1.27.1 integration via TraCI sockets with dynamic edge speed degradation and closed road blocks. In internal fallback mode, digital twin speed/queue equations simulate traffic density dynamics.
- **Dynamic Hazard Engines**:
  - Flood: `src/hazard/flood_engine.py` (Cellular automata stochastic diffusion, elevation sensitivity, closure threshold $\ge 0.70$, degradation $\ge 0.35$).
  - Fire: `src/hazard/fire_engine.py` (Wind-directed anisotropic spread, vegetation factor, closure $\ge 0.65$, degradation $\ge 0.30$).

### 2.2 Observation Space (34 Dimensions)
The observation vector is strictly continuous $\in [0, 1]^{34}$ composed of:
- **14 Base Features**:
  1. `curr_x_norm`: Normalized agent X coordinate $[0, 1]$.
  2. `curr_y_norm`: Normalized agent Y coordinate $[0, 1]$.
  3. `dest_x_norm`: Normalized destination X coordinate $[0, 1]$.
  4. `dest_y_norm`: Normalized destination Y coordinate $[0, 1]$.
  5. `rem_dist_norm`: Shortest road distance to destination along graph $[0, 1]$.
  6. `curr_haz`: Hazard intensity at agent's current node $[0, 1]$.
  7. `mean_out_haz`: Average hazard intensity across outgoing candidate road edges $[0, 1]$.
  8. `frac_closed`: Fraction of outgoing edges completely blocked by hazard $[0, 1]$.
  9. `frac_degraded`: Fraction of outgoing edges speed-degraded $[0, 1]$.
  10. `congestion`: Local road density / congestion level $[0, 1]$.
  11. `avg_speed`: Local traffic speed normalized to free-flow $[0, 1]$.
  12. `queue_len`: Local vehicle queue length $[0, 1]$.
  13. `time_norm`: Current episode step normalized to `max_episode_steps` $[0, 1]$.
  14. `degree_norm`: Out-degree of current node normalized to `max_neighbors` (4) $[0, 1]$.
- **20 Candidate Action Edge Features** ($4 \text{ slots} \times 5 \text{ features}$):
  - Slot 0: Primary candidate (shortest road path to destination).
  - Slot 1: First detour candidate (second shortest path).
  - Slot 2: Second detour candidate (third shortest path).
  - Slot 3: Third detour candidate (fourth shortest path).
  - Each slot encodes: `[is_valid, dest_alignment_cos, dist_gain_norm, hazard_intensity, congestion]`.

### 2.3 Action Space & Hard Action Masking
- `action_space = spaces.Discrete(4)`.
- `action_masks() -> np.ndarray(shape=(4,), dtype=bool)`:
  - An action slot $k$ is valid (`True`) if the outgoing edge status is not `STATE_CLOSED`.
  - Fallback preservation: If all candidate edges are severed, slot 0 is retained to permit emergency stationary hold.
  - Invariant assertion: `assert action_mask[action] == True` must be strictly enforced on every single decision step.

### 2.4 Reward Decomposition
$$R = R_{\text{progress}} + R_{\text{arrival}} - R_{\text{hazard}} - R_{\text{closure}} - R_{\text{congestion}} - R_{\text{delay}} - R_{\text{dead\_end}} - R_{\text{timeout}} - R_{\text{u\_turn}}$$
- $R_{\text{progress}} = w_{\text{progress}} \cdot (\Delta \text{dist} / \text{scale})$: Positive when closing road distance to destination.
- $R_{\text{arrival}} = +100.0$ upon reaching destination.
- Penalties: hazard ($4.0 \cdot H$), closure ($-25.0$), congestion ($1.5 \cdot C$), delay ($-0.5$), dead-end ($-30.0$), timeout ($-40.0$), U-turn ($-3.0$).

---

## 3. Multi-Agent Formulation & Design Specifications

### 3.1 Entity Representation: Evacuation Units
- **Agent $i \in \{0, 1, \dots, N-1\}$**: Represents an independent evacuation convoy or localized vehicle cohort evacuating concurrently through the urban road network.
- **Default Scale**: $N = 4$ agents per scenario (tested up to $N = 8$ for scalability validation).
- **Heterogeneous Cohort Types**:
  - **Type A (Standard Cohort)**: Balanced progress and safety weights.
  - **Type B (Congestion-Sensitive Cohort)**: Large vehicle group or bus convoy with heightened penalty for high road density and bottlenecks ($w_{\text{congestion}} = 3.0$).
  - **Type C (Vulnerable / Hazard-Sensitive Cohort)**: Special medical or elder transport with amplified hazard avoidance penalty ($w_{\text{hazard}} = 8.0, w_{\text{closure}} = 40.0$).
  - **Type D (Long-Distance Perimeter Cohort)**: Peripheral evacuation cohort requiring high route efficiency and highway corridor prioritization ($w_{\text{progress}} = 6.0$).

### 3.2 Real Agent-to-Agent Interactions
Agents interact directly through environmental physics rather than artificial heuristics:
1. **Dynamic Road Occupancy & Congestion Coupling**: When multiple agents traverse or occupy the same road segment $(u, v)$ simultaneously, edge congestion increases quadratically:
   $$C_{e}^{\text{eff}} = C_{e}^{\text{ambient}} + \kappa \cdot \frac{N_{e}^{\text{agents}}}{K_{e}^{\text{capacity}}}$$
   causing mutual delay and queue accumulation.
2. **Spatial Bottleneck Contention (Conflicts)**: When multiple agents attempt to enter a constrained bridge or single-lane arterial concurrently, queueing conflicts occur, tracked via `conflict_count`.
3. **Shared Spatio-Temporal Hazard Exposure**: All $N$ agents navigate in the same live, spreading hazard footprint (flood diffusion or wind-blown fire CA). One agent's detour choices alter the traffic density encountered by peer agents attempting nearby bypasses.
4. **Joint vs Individual Objectives**: Joint episode success requires all $N$ agents to evacuate safely before timeout.

### 3.3 Multi-Agent Environment Architecture (`src/envs/multi_agent_evacuation_env.py`)
- Wraps the core `EvacuationEnv` digital twin without modifying underlying physics.
- `reset(seed, options)` -> returns joint observation `obs_n = [obs_0, obs_1, ..., obs_{N-1}]` where each `obs_i` is a 34-dimensional local vector, plus `info_n` containing individual action masks.
- `step(actions_n)` -> takes joint action vector `actions_n = [a_0, a_1, ..., a_{N-1}]`, checks all action masks (`assert mask_i[a_i] == True`), updates coupled edge occupancy, steps hazard and traffic, computes individual and shared rewards, and returns `(obs_n, rewards_n, terminations_n, truncations_n, infos_n)`.

---

## 4. Centralized Training with Decentralized Execution (CTDE)

| Phase | Decentralized Actor Network | Centralized Critic Network |
|---|---|---|
| **Training** | Receives local agent observation $o_i \in \mathbb{R}^{34}$ (and local ego-subgraph). | Receives global state $S = [o_0, o_1, \dots, o_{N-1}, G_{\text{global}}, C_{\text{map}}, H_{\text{map}}]$ and joint actions $(a_0, a_1, \dots, a_{N-1})$. |
| **Execution / Inference** | Strictly receives only $o_i$ and local action mask $M_i$. Zero access to peer private observations or centralized critic. | **Not used during inference**. Only local actors evaluate. |

---

## 5. Architectural Blueprint for the Six MARL Models

### Model 1: MADDPG (Discrete-Compatible CTDE)
- **Actor**: Policy network $\pi_{\theta_i}(o_i)$ with Gumbel-Softmax continuous relaxation during training, allowing differentiable backpropagation through categorical road candidate selections with hard $-\infty$ action masking.
- **Centralized Critic**: $Q_{\phi_i}(s, a_1, \dots, a_N)$ evaluating joint Q-values.
- **Target Networks & Replay Buffer**: Polyak averaging ($\tau = 0.005$) and multi-agent joint experience buffer.

### Model 2: RS-MADDPG (Refined MADDPG)
- Documented in [`docs/RS_MADDPG_DEFINITION.md`](file:///c:/Users/venka/OneDrive/Desktop/FinalYear/docs/RS_MADDPG_DEFINITION.md).
- **Refinement 1: Prioritized Experience Replay (PER)**: Multi-agent TD-error prioritized sampling ($\alpha=0.6, \beta=0.4 \to 1.0$).
- **Refinement 2: Graph-Aware Centralized Critic**: Critic incorporates an inter-agent relational attention layer over road network edge conflicts.
- **Refinement 3: Hazard Barrier Reward Shaping**: Negative potential-based hazard barrier function providing dense gradient guidance away from advancing flood fronts.

### Model 3: Offline-to-Online MARL
- **Phase A (Offline Pretraining)**:
  - Generates disjoint offline multi-agent dataset in `data/offline_marl/` using historical policies (PPO, DQN, Dijkstra, A*, Random-valid).
  - Trains via Multi-Agent Conservative Q-Learning (CQL) to penalize out-of-distribution actions.
- **Phase B (Online Fine-Tuning)**:
  - Initializes actors and critics from offline checkpoints; fine-tunes directly in `MultiAgentEvacuationEnv` with conservative policy regularization.
  - Quantifies offline performance vs online fine-tuned delta.

### Model 4: MA-GRL (Multi-Agent Graph Reinforcement Learning)
- **Graph Representation**: Explicit variable-size road network $G = (V, E)$ with dynamic node features (hazard, congestion, occupancy) and edge features (length, closure status).
- **Decentralized Actor**: 2-layer Ego-Graph message passing centered on the agent's current intersection, outputting permutation-invariant candidate scores.
- **Centralized Critic**: Global graph readout pooling + joint agent states. Zero-shot transfer ready across arbitrary graph dimensions.

### Model 5: GraphSAGE-MAPPO
- **Inductive GraphSAGE Layers**: Uniform neighbor sampling and mean aggregation:
  $$h_v^{(l)} = \sigma\left(W_1^{(l)} h_v^{(l-1)} + W_2^{(l)} \cdot \text{MEAN}_{u \in \mathcal{N}(v)} h_u^{(l-1)}\right)$$
- **Decentralized Actor**: PPO actor head with GAE advantage estimation, PPO clipping ($\epsilon=0.2$), entropy regularization, and hard action masking.
- **Centralized Critic**: Evaluates joint value function $V_\psi(s)$ conditioned on global GraphSAGE pooling.

### Model 6: HARL (Heterogeneous-Agent Reinforcement Learning - HAPPO)
- **Sequential Multi-Agent Updates**: Implements Heterogeneous-Agent PPO (HAPPO) with advantage factorization:
  $$A_{\pi}^i(s, a_{1:i}) = Q_{\pi}(s, a_{1:i}) - Q_{\pi}(s, a_{1:i-1})$$
- Agents update policies sequentially in random permutation order, guaranteeing monotonic improvement for heterogeneous agent types (Type A, B, C, D).
- Dedicated policy parameters per heterogeneous cohort.

---

## 6. Directory Structure & File Manifest

### Exact Files to be Created:
- **Core Multi-Agent Environment**:
  - `src/envs/__init__.py`
  - `src/envs/multi_agent_evacuation_env.py`
- **Agent Modules**:
  - `src/agents/marl_six/__init__.py`
  - `src/agents/marl_six/maddpg_agent.py`
  - `src/agents/marl_six/rs_maddpg_agent.py`
  - `src/agents/marl_six/offline_online_agent.py`
  - `src/agents/marl_six/ma_grl_agent.py`
  - `src/agents/marl_six/graphsage_mappo_agent.py`
  - `src/agents/marl_six/harl_agent.py`
- **Configuration Files**:
  - `configs/marl_six/maddpg.yaml`
  - `configs/marl_six/rs_maddpg.yaml`
  - `configs/marl_six/offline_online.yaml`
  - `configs/marl_six/ma_grl.yaml`
  - `configs/marl_six/graphsage_mappo.yaml`
  - `configs/marl_six/harl.yaml`
- **Pipelines & Experiment Scripts**:
  - `scripts/train_marl_six.py`
  - `scripts/tune_marl_six.py`
  - `scripts/evaluate_marl_six.py`
  - `scripts/evaluate_zero_shot_marl_six.py`
  - `scripts/statistical_analysis_marl_six.py`
  - `scripts/pareto_analysis_marl_six.py`
  - `scripts/generate_final_marl_report.py`
- **Unit Tests**:
  - `tests/test_marl_six.py`
- **Documentation**:
  - `docs/MARL_SIX_MODEL_AUDIT.md` (this file)
  - `docs/RS_MADDPG_DEFINITION.md`
  - `reports/marl_six/FINAL_SIX_MODEL_REPORT.md`
- **Results Storage**:
  - `results/marl_six/raw/`
  - `results/marl_six/tables/final_marl_comparison.csv`
  - `results/marl_six/figures/` (15 publication figures)
  - `models/marl_six/` (model checkpoints across 5 seeds)

---

## 7. Audit Sign-Off & Next Steps
With the repository audit completed and all single-agent dependencies cataloged, execution will proceed strictly according to the implementation plan:
1. Construct and unit-test `MultiAgentEvacuationEnv`.
2. Implement all six MARL agent architectures natively in PyTorch with hard action masking.
3. Build offline dataset in `data/offline_marl/`.
4. Execute quick-mode smoke tests across all models.
5. Perform validation hyperparameter tuning.
6. Execute 5-seed training across the benchmark.
7. Conduct in-distribution (84-node, 336-node) and zero-shot (George Town, OMR) evaluations.
8. Perform Wilcoxon/t-test statistical analysis, Pareto multi-objective optimization, and generate all 15 publication figures and final reports.
