# Architectural Specification: RS-MADDPG (Refined Multi-Agent DDPG)

**Project**: Adaptive RL Traffic Digital Twin for Multi-Hazard Evacuation Route Optimization  
**Document**: Mathematical Formulation, Algorithmic Refinements, and Implementation Mapping  
**Date**: September 2026  
**Status**: Formal Architectural Specification  

---

## 1. Motivation & Context

Multi-Agent Deep Deterministic Policy Gradient (MADDPG; Lowe et al., 2017) provides a foundational Centralized Training with Decentralized Execution (CTDE) framework for multi-agent systems. However, standard MADDPG exhibits specific vulnerabilities in dynamic disaster evacuation routing environments:
1. **Uniform Replay Inefficiency**: Rare, high-consequence failure events (such as bridge cutoffs or rapid flood trapping) are diluted by trivial free-flow navigation transitions in standard uniform experience replay.
2. **Flat Joint-Critic Limitation**: Standard centralized critics concatenate all agent actions and states into a flat multi-layer perceptron (MLP), failing to capture the spatial graph structure and physical bottleneck interactions between agents sharing road segments.
3. **Sparse Gradient Guidance Near Hazards**: Standard penalty-based rewards fail to provide dense gradient directionality when agents navigate near advancing flood or wildfire fronts.

To resolve these challenges without falsely claiming an unverified third-party reproduction, **RS-MADDPG ("Refined MADDPG")** introduces three mathematically defined, explicitly documented refinements over base MADDPG.

---

## 2. Base MADDPG Formulation for Discrete Action Routing

In our discrete road-successor selection setting ($a_i \in \{0, 1, 2, 3\}$ representing canonical candidate edges), standard continuous MADDPG cannot directly output continuous forces. We employ the discrete-action Gumbel-Softmax formulation:

### 2.1 Decentralized Actor
Each agent $i \in \{1, \dots, N\}$ parameterizes an actor policy $\pi_{\theta_i}(a_i \mid o_i)$. During training, continuous relaxed action probabilities are sampled via the Gumbel-Softmax distribution with hard invalid action masking:
$$z_{i, k} = \frac{\exp\left((\log \pi_{\theta_i}(a_i = k \mid o_i) + g_{i, k} + M_{i, k}) / \tau\right)}{\sum_{j=0}^{K-1} \exp\left((\log \pi_{\theta_i}(a_i = j \mid o_i) + g_{i, j} + M_{i, j}) / \tau\right)}$$
where $g_{i, k} \sim \text{Gumbel}(0, 1)$, $\tau$ is the temperature, and $M_{i, k} = 0$ if action $k$ is permitted by the action mask, and $-\infty$ otherwise.

During decentralized execution/inference, deterministic greedy selection is used:
$$a_i^* = \arg\max_{k \in \{0, \dots, K-1\}} \left( \pi_{\theta_i}(a_i = k \mid o_i) + M_{i, k} \right)$$
satisfying `assert action_mask[action] == True`.

### 2.2 Centralized Critic
Each agent has a centralized action-value function $Q_{\phi_i}(s, a_1, \dots, a_N)$ conditioned on the global state $s = [o_1, \dots, o_N]$ and joint action vector. The critic loss is:
$$\mathcal{L}(\phi_i) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}} \left[ \left( Q_{\phi_i}(s, a_1, \dots, a_N) - y_i \right)^2 \right]$$
$$y_i = r_i + \gamma (1 - d_i) Q_{\phi_i'}(s', a_1', \dots, a_N') \big|_{a_j' = \pi_{\theta_j'}(o_j')}$$

The actor policy gradient is:
$$\nabla_{\theta_i} \mathcal{J}(\theta_i) = \mathbb{E}_{s \sim \mathcal{D}, a_i \sim \pi_i} \left[ \nabla_{\theta_i} \log \pi_{\theta_i}(a_i \mid o_i) \cdot Q_{\phi_i}(s, a_1, \dots, a_i, \dots, a_N) \right]$$

---

## 3. The Three Key Refinements in RS-MADDPG

### Refinement 1: Multi-Agent Prioritized Experience Replay (MA-PER)
Instead of uniform sampling, transitions are sampled with probability proportional to their joint TD error:
$$P(k) = \frac{p_k^\alpha}{\sum_j p_j^\alpha}, \quad p_k = \max_{i \in \{1..N\}} |\delta_{i, k}| + \epsilon$$
$$\delta_{i, k} = y_{i, k} - Q_{\phi_i}(s_k, a_{1, k}, \dots, a_{N, k})$$
Importance sampling weights $w_k = (N_{\text{buffer}} \cdot P(k))^{-\beta}$ compensate for the non-uniform sampling bias, where $\beta$ is annealed linearly from $\beta_0 = 0.4$ to $1.0$ during training. High-conflict and near-miss hazard scenarios are sampled preferentially, accelerating learning of multi-agent bottleneck coordination.

### Refinement 2: Graph-Aware Relational Critic (GARC)
Rather than concatenating agent states into an unstructured flat vector, the centralized critic in RS-MADDPG encodes the relational graph topology connecting agents:
1. **Agent Node Embedding**: Each agent's state $o_i$ is mapped to node feature $h_i^{(0)} = \text{MLP}_{\text{node}}(o_i)$.
2. **Relational Conflict Edge Embedding**: For each agent pair $(i, j)$, an interaction edge feature $e_{ij}$ represents:
   - Road network topological distance between agent $i$ and agent $j$: $d_{\text{graph}}(u_i, u_j)$.
   - Destination road path overlap indicator: $I_{\text{overlap}}(i, j)$.
   - Joint edge occupancy indicator: $I(e_i = e_j)$.
3. **Multi-Head Relational Attention**:
   $$h_i^{(1)} = \sum_{j=1}^N \alpha_{ij} W_V h_j^{(0)}, \quad \alpha_{ij} = \frac{\exp\left( (W_Q h_i^{(0)})^T (W_K h_j^{(0)} + W_E e_{ij}) / \sqrt{d} \right)}{\sum_{m=1}^N \exp\left( (W_Q h_i^{(0)})^T (W_K h_m^{(0)} + W_E e_{im}) / \sqrt{d} \right)}$$
4. **Value Readout**: $Q_{\phi_i}(s, a) = \text{MLP}_{\text{val}}([h_i^{(1)}; a_1; \dots; a_N])$.
This allows the critic to explicitly evaluate the cost of spatial congestion and road contention at choke-point bridges.

### Refinement 3: Spatial Hazard Barrier Potential Shaping (SHB-PS)
To provide dense directional safety gradients, we introduce an admissible potential-based reward shaping function $\Phi(s_i)$ satisfying Ng et al. (1999) policy invariance:
$$F(s_i, s_i') = \gamma \Phi(s_i') - \Phi(s_i)$$
$$\Phi(s_i) = - \lambda_{\text{barrier}} \cdot \ln\left( 1.0 + \frac{H(u_i)}{1.01 - H(u_i)} \right)$$
where $H(u_i) \in [0, 1]$ is the local hazard intensity.
- As the agent approaches an impassable flood front ($H \to 1.0$), $\Phi(s_i) \to -\infty$, creating a steep barrier repelling the agent away from hazard traps before road closure occurs.
- As the agent moves away from hazard ($H \to 0$), $\Phi(s_i) \to 0$, providing a positive shaping reward.

---

## 4. Implementation Mapping

| Module Component | PyTorch Implementation in `src/agents/marl_six/rs_maddpg_agent.py` |
|---|---|
| Decentralized Actor | `RSMADDPGActor`: Linear(34 -> 128 -> 128 -> 4), Action Masking, Gumbel-Softmax |
| Graph-Aware Critic | `RSMADDPGCritic`: Multi-head relational attention over inter-agent spatial distances |
| Prioritized Buffer | `PrioritizedMultiAgentReplayBuffer`: Sum-Tree data structure with IS weight updates |
| Reward Shaping | `apply_hazard_barrier_potential(r_i, curr_node, next_node, hazard_engine)` |
| Action Mask Invariant | Strict assertion `assert mask[action] == True` prior to environment step |

---

## 5. Summary of Differences: MADDPG vs RS-MADDPG

| Dimension | Base MADDPG | RS-MADDPG |
|---|---|---|
| **Experience Replay** | Uniform random sampling | Multi-Agent Prioritized Replay (TD error) |
| **Critic Architecture** | Flat MLP over all $o_i$ and $a_i$ | Relational Multi-Head Attention over road topology |
| **Reward Guidance** | Stepwise penalty subtraction | Potential-based Hazard Barrier shaping ($\gamma \Phi' - \Phi$) |
| **Sample Efficiency** | Baseline sample efficiency | Higher sample efficiency on bottleneck conflicts |
| **Safety Certification** | Soft hazard penalty only | Dense barrier gradient preventing cutoffs |
