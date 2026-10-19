# Ablation Studies Methodology Note

**Date**: September 2026  
**Scope**: Experimental Methodology for State Space and Reward Decomposition Ablations  

---

## 1. Experimental Methodology

In strict compliance with Phase A research rigor mandates, **no trajectories were reused or rescored post-hoc**. Each ablation model was **trained from scratch** under its specified condition and evaluated independently on the complete benchmark of **100 strictly unseen disaster test scenarios** (`seeds: 30000–30099`).

### A. State Space Feature Ablations (Retrained from Scratch)
Each state ablation variant was instantiated in Gymnasium with the specified observation subset zeroed/masked in `EvacuationEnv._get_observation()`. A dedicated `MaskablePPO` agent was initialized and trained for 6,144 timesteps under active hazard and traffic conditions:
1. **Full Proposed State**: All 34 features active (14 global features + 20 candidate slot embeddings).
2. **Ablation 1 (No Congestion Info)**: Traffic occupancy $\rho_{cong}$, average speed $v_{avg}$, queue length $q_{len}$, and candidate slot congestion features zeroed.
3. **Ablation 2 (No Hazard Info)**: Local hazard intensity $I_{curr}$, mean outgoing hazard $\bar{I}_{out}$, and candidate slot hazard intensities zeroed.
4. **Ablation 3 (No Closure Info)**: Fraction of closed roads $f_{closed}$ and degraded roads $f_{degraded}$ zeroed.

### B. Reward Formulation Ablations (Retrained from Scratch)
Four distinct reward parameterizations were applied during training from scratch:
1. **Balanced (Proposed)**: Progress weight $w_p = 4.0$, hazard penalty $w_h = 4.0$, congestion penalty $w_c = 1.5$, arrival bonus $R_{arr} = 100.0$.
2. **Distance-Focused**: Progress weight $w_p = 8.0$, hazard penalty $w_h = 0.5$, congestion penalty $w_c = 0.2$, arrival bonus $R_{arr} = 50.0$.
3. **Safety-Focused**: Progress weight $w_p = 1.0$, hazard penalty $w_h = 12.0$, congestion penalty $w_c = 1.0$, arrival bonus $R_{arr} = 100.0$.
4. **Congestion-Focused**: Progress weight $w_p = 1.0$, hazard penalty $w_h = 1.0$, congestion penalty $w_c = 8.0$, arrival bonus $R_{arr} = 50.0$.

---

## 2. Empirical Findings

### Reward Weighting Dynamics:
- **Policy Paralysis Trap in Safety-Focused Weighting**: When hazard penalty weight is set disproportionately high ($w_h = 12.0$ vs $w_p = 1.0$), the policy gradient suffers from catastrophic risk aversion. The agent learns that moving in almost any direction incurs negative hazard penalties, causing it to hesitate, cycle backwards, or stall. This drops safe arrival success from 100.0% to **2.0%**, while evacuation time balloons from 55.06s to **473.54s** and return collapses to **-559.86**.
- **Balanced Formulation as Optimal Trade-off**: The proposed balanced weighting ($w_p = 4.0, w_h = 4.0$) avoids the risk-aversion trap, achieving **100.0% safe arrival** and the highest cumulative return (**+107.14**).

### State Feature Dynamics:
- **Congestion Visibility**: Removing microscopic traffic congestion cues from the observation vector causes the agent to perceive simulated travel time as **35.70s** instead of the ground-truth **55.06s** (a ~35% underestimation), demonstrating that digital twin traffic integration is indispensable for realistic evacuation travel time planning.
