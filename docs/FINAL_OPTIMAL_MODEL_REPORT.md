# Research Report 2: Final Optimal Model Selection Report: TriHybrid Evacuation Agent & SafetyShieldPPO

**Project**: Adaptive RL Traffic Digital Twin for Multi-Hazard Evacuation Route Optimization  
**Date**: September 2026  
**Status**: Pareto Frontier Analysis & Winner Selection Completed  

---

## 1. Selected Pareto-Optimal Architecture

Following multi-objective Pareto analysis across the five evaluated hybrid models and historical baselines, **TriHybridEvacuationAgent** (Model 5) and **SafetyShieldPPO** (Model 4) emerge as the non-dominated Pareto-optimal champions of the entire benchmark.

```
Pareto Frontier Champion: TriHybridEvacuationAgent
  - Reliability: 96.20 ± 0.14% Success Rate, 3.80% Timeout Rate
  - Efficiency Winner: 133.85 ± 0.80 s Mean Evacuation Time (fastest across all models)
  - Routing Winner: 0.9342 Weighted F1 Agreement vs Optimal Routing (highest among RL models)
  - Zero-Shot Winner: 94.8% Success on OMR Suburban Corridor (vs 93.2% Baseline PPO)
```

---

## 2. Exact Architecture & Implementation Parameters

### 2.1 TriHybrid Composite Decision Equation
$$\text{DecisionScore}(a) = w_{\text{ppo}} \cdot \pi_{\text{GNN-PPO}}(a) + w_{\text{dqn}} \cdot Q_{\text{GNN-QRDQN}}^{\text{norm}}(a) + w_{\text{dijkstra}} \cdot I_{\text{Dijkstra}}(a) + w_{\text{safety}} \cdot S(a)$$

- **Component Weights**:
  - $w_{\text{ppo}} = 0.40$ (Policy exploration & dynamic traffic avoidance)
  - $w_{\text{dqn}} = 0.30$ (Quantile risk-sensitive return estimate)
  - $w_{\text{dijkstra}} = 0.20$ (Deterministic shortest-path guidance)
  - $w_{\text{safety}} = 0.10$ (Live local environmental safety certificate)

### 2.2 Safety Certificate Layer (SafetyShieldPPO)
$$S(a) = \max\left(0, 1.0 - [3.0 \cdot H(a) + 1.0 \cdot C(a)] / 4.0\right)$$
- **Tuned Weights**: $\alpha = 0.40, \beta = 0.45, \gamma = 0.15$
- **Dead-end sink rejection**: Active (rejects transitions into nodes with out-degree $0$ when alternatives exist).
- **Action Mask Compliance**: $100.0\%$ verified across all 15,000 transition steps.

---

## 3. Full Multi-Network Verification Summary

| Evaluation Benchmark | Success Rate | Mean Evac Time | Mean HEI | Travel Distance | Timeout Rate | Inference Latency |
|---|---|---|---|---|---|---|
| **84-Node Urban Core (100 Scenarios)** | 100.0% | 55.04 s | 0.42 | 610.5 m | 0.0% | 3.42 ms |
| **336-Node Multi-District (500 Scenarios)** | 96.20% | 133.85 s | 11.00 | 1520.43 m | 3.80% | 5.66 ms |
| **George Town Zero-Shot (250 Scenarios)** | 96.80% | 140.25 s | 9.69 | 1136.01 m | 3.20% | 6.36 ms |
| **OMR Suburban Zero-Shot (250 Scenarios)** | 94.80% | 147.89 s | 15.21 | 1073.68 m | 5.20% | 5.93 ms |

---

## 4. Multi-Seed Stability & Confidence Intervals (95% CI)
- **Success Rate across 5 Seeds**: $96.20 \pm 0.14\%$ (95% CI: $[96.08\%, 96.32\%]$)
- **Evacuation Time across 5 Seeds**: $133.85 \pm 0.80$ s (95% CI: $[133.15, 134.55]$ s)
- **HEI across 5 Seeds**: $11.00 \pm 0.46$ (95% CI: $[10.60, 11.40]$)
- **Routing Agreement F1**: $0.9342 \pm 0.005$

---

## 5. Comparison against Baselines
- **vs Baseline MaskablePPO**:
  - Success Rate: $+0.20\%$ higher ($96.20\%$ vs $96.00\%$)
  - Evacuation Time: $-1.09$ s faster ($133.85$ s vs $134.94$ s)
  - Hazard Exposure: $-4.9\%$ lower HEI ($11.00$ vs $11.57$)
  - Timeout Rate: $-0.20\%$ lower ($3.80\%$ vs $4.00\%$)
  - OMR Zero-Shot Success: $+1.60\%$ higher ($94.8\%$ vs $93.2\%$)
- **vs Baseline Masked DQN**:
  - Success Rate: $+11.60\%$ higher ($96.20\%$ vs $84.60\%$)
  - Evacuation Time: $-69.95$ s faster ($133.85$ s vs $203.80$ s)
  - Hazard Exposure: $-52.8\%$ lower HEI ($11.00$ vs $23.31$)
  - Timeout Rate: $-11.60\%$ lower ($3.80\%$ vs $15.40\%$)

---

## 6. Representative Failure Cases & Forensic Diagnostics
1. **Flash Flood Chokepoint Cutoff**:
   - *Scenario*: Rapid flood surge simultaneously severed multiple downstream arterial links.
   - *Diagnosis*: All outgoing candidate slots were declared `STATE_CLOSED`. The agent took emergency stationary hold, sustaining timeout rather than driving into impassable floodwaters.
2. **Dynamic Fire Wind Shift Trap**:
   - *Scenario*: Sudden wind direction shift drove wildfire across a peripheral bypass.
   - *Diagnosis*: When fire breached the downstream edge, the safety shield triggered an immediate reroute to slot 1, mitigating exposure with minimal travel time delay.
