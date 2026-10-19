# DATASET CARD: Urban Road Network & Disaster Evacuation Scenarios

## 1. Dataset Overview

- **Dataset Name**: Chennai Guindy-Adyar Urban Evacuation Benchmark (CG-AUEB)
- **Domain**: Microscopic Traffic Digital Twin, Environmental Hazard Propagation, Dynamic Evacuation Routing
- **Geographic Coverage**: Guindy / Adyar Urban Arterial Corridor, Chennai, Tamil Nadu, India
- **Bounding Box**: Latitudes $12.990^\circ\text{N}$ to $13.015^\circ\text{N}$, Longitudes $80.205^\circ\text{E}$ to $80.240^\circ\text{E}$
- **Primary Purpose**: Benchmarking adaptive sequential decision-making, reinforcement learning, and dynamic graph algorithms under non-stationary flood and fire propagation with microscopic vehicle queuing.
- **License**: Open Data Commons Open Database License (ODbL) for OpenStreetMap road topology; MIT License for scenario generation pipeline.

---

## 2. Road Network Topology

The road network was extracted from OpenStreetMap using OSMnx, filtered for motorized vehicular infrastructure, and topologically repaired to eliminate sink nodes and disconnected components:

| Network Attribute | Exact Value | Description / Verification |
|---|---|---|
| **Node Count ($|V|$)** | **84** | Intersections and road junction vertices |
| **Directed Edge Count ($|E|$)** | **183** | One-way and bidirectional motorized street segments |
| **Total Road Length** | **12.80 km** | Cumulative metric length across all graph edges |
| **Graph Density** | **0.026248** | $\frac{\|E\|}{\|V\|(\|V\|-1)}$, characteristic of planar urban street networks |
| **Average Out-Degree** | **2.18** | Mean outgoing decision branches per intersection |
| **Strongly Connected Components (SCC)** | **1** | Single SCC; 100% of nodes reachable from any origin ($|V_{SCC}| = 84$) |
| **Dead-End Sinks (Out-Degree 0)** | **0** | Bidirectional arterial restoration ensures no vehicular dead ends |
| **Maximum Neighbor Degree** | **4** | Bound by action space $K=4$ in canonical candidate ordering |
| **Average Speed Limit** | **12.4 m/s (44.6 km/h)** | Extracted from OSM tags with arterial speed defaults |

- **Graph Storage File**: `data/processed/road_graph.json` (65.9 KB)
- **SUMO Network Representation**: `data/processed/sumo_net.net.xml` (94.2 KB)

---

## 3. Disaster Evacuation Scenario Datasets

A total of **1,300 unique evacuation scenarios** were generated across three disjoint random seed spaces to guarantee zero data leakage between training, curriculum development, and final statistical evaluation.

### 3.1 Split Summary & Partitioning

| Split Name | Scenario Count | File Path | File Size | Seed Range | Purpose |
|---|:---:|---|:---:|:---:|---|
| **Train Split** | 1,000 | `data/generated/train_scenarios.json` | 336.1 KB (0.33 MB) | `[10000, 10999]` | 5-stage progressive curriculum RL training |
| **Validation Split** | 200 | `data/generated/val_scenarios.json` | 66.4 KB (0.06 MB) | `[20000, 20199]` | Periodic evaluation during training / early stopping |
| **Test Split** | 100 | `data/generated/test_scenarios.json` | 33.4 KB (0.03 MB) | `[30000, 30099]` | Strictly untouched final statistical evaluation |

*Note: The seed ranges $[10000, 10999]$, $[20000, 20199]$, and $[30000, 30099]$ are mutually exclusive, guaranteeing independent disaster origins, wind patterns, and background traffic seeds.*

### 3.2 Environmental Hazard Distribution

Each scenario specifies a dynamic natural hazard initialized at a randomized origin node with cellular propagation parameters:

| Split | Flood Scenarios | Fire Scenarios | Flood Ratio | Fire Ratio | Mean Spread Multiplier | Spread Multiplier Range |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (1,000)** | 511 | 489 | 51.1% | 48.9% | 0.992 | $[0.600, 1.399]$ |
| **Validation (200)** | 98 | 102 | 49.0% | 51.0% | 0.982 | $[0.604, 1.397]$ |
| **Test (100)** | 51 | 49 | 51.0% | 49.0% | 0.991 | $[0.613, 1.368]$ |

- **Flood Dynamics**: Simulated via a stochastic digital elevation cellular automaton with downhill flow accumulation, runoff saturation, and road immersion stages (Safe $\to$ Degraded $\to$ Closed).
- **Fire Dynamics**: Simulated via anisotropic thermal diffusion incorporating live wind vectors ($\theta \in [0^\circ, 360^\circ]$, speed $\in [1.0, 8.0]\text{ m/s}$) and flammable fuel loads along road verges.

### 3.3 Microscopic Background Traffic Density

Background traffic is generated in Eclipse SUMO using randomized passenger trips synchronized via TraCI:

| Traffic Level | Vehicle Insertion Rate | Train Count (%) | Val Count (%) | Test Count (%) |
|---|---|:---:|:---:|:---:|
| **Low** | 5 veh/min | 178 (17.8%) | 34 (17.0%) | 17 (17.0%) |
| **Normal** | 12 veh/min | 201 (20.1%) | 47 (23.5%) | 26 (26.0%) |
| **Morning Peak** | 25 veh/min | 218 (21.8%) | 42 (21.0%) | 22 (22.0%) |
| **Evening Peak** | 25 veh/min | 193 (19.3%) | 38 (19.0%) | 20 (20.0%) |
| **Heavy** | 35 veh/min | 210 (21.0%) | 39 (19.5%) | 15 (15.0%) |

---

## 4. Compute & Simulation Step Accounting

Grand total simulation steps and compute budget consumed across all phases of the project:

| Experimental Phase | Algorithm / Task | Active Steps | Wall-Clock Time (s) | GPU Hours | Energy (kWh) |
|---|---|:---:|:---:|:---:|:---:|
| **Multi-Seed Scaled Training** | MaskablePPO (Seeds 42, 101, 2024) | 76,800 | 186.4 | 0.0518 | 0.00233 |
| **Multi-Seed Scaled Training** | Masked DQN (Seeds 42, 101, 2024) | 75,000 | 182.0 | 0.0506 | 0.00228 |
| **Ablation Studies Retraining** | 8 PPO Models Retrained (State & Reward) | 49,152 | 119.3 | 0.0331 | 0.00149 |
| **Curriculum Tuning & Diagnostic Runs** | Initial Validation & Diagnosis | 40,000 | 97.1 | 0.0270 | 0.00121 |
| **Statistical Benchmark Evaluation** | 900 Test Scenario Episodes | 6,426 | 15.6 | 0.0043 | 0.00019 |
| **Total Project Compute Budget** | **All Phases Combined** | **247,378** | **600.4 s (~10.0 min)** | **0.1668 h** | **0.00750 kWh** |

*Hardware Profiling*: Intel Core i5/i7 (12 logical cores), NVIDIA GeForce RTX 3050 6GB Laptop GPU (TGP ~45W), CUDA 12.1. Measured training throughput: $\approx 412.0\text{ steps/sec}$.

---

## 5. Software Dependencies & Reproducibility Environment

Exact runtime environment used to generate datasets, train models, and evaluate baselines:

```
Python:              3.11.15
Operating System:    Windows 11 (build 26100)
PyTorch:             2.5.1+cu121 (CUDA Active, cuDNN Enabled)
GPU Model:           NVIDIA GeForce RTX 3050 6GB Laptop GPU
Driver Version:      566.14 (CUDA 12.7 / Driver Compatible)
Eclipse SUMO:        1.27.1 (binaries: sumo.exe, netconvert.exe, traci 1.27.1)
Gymnasium:           1.3.0
Stable-Baselines3:   2.9.0
sb3-contrib:         2.9.0
NetworkX:            3.6.1
OSMnx:               2.1.1
NumPy:               2.2.3
Pandas:              2.2.3
SciPy:               1.15.2
Scikit-Learn:        1.6.1
Matplotlib:          3.10.0
Seaborn:             0.13.2
```

---

## 6. External Tools & Citations

1. **OpenStreetMap**: OpenStreetMap contributors, *OpenStreetMap Road Data Extract (ODbL)*, 2026. [https://www.openstreetmap.org](https://www.openstreetmap.org).
2. **Eclipse SUMO**: Lopez, P. A. et al., *Microscopic Traffic Simulation using SUMO*, IEEE ITSC, 2018. EPL 2.0 / GPL 2.0. [https://eclipse.dev/sumo/](https://eclipse.dev/sumo/).
3. **OSMnx**: Boeing, G., *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks*, Computers, Environment and Urban Systems, 2017. MIT License. [https://osmnx.readthedocs.io/](https://osmnx.readthedocs.io/).
4. **Stable-Baselines3 & sb3-contrib**: Raffin, A. et al., *Stable-Baselines3: Reliable Reinforcement Learning Implementations*, JMLR, 2021. MIT License. [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
5. **Gymnasium**: Towers, M. et al., *Gymnasium: A Standard Interface for Reinforcement Learning Environments*, arXiv:2407.17032, 2024. MIT License. [https://gymnasium.farama.org/](https://gymnasium.farama.org/).
