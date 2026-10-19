# Offline-to-Online MARL Model Setup & Inference Guide

## Production Architecture

In accordance with the product architecture, the platform uses **EXCLUSIVELY ONE RL MODEL** in production:

**Offline-to-Online Multi-Agent Reinforcement Learning (MARL)**

The checkpoint is located at:
`models/marl_six/offline_online/seed_42/best_model.pt`

---

## Benchmark Performance Reference

The Offline-to-Online MARL model was evaluated across 500 multi-hazard test scenarios in the peer-reviewed research benchmark.

| Metric | Offline-to-Online MARL |
|---|---|
| **Joint Success Rate** | **92.88 ± 0.73%** |
| **Individual Success Rate** | **98.01%** |
| **Mean Evacuation Time** | **137.61 ± 1.42 s** |
| **Hazard Exposure Index (HEI)** | **6.836 ± 0.134** |
| **Routing F1 Score** | **0.9601** |
| **Inference Latency** | **3.575 ms (GPU) / ~15 ms (CPU)** |
| **Parameter Count** | **327,204** |

*Note: These are benchmark metrics and are strictly distinguished from live user-session measurements.*

---

## Observation & Action Specification

### Observation Space (34 Features)
The observation vector matches the exact input representation expected by the trained actor networks:

1. **Global & State Features (14 Dimensions)**:
   - Normalized agent $(x, y)$ coordinates in the road bounding box.
   - Normalized destination $(x, y)$ coordinates.
   - Shortest path distance to destination.
   - Normalized simulation step.
   - Current node hazard level (0.0 to 1.0).
   - Mean peer agent distance and relative positions.
   - Traffic congestion index on the current road segment.

2. **Candidate Successor Features (20 Dimensions = 4 Slots × 5 Attributes)**:
   - For up to 4 outgoing topological successors (sorted by distance to destination):
     - `is_valid`: Boolean flag (1.0 = navigable, 0.0 = invalid/masked).
     - `destination_alignment`: Cosine alignment towards evacuation target.
     - `distance_gain`: Normalized reduction in remaining path length.
     - `hazard_intensity`: Water depth or fire intensity on the target edge.
     - `congestion`: Microscopic vehicular density on the target edge.

### Action Space & Hard Action Masking
- **Action Space**: Discrete action space of size 4 representing candidate edge selections.
- **Action Masking**: Closed edges, flooded roads (>0.3m depth), or non-existent topological transitions are strictly masked out (`mask[a] = False`), preventing invalid action execution.

---

## Dynamic Fallback Engine
If an extreme compound disaster blocks all topologically viable paths or if PyTorch encounters an unrecoverable hardware exception, the system activates **Dynamic A\*** with live hazard and congestion impedance penalties. Fallback activation is completely transparent to the user via the `fallback_used` and `fallback_reason` fields.
