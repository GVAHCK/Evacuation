# Scientific Benchmark Data & Research Integrity Specification

## Overview

The research repository contains peer-reviewed empirical benchmarks evaluating Multi-Agent Reinforcement Learning (MARL) for urban emergency evacuation.

---

## Architectural Decision Regarding Benchmark Models

The research evaluation compared six MARL model formulations:
1. Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
2. Risk-Sensitive MADDPG (RS-MADDPG)
3. Multi-Agent Graph Reinforcement Learning (MA-GRL)
4. GraphSAGE Multi-Agent Proximal Policy Optimization (GraphSAGE-MAPPO)
5. Heterogeneous Agent Reinforcement Learning (HARL/HAPPO)
6. **Offline-to-Online MARL**

### Decision for Production Application:
**USE EXCLUSIVELY Offline-to-Online MARL IN THE PRODUCT.**

- The other five models remain documented as research artifacts and benchmark datasets.
- The production mobile application and backend router DO NOT load MADDPG, RS-MADDPG, MA-GRL, GraphSAGE-MAPPO, or HARL.
- The UI exposes **Offline-to-Online MARL** automatically as the selected routing engine.

---

## Benchmark Integrity Verification

A verification script (`scripts/verify_research_integrity.py`) guarantees that research checkpoints, statistical tables, and reports remain pristine:

```bash
.\run_py.bat scripts\verify_research_integrity.py
```

### Verified Benchmark Artifacts:
- Checkpoints: 6 model seeds in `models/marl_six/offline_online/` (Intact).
- Benchmark comparison table: `results/marl_six/tables/final_marl_comparison.csv` (Intact).
- Zero-shot cross-city transfer tables: `zero_shot_marl_george_town.csv`, `zero_shot_marl_omr.csv` (Intact).
- Ablation study table: `ablation_study_marl.csv` (Intact).
- Statistical significance tests: `statistical_tests_marl.csv` (Intact).
- Pareto frontier analysis: `pareto_frontier_marl.csv` (Intact).
- Comprehensive research report: `results/FINAL_REPORT.md` (Intact).
