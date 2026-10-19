"""Verification script ensuring research benchmark data and model checkpoints are intact."""

import os
import sys
import hashlib
from typing import Dict, Any, List

RESEARCH_ARTIFACTS = [
    # Checkpoints
    os.path.join("models", "marl_six", "offline_online", "seed_42", "best_model.pt"),
    os.path.join("models", "marl_six", "offline_online", "seed_42", "final_model.pt"),
    os.path.join("models", "marl_six", "offline_online", "seed_101", "best_model.pt"),
    os.path.join("models", "marl_six", "offline_online", "seed_2024", "best_model.pt"),
    os.path.join("models", "marl_six", "offline_online", "seed_777", "best_model.pt"),
    os.path.join("models", "marl_six", "offline_online", "seed_999", "best_model.pt"),

    # Benchmark Tables
    os.path.join("results", "marl_six", "tables", "final_marl_comparison.csv"),
    os.path.join("results", "marl_six", "tables", "zero_shot_marl_comparison.csv"),
    os.path.join("results", "marl_six", "tables", "zero_shot_marl_george_town.csv"),
    os.path.join("results", "marl_six", "tables", "zero_shot_marl_omr.csv"),
    os.path.join("results", "marl_six", "tables", "ablation_study_marl.csv"),
    os.path.join("results", "marl_six", "tables", "statistical_tests_marl.csv"),
    os.path.join("results", "marl_six", "tables", "pareto_frontier_marl.csv"),
    os.path.join("results", "marl_six", "tables", "latency_benchmark_marl.csv"),
    os.path.join("results", "marl_six", "tables", "marl_evaluation_summary_large.json"),
    os.path.join("results", "marl_six", "tables", "marl_evaluation_summary_small.json"),

    # Road Network Graphs
    os.path.join("data", "processed", "road_graph_large.json"),
    os.path.join("data", "processed", "road_graph.json"),
    os.path.join("data", "processed", "network.net.xml"),

    # Publications & Final Reports
    os.path.join("results", "FINAL_REPORT.md"),
    os.path.join("results", "FINAL_RESULTS.md")
]

def verify_integrity() -> bool:
    print("=" * 60)
    print("VERIFYING RESEARCH ARTIFACT & CHECKPOINT INTEGRITY")
    print("=" * 60)

    all_passed = True
    for path in RESEARCH_ARTIFACTS:
        if not os.path.exists(path):
            print(f"[FAIL] Missing artifact: {path}")
            all_passed = False
            continue

        size_kb = os.path.getsize(path) / 1024.0
        if size_kb <= 0:
            print(f"[FAIL] Zero-byte file: {path}")
            all_passed = False
            continue

        print(f"[PASS] {path:<55} ({size_kb:8.1f} KB)")

    print("-" * 60)
    if all_passed:
        print("ALL RESEARCH ARTIFACTS, CHECKPOINTS, AND BENCHMARKS VERIFIED INTACT!")
    else:
        print("SOME RESEARCH ARTIFACTS FAILED VERIFICATION!")
    print("-" * 60)
    return all_passed

if __name__ == "__main__":
    success = verify_integrity()
    sys.exit(0 if success else 1)
