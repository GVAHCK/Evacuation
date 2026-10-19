"""Master Post-Training Benchmark Pipeline.
Runs evaluation, zero-shot transfers, statistical significance testing, Pareto optimality,
publication figures, and markdown report generation in a single automated flow.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger

logger = setup_logger("FullMARLBenchmark")


def run_cmd(cmd_list, desc):
    logger.info(f"==> Step: {desc}")
    logger.info(f"Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Error in {desc}:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        raise RuntimeError(f"Step {desc} failed with exit code {res.returncode}")
    else:
        logger.info(f"Successfully completed {desc}.\n{res.stdout[-300:] if len(res.stdout) > 300 else res.stdout}")


def main():
    py = sys.executable

    # Step 1: Evaluate all 6 models on Small Network
    run_cmd([py, "scripts/evaluate_marl_six.py", "--network", "small", "--max_scenarios", "50"],
            "1. Evaluate Six MARL Models on Small Network")

    # Step 2: Zero-Shot Cross-Topology Transfers
    run_cmd([py, "scripts/evaluate_zero_shot_marl_six.py", "--max_scenarios", "50"],
            "2. Zero-Shot Transfers on George Town and OMR")

    # Step 3: Statistical Significance Analysis
    run_cmd([py, "scripts/statistical_analysis_marl_six.py", "--network", "small"],
            "3. Pairwise Statistical Significance & Effect Size Testing")

    # Step 4: Multi-Objective Pareto Analysis
    run_cmd([py, "scripts/pareto_analysis_marl_six.py", "--network", "small"],
            "4. Multi-Objective Pareto Analysis & Category Winners")

    # Step 5: 15 Publication Figures & Master Final Report
    run_cmd([py, "scripts/generate_final_marl_report.py"],
            "5. 15 Publication Figures & Master Research Report")

    logger.info("ALL MARL BENCHMARK EVALUATIONS & REPORTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
