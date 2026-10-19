"""Comprehensive Evaluation Framework running unseen test scenarios,

computing bootstrap confidence intervals, and paired statistical hypothesis tests.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple

from src.utils.logger import setup_logger
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from src.baselines.astar import AStarBaseline
from src.baselines.random_routing import RandomValidRoutingBaseline

logger = setup_logger("Evaluator")


def bootstrap_ci(data: Any, n_boot: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculates non-parametric bootstrap confidence intervals."""
    if data is None or len(data) < 2:
        return 0.0, 0.0
    arr = np.asarray(data)
    boot_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return lower, upper


class EvacuationEvaluator:
    """Evaluates RL policies and classical baselines across unseen test disaster scenarios."""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        test_scenarios_path: str = "data/generated/test_scenarios.json",
        use_traci: bool = False
    ):
        self.config_path = config_path
        self.use_traci = use_traci

        with open(test_scenarios_path, "r", encoding="utf-8") as f:
            self.test_scenarios = json.load(f)

        logger.info(f"Evaluator initialized with {len(self.test_scenarios)} unseen test scenarios")

    def evaluate_model(
        self,
        model_name: str,
        policy_or_baseline: Any,
        is_rl_agent: bool = True,
        max_scenarios: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Evaluates a single model or baseline over the test scenarios."""
        scenarios = self.test_scenarios[:max_scenarios] if max_scenarios else self.test_scenarios
        logger.info(f"Evaluating {model_name} on {len(scenarios)} scenarios...")

        env = EvacuationEnv(
            config_path=self.config_path,
            scenarios=scenarios,
            use_traci=self.use_traci,
            curriculum_stage=5
        )

        records = []
        for i, scen in enumerate(scenarios):
            obs, info = env.reset(options={"scenario": scen})
            done = False
            step = 0
            ep_reward = 0.0
            inf_times = []

            while not done:
                t0 = time.perf_counter()
                if is_rl_agent:
                    mask = env.action_masks()
                    try:
                        action, _ = policy_or_baseline.predict(obs, action_masks=mask, deterministic=True)
                    except TypeError:
                        if hasattr(policy_or_baseline, "q_net"):
                            obs_t = torch.as_tensor(obs, device=policy_or_baseline.device).unsqueeze(0)
                            with torch.no_grad():
                                q = policy_or_baseline.q_net(obs_t).squeeze(0).cpu().numpy()
                            q[~mask] = -1e9
                            action = int(np.argmax(q))
                        else:
                            action, _ = policy_or_baseline.predict(obs, deterministic=True)
                    action = int(action)
                else:
                    action = policy_or_baseline.select_action(env)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                inf_times.append(dt_ms)

                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                step += 1

            rec = {
                "model": model_name,
                "scenario_id": scen["scenario_id"],
                "hazard_type": scen["hazard_type"],
                "traffic_level": scen["traffic_level"],
                "success": bool(step_info.get("success", False)),
                "evacuation_time": float(step_info.get("cumulative_travel_time", 0.0)),
                "hazard_exposure": float(step_info.get("hazard_exposure_index", 0.0)),
                "travel_distance": float(step_info.get("cumulative_distance", 0.0)),
                "closure_encounters": int(step_info.get("closure_encounters", 0)),
                "timeout": bool(step_info.get("timeout", False)),
                "total_reward": float(ep_reward),
                "steps": int(step),
                "inference_time_ms": float(np.mean(inf_times)) if inf_times else 0.0
            }
            records.append(rec)

        env.close()
        return records

    def run_benchmark_suite(
        self,
        ppo_model=None,
        dqn_model=None,
        max_scenarios: Optional[int] = None
    ) -> pd.DataFrame:
        """Runs the entire benchmark comparison across PPO, DQN, Dijkstra, A*, and Random."""
        all_records = []

        # 1. Classical Baselines
        baselines = [
            ("Dijkstra", DijkstraBaseline()),
            ("A*", AStarBaseline()),
            ("Random", RandomValidRoutingBaseline(seed=42))
        ]
        for name, baseline_obj in baselines:
            recs = self.evaluate_model(name, baseline_obj, is_rl_agent=False, max_scenarios=max_scenarios)
            all_records.extend(recs)

        # 2. PPO Agent
        if ppo_model is not None:
            recs = self.evaluate_model("PPO", ppo_model, is_rl_agent=True, max_scenarios=max_scenarios)
            all_records.extend(recs)

        # 3. DQN Agent
        if dqn_model is not None:
            recs = self.evaluate_model("DQN", dqn_model, is_rl_agent=True, max_scenarios=max_scenarios)
            all_records.extend(recs)

        df = pd.DataFrame(all_records)
        os.makedirs("results/tables", exist_ok=True)
        raw_csv = "results/tables/raw_evaluation_records.csv"
        df.to_csv(raw_csv, index=False)
        logger.info(f"Raw evaluation records saved to {raw_csv}")

        # Compute summary table with confidence intervals
        summary_rows = []
        for m, group in df.groupby("model"):
            succ = group["success"].tolist()
            evac_t = group["evacuation_time"].tolist()
            hei = group["hazard_exposure"].tolist()
            dist = group["travel_distance"].tolist()
            to = group["timeout"].tolist()
            rew = group["total_reward"].tolist()
            inf_t = group["inference_time_ms"].tolist()

            t_ci = bootstrap_ci(evac_t)
            hei_ci = bootstrap_ci(hei)

            summary_rows.append({
                "Model": m,
                "Success_Rate": round(float(np.mean(succ)) * 100, 2),
                "Mean_Evacuation_Time": round(float(np.mean(evac_t)), 2),
                "Evac_Time_CI_95": f"[{t_ci[0]:.1f}, {t_ci[1]:.1f}]",
                "Median_Evacuation_Time": round(float(np.median(evac_t)), 2),
                "Hazard_Exposure": round(float(np.mean(hei)), 2),
                "HEI_CI_95": f"[{hei_ci[0]:.2f}, {hei_ci[1]:.2f}]",
                "Travel_Distance": round(float(np.mean(dist)), 2),
                "Timeout_Rate": round(float(np.mean(to)) * 100, 2),
                "Average_Reward": round(float(np.mean(rew)), 2),
                "Inference_Time_ms": round(float(np.mean(inf_t)), 3)
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = "results/tables/model_comparison.csv"
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Summary comparison table saved to {summary_csv}")

        # Statistical significance test (PPO vs Dijkstra / A*)
        if "PPO" in df["model"].unique() and "Dijkstra" in df["model"].unique():
            ppo_hei = df[df["model"] == "PPO"]["hazard_exposure"].values
            dijk_hei = df[df["model"] == "Dijkstra"]["hazard_exposure"].values
            if len(ppo_hei) == len(dijk_hei) and len(ppo_hei) > 5:
                t_stat, p_val = stats.ttest_rel(ppo_hei, dijk_hei)
                logger.info(f"Paired t-test HEI (PPO vs Dijkstra): t={t_stat:.3f}, p={p_val:.4e}")

        return summary_df
