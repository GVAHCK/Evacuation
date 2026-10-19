"""PPO Agent implementation with 5-stage curriculum learning using Stable-Baselines3."""

import os
import json
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("PPOAgent")


class StageMetricsCallback(BaseCallback):
    """Callback to log episodic success rates, evacuation times, and hazard exposure."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[bool] = []
        self.hazard_exposures: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "success" in info:
                self.successes.append(info["success"])
            if "hazard_exposure_index" in info:
                self.hazard_exposures.append(info["hazard_exposure_index"])
        return True


class PPOCurriculumTrainer:
    """Trains PPO agent across 5 distinct curriculum stages using MaskablePPO,

    logging TensorBoard metrics and saving stage checkpoints.
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        ppo_config_path: str = "configs/ppo.yaml",
        seed: int = 42,
        use_traci: bool = False
    ):
        self.seed = seed
        set_seed(seed)

        with open(config_path, "r", encoding="utf-8") as f:
            self.default_cfg = yaml.safe_load(f)
        with open(ppo_config_path, "r", encoding="utf-8") as f:
            self.ppo_cfg = yaml.safe_load(f)

        self.use_traci = use_traci
        self.models_dir = "models/ppo"
        self.logs_dir = self.ppo_cfg.get("logging", {}).get("tensorboard_log", "logs/ppo_tensorboard")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Load train scenarios
        scen_file = "data/generated/train_scenarios.json"
        self.train_scenarios = []
        if os.path.isfile(scen_file):
            with open(scen_file, "r", encoding="utf-8") as f:
                self.train_scenarios = json.load(f)

        # Hardware selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PPO training initialized on hardware device: {self.device}")

        # Instantiate environment wrapped with ActionMasker
        raw_env = EvacuationEnv(
            config_path=config_path,
            scenarios=self.train_scenarios,
            use_traci=self.use_traci,
            curriculum_stage=1
        )
        masked_env = ActionMasker(raw_env, lambda env: env.unwrapped.action_masks())
        self.env = Monitor(masked_env, filename=os.path.join(self.logs_dir, "monitor.csv"))

        # Setup PPO hyperparameters
        hp = self.ppo_cfg.get("hyperparameters", {})
        net_arch = self.ppo_cfg.get("policy", {}).get("net_arch", dict(pi=[128, 128], vf=[128, 128]))

        self.model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=self.env,
            learning_rate=float(hp.get("learning_rate", 3e-4)),
            n_steps=int(hp.get("n_steps", 2048)),
            batch_size=int(hp.get("batch_size", 64)),
            n_epochs=int(hp.get("n_epochs", 10)),
            gamma=float(hp.get("gamma", 0.99)),
            gae_lambda=float(hp.get("gae_lambda", 0.95)),
            clip_range=float(hp.get("clip_range", 0.2)),
            ent_coef=float(hp.get("ent_coef", 0.01)),
            vf_coef=float(hp.get("vf_coef", 0.5)),
            max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
            tensorboard_log=self.logs_dir,
            device=self.device,
            seed=seed,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1
        )

        self.curriculum_stages = self.ppo_cfg.get("curriculum", {}).get("stages", [])

    def train_curriculum(self, stage_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """Executes full 5-stage progressive curriculum training."""
        logger.info(f"Beginning PPO 5-stage curriculum learning ({len(self.curriculum_stages)} stages)...")
        stage_results = {}

        metrics_cb = StageMetricsCallback()

        for stg in self.curriculum_stages:
            stg_num = stg.get("stage", 1)
            stg_name = stg.get("name", f"Stage {stg_num}")
            stg_steps = stage_timesteps if stage_timesteps is not None else int(stg.get("timesteps", 10000))

            logger.info(f"=== Starting Curriculum Stage {stg_num}/5: {stg_name} ({stg_steps} steps) ===")

            # Update environment curriculum parameters
            self.env.unwrapped.curriculum_stage = stg_num

            # Learn for this stage
            self.model.learn(
                total_timesteps=stg_steps,
                callback=metrics_cb,
                reset_num_timesteps=False,
                tb_log_name=f"ppo_stage_{stg_num}"
            )

            # Save stage checkpoint
            stage_ckpt = os.path.join(self.models_dir, f"stage_{stg_num}_model.zip")
            self.model.save(stage_ckpt)
            logger.info(f"Stage {stg_num} checkpoint saved at {stage_ckpt}")

            # Record stage summary metrics
            recent_success = metrics_cb.successes[-50:] if metrics_cb.successes else [False]
            recent_hei = metrics_cb.hazard_exposures[-50:] if metrics_cb.hazard_exposures else [0.0]
            stage_results[f"stage_{stg_num}"] = {
                "name": stg_name,
                "timesteps": stg_steps,
                "recent_success_rate": float(np.mean(recent_success)),
                "mean_hazard_exposure": float(np.mean(recent_hei)),
                "checkpoint": stage_ckpt
            }

        # Save final models
        best_path = os.path.join(self.models_dir, "best_model.zip")
        latest_path = os.path.join(self.models_dir, "latest_model.zip")
        self.model.save(best_path)
        self.model.save(latest_path)
        logger.info(f"Final PPO model saved to {best_path} and {latest_path}")

        # Save training summary JSON
        with open(os.path.join(self.models_dir, "curriculum_summary.json"), "w", encoding="utf-8") as f:
            json.dump(stage_results, f, indent=2)

        self.env.close()
        return stage_results
