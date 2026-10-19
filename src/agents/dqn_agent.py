"""DQN Agent implementation using Stable-Baselines3 for RL comparison."""

import os
import json
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv

logger = setup_logger("DQNAgent")


class DQNTrainer:
    """Trains Deep Q-Network (DQN) agent on the identical evacuation digital twin environment."""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        dqn_config_path: str = "configs/dqn.yaml",
        seed: int = 42,
        use_traci: bool = False
    ):
        self.seed = seed
        set_seed(seed)

        with open(config_path, "r", encoding="utf-8") as f:
            self.default_cfg = yaml.safe_load(f)
        with open(dqn_config_path, "r", encoding="utf-8") as f:
            self.dqn_cfg = yaml.safe_load(f)

        self.use_traci = use_traci
        self.models_dir = "models/dqn"
        self.logs_dir = self.dqn_cfg.get("logging", {}).get("tensorboard_log", "logs/dqn_tensorboard")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        scen_file = "data/generated/train_scenarios.json"
        self.train_scenarios = []
        if os.path.isfile(scen_file):
            with open(scen_file, "r", encoding="utf-8") as f:
                self.train_scenarios = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"DQN training initialized on hardware device: {self.device}")

        raw_env = EvacuationEnv(
            config_path=config_path,
            scenarios=self.train_scenarios,
            use_traci=self.use_traci,
            curriculum_stage=5
        )
        self.env = Monitor(raw_env, filename=os.path.join(self.logs_dir, "monitor.csv"))

        hp = self.dqn_cfg.get("hyperparameters", {})
        net_arch = self.dqn_cfg.get("policy", {}).get("net_arch", [128, 128])

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=float(hp.get("learning_rate", 1e-4)),
            buffer_size=int(hp.get("buffer_size", 50000)),
            learning_starts=int(hp.get("learning_starts", 1000)),
            batch_size=int(hp.get("batch_size", 64)),
            tau=float(hp.get("tau", 1.0)),
            gamma=float(hp.get("gamma", 0.99)),
            train_freq=int(hp.get("train_freq", 4)),
            target_update_interval=int(hp.get("target_update_interval", 1000)),
            exploration_fraction=float(hp.get("exploration_fraction", 0.3)),
            exploration_initial_eps=float(hp.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(hp.get("exploration_final_eps", 0.05)),
            max_grad_norm=float(hp.get("max_grad_norm", 10.0)),
            tensorboard_log=self.logs_dir,
            device=self.device,
            seed=seed,
            policy_kwargs=dict(net_arch=net_arch),
            verbose=1
        )

    def train(self, total_timesteps: int = 25000) -> str:
        """Trains DQN agent for specified timesteps and saves checkpoints."""
        logger.info(f"Training DQN agent for {total_timesteps} timesteps on {self.device}...")
        self.model.learn(total_timesteps=total_timesteps, tb_log_name="dqn_run")

        best_path = os.path.join(self.models_dir, "best_model.zip")
        latest_path = os.path.join(self.models_dir, "latest_model.zip")
        self.model.save(best_path)
        self.model.save(latest_path)
        logger.info(f"DQN models saved to {best_path} and {latest_path}")

        # Save hyperparams JSON
        with open(os.path.join(self.models_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
            json.dump(self.dqn_cfg, f, indent=2)

        self.env.close()
        return best_path
