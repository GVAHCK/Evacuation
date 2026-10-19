"""Model 1: GNN-MaskablePPO Agent
Topology-Aware Graph Neural Network Policy with Invalid Action Masking.
"""

import os
import yaml
import torch
from typing import Dict, Any, Optional, List
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv
from src.network.gnn_layers import EgoGraphGNN

logger = setup_logger("GNN-MaskablePPO")


class GNNPPOTrainer:
    """Trainer for GNN-MaskablePPO combining an Ego-Graph message-passing encoder

    with Maskable PPO for topology-aware evacuation routing.
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        gnn_config_path: str = "configs/hybrid/gnn_ppo.yaml",
        seed: int = 42,
        use_traci: bool = False,
        n_envs: int = 4
    ):
        self.seed = seed
        set_seed(seed)
        self.config_path = config_path
        self.gnn_config_path = gnn_config_path
        self.use_traci = use_traci
        self.n_envs = n_envs

        with open(config_path, "r", encoding="utf-8") as f:
            self.default_cfg = yaml.safe_load(f)

        if os.path.isfile(gnn_config_path):
            with open(gnn_config_path, "r", encoding="utf-8") as f:
                self.gnn_cfg = yaml.safe_load(f)
        else:
            self.gnn_cfg = {}

        self.models_dir = f"models/hybrid/gnn_ppo/seed_{seed}"
        self.logs_dir = f"logs/hybrid/gnn_ppo/seed_{seed}"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"GNN-MaskablePPO initialized (seed={seed}, device={self.device})")

    def build_env(self, scenarios: Optional[List[Dict[str, Any]]] = None, curriculum_stage: int = 4):
        """Builds vectorized environment wrapped with ActionMasker."""
        def make_env():
            def _thunk():
                raw_env = EvacuationEnv(
                    config_path=self.config_path,
                    scenarios=scenarios,
                    use_traci=self.use_traci,
                    curriculum_stage=curriculum_stage
                )
                masked_env = ActionMasker(raw_env, lambda env: env.unwrapped.action_masks())
                return Monitor(masked_env)
            return _thunk

        return DummyVecEnv([make_env() for _ in range(self.n_envs)])

    def build_model(self, env) -> MaskablePPO:
        """Constructs MaskablePPO model with EgoGraphGNN feature extractor."""
        hp = self.gnn_cfg.get("hyperparameters", {})
        gnn_hp = self.gnn_cfg.get("gnn", {})

        features_dim = int(gnn_hp.get("features_dim", 64))
        hidden_dim = int(gnn_hp.get("hidden_dim", 64))
        num_layers = int(gnn_hp.get("num_layers", 2))

        policy_kwargs = dict(
            features_extractor_class=EgoGraphGNN,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_dim=hidden_dim,
                num_candidates=4,
                candidate_feature_dim=5,
                num_layers=num_layers
            ),
            net_arch=self.gnn_cfg.get("policy", {}).get("net_arch", dict(pi=[64, 64], vf=[64, 64]))
        )

        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=env,
            learning_rate=float(hp.get("learning_rate", 3e-4)),
            n_steps=int(hp.get("n_steps", 256)),
            batch_size=int(hp.get("batch_size", 64)),
            n_epochs=int(hp.get("n_epochs", 5)),
            gamma=float(hp.get("gamma", 0.99)),
            gae_lambda=float(hp.get("gae_lambda", 0.95)),
            clip_range=float(hp.get("clip_range", 0.2)),
            ent_coef=float(hp.get("ent_coef", 0.01)),
            vf_coef=float(hp.get("vf_coef", 0.5)),
            max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
            tensorboard_log=self.logs_dir,
            device=self.device,
            seed=self.seed,
            policy_kwargs=policy_kwargs,
            verbose=0
        )
        return model

    def train(
        self,
        train_scenarios: List[Dict[str, Any]],
        val_scenarios: Optional[List[Dict[str, Any]]] = None,
        total_timesteps: int = 50000
    ) -> str:
        """Trains GNN-MaskablePPO across specified timesteps."""
        env = self.build_env(scenarios=train_scenarios, curriculum_stage=1)
        model = self.build_model(env)

        logger.info(f"Training GNN-MaskablePPO across 5 curriculum stages for {total_timesteps} steps...")
        stage_steps = max(1000, total_timesteps // 5)
        for stg in range(1, 6):
            for sub_env in env.envs:
                sub_env.unwrapped.curriculum_stage = stg
            model.learn(total_timesteps=stage_steps, reset_num_timesteps=False)
            logger.info(f"GNN-MaskablePPO Seed {self.seed}: Completed Curriculum Stage {stg}/5")

        best_path = os.path.join(self.models_dir, "best_model.zip")
        latest_path = os.path.join(self.models_dir, "latest_model.zip")
        model.save(best_path)
        model.save(latest_path)

        # Save config
        with open(os.path.join(self.models_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.gnn_cfg, f)

        env.close()
        logger.info(f"Saved GNN-MaskablePPO checkpoint to {best_path}")
        return best_path


def load_gnn_ppo_agent(model_path: str, device: str = "auto") -> MaskablePPO:
    """Helper function to load a trained GNN-MaskablePPO agent."""
    custom_objects = {
        "EgoGraphGNN": EgoGraphGNN
    }
    return MaskablePPO.load(model_path, device=device, custom_objects=custom_objects)
