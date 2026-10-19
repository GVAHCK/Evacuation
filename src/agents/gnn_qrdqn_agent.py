"""Model 3: GNN-QRDQN Agent
Distributional Quantile Regression Q-Network with Topology-Aware Graph Encoder
and Hard Action Masking.
"""

import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor

from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.environment.evacuation_env import EvacuationEnv
from src.network.gnn_layers import EgoGraphGNN

logger = setup_logger("GNN-QRDQN")


class GNNQRDQNTrainer:
    """Trainer for GNN-QRDQN combining distributional quantile regression

    with an Ego-Graph message-passing feature extractor and hard action masking.
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        qrdqn_config_path: str = "configs/hybrid/gnn_qrdqn.yaml",
        seed: int = 42,
        use_traci: bool = False
    ):
        self.seed = seed
        set_seed(seed)
        self.config_path = config_path
        self.qrdqn_config_path = qrdqn_config_path
        self.use_traci = use_traci

        with open(config_path, "r", encoding="utf-8") as f:
            self.default_cfg = yaml.safe_load(f)

        if os.path.isfile(qrdqn_config_path):
            with open(qrdqn_config_path, "r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_dir = f"models/hybrid/gnn_qrdqn/seed_{seed}"
        self.logs_dir = f"logs/hybrid/gnn_qrdqn/seed_{seed}"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        logger.info(f"GNN-QRDQN initialized (seed={seed}, device={self.device})")

    def build_model(self, env) -> QRDQN:
        """Constructs QRDQN model with EgoGraphGNN feature extractor and quantiles."""
        hp = self.cfg.get("hyperparameters", {})
        gnn_hp = self.cfg.get("gnn", {})

        n_quantiles = int(hp.get("n_quantiles", 25))
        features_dim = int(gnn_hp.get("features_dim", 64))
        hidden_dim = int(gnn_hp.get("hidden_dim", 64))

        policy_kwargs = dict(
            n_quantiles=n_quantiles,
            features_extractor_class=EgoGraphGNN,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                hidden_dim=hidden_dim,
                num_candidates=4,
                candidate_feature_dim=5,
                num_layers=2
            ),
            net_arch=self.cfg.get("policy", {}).get("net_arch", [64, 64])
        )

        model = QRDQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=float(hp.get("learning_rate", 1e-4)),
            buffer_size=int(hp.get("buffer_size", 25000)),
            learning_starts=int(hp.get("learning_starts", 500)),
            batch_size=int(hp.get("batch_size", 64)),
            tau=float(hp.get("tau", 1.0)),
            gamma=float(hp.get("gamma", 0.99)),
            train_freq=int(hp.get("train_freq", 4)),
            target_update_interval=int(hp.get("target_update_interval", 1000)),
            exploration_fraction=float(hp.get("exploration_fraction", 0.2)),
            exploration_initial_eps=float(hp.get("exploration_initial_eps", 1.0)),
            exploration_final_eps=float(hp.get("exploration_final_eps", 0.05)),
            max_grad_norm=float(hp.get("max_grad_norm", 10.0)),
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
        """Trains GNN-QRDQN agent across specified timesteps."""
        raw_env = EvacuationEnv(
            config_path=self.config_path,
            scenarios=train_scenarios,
            use_traci=self.use_traci,
            curriculum_stage=4
        )
        env = Monitor(raw_env)
        model = self.build_model(env)

        logger.info(f"Training GNN-QRDQN for {total_timesteps} steps on {self.device}...")
        model.learn(total_timesteps=total_timesteps)

        best_path = os.path.join(self.models_dir, "best_model.zip")
        latest_path = os.path.join(self.models_dir, "latest_model.zip")
        model.save(best_path)
        model.save(latest_path)

        with open(os.path.join(self.models_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(self.cfg, f)

        env.close()
        logger.info(f"Saved GNN-QRDQN checkpoint to {best_path}")
        return best_path


class MaskedGNNQRDQNAgent:
    """Wrapper for trained GNN-QRDQN providing hard action masking, quantile statistics,

    and risk-aware (mean or CVaR) action selection.
    """

    def __init__(self, model_path: str, device: str = "auto", risk_metric: str = "mean", cvar_alpha: float = 0.2):
        self.device = device
        custom_objects = {"EgoGraphGNN": EgoGraphGNN}
        self.model = QRDQN.load(model_path, device=device, custom_objects=custom_objects)
        self.risk_metric = risk_metric
        self.cvar_alpha = cvar_alpha

    def get_quantiles(self, obs: np.ndarray) -> np.ndarray:
        """Evaluates quantile distributions: returns (num_actions, num_quantiles)."""
        obs_t = torch.as_tensor(obs, device=self.model.device).float()
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            out = self.model.quantile_net(obs_t).squeeze(0)  # (n_quantiles, n_actions)
            # Transpose to (n_actions, n_quantiles)
            if out.dim() == 2 and out.shape[1] == 4:
                return out.transpose(0, 1).cpu().numpy()
            elif out.dim() == 2 and out.shape[0] == 4:
                return out.cpu().numpy()
            else:
                return out.view(4, -1).cpu().numpy()

    def predict(
        self,
        obs: np.ndarray,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, None]:
        """Selects action via hard action masking over quantile distribution."""
        quantiles = self.get_quantiles(obs)  # (4, N)

        if self.risk_metric == "cvar":
            # Conditional Value-at-Risk (mean of bottom alpha percent quantiles)
            k = max(1, int(quantiles.shape[1] * self.cvar_alpha))
            scores = np.mean(np.sort(quantiles, axis=1)[:, :k], axis=1)
        else:
            # Expected Q-value (mean across all quantiles)
            scores = np.mean(quantiles, axis=1)

        if action_masks is not None:
            mask_arr = np.array(action_masks, dtype=bool)
            scores[~mask_arr] = -1e9

        action = int(np.argmax(scores))
        return action, None
