import os
import time
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch

from backend.app.core.config import settings
from backend.app.core.logging import logger
from src.agents.marl_six import OfflineOnlineMARLModel

class ModelManager:
    """Thread-safe Singleton Model Manager for the production Offline-to-Online MARL engine."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_name = "offline_to_online"
        self.model_family = "Conservative Q-Learning (CQL) Offline-to-Online MARL"
        self.status = "uninitialized"  # ready, degraded, unavailable
        self.model: Optional[OfflineOnlineMARLModel] = None
        self.device = "cpu"
        self.checkpoint_path = ""
        self.checkpoint_size_kb = 0.0
        self.n_agents = 4
        self.obs_dim = 34
        self.act_dim = 4
        self.parameter_count = 327204
        self.warmup_completed = False
        self._inference_lock = threading.Lock()

        self._initialize_and_validate()
        self._initialized = True

    def _initialize_and_validate(self):
        """Locates, validates, and warms the Offline-to-Online MARL model."""
        target_path = settings.MODEL_CHECKPOINT
        if not os.path.isabs(target_path):
            target_path = os.path.abspath(target_path)

        self.checkpoint_path = target_path

        # 1. Verify checkpoint existence
        if not os.path.isfile(target_path):
            logger.error(f"Offline-to-Online model checkpoint not found at {target_path}")
            self.status = "unavailable"
            return

        try:
            self.checkpoint_size_kb = os.path.getsize(target_path) / 1024.0
            if self.checkpoint_size_kb <= 0:
                logger.error(f"Checkpoint file {target_path} is empty (0 KB).")
                self.status = "unavailable"
                return

            # 2. Select Device
            requested_device = settings.MODEL_DEVICE
            if requested_device.startswith("cuda") and torch.cuda.is_available():
                self.device = requested_device
            else:
                self.device = "cpu"

            logger.info(f"Loading Offline-to-Online MARL model on device '{self.device}'...")

            # 3. Instantiate model architecture
            self.model = OfflineOnlineMARLModel(
                n_agents=self.n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                device=self.device
            )

            # 4. Load checkpoint
            self.model.load(target_path)

            # 5. Model Warmup & Verification
            self._warmup_and_test()
            self.status = "ready"
            logger.info(f"Offline-to-Online MARL model successfully validated & warmed. Status: READY.")

        except Exception as e:
            logger.error(f"Failed to load or validate Offline-to-Online model: {e}", exc_info=True)
            self.status = "unavailable"
            self.model = None

    def _warmup_and_test(self):
        """Runs test inference to warm CUDA kernels and verify action masking."""
        if not self.model:
            raise RuntimeError("Model is None, cannot run warmup.")

        dummy_obs = [np.random.uniform(0.0, 1.0, size=(self.obs_dim,)).astype(np.float32) for _ in range(self.n_agents)]
        # Mask: only action 1 is valid
        dummy_masks = [np.array([False, True, False, False], dtype=bool) for _ in range(self.n_agents)]

        with self._inference_lock:
            actions = self.model.predict(dummy_obs, dummy_masks, deterministic=True)
            for idx, a in enumerate(actions):
                if a != 1:
                    raise AssertionError(f"Model selected action {a}, but action mask constrained to action 1.")

        self.warmup_completed = True
        logger.info("Model warmup and hard action-masking verification PASSED.")

    def predict(
        self,
        obs_n: List[np.ndarray],
        action_masks_n: List[np.ndarray],
        deterministic: bool = True
    ) -> Tuple[List[int], float]:
        """Performs multi-agent inference with hard action masking and returns actions + latency (ms)."""
        if self.status != "ready" or self.model is None:
            raise RuntimeError(f"Offline-to-Online model is not ready (Status: {self.status}).")

        t0 = time.perf_counter()
        with self._inference_lock:
            actions = self.model.predict(obs_n, action_masks_n, deterministic=deterministic)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return actions, latency_ms

    def get_metadata(self) -> Dict[str, Any]:
        """Returns metadata for API inspection and health endpoints."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "status": self.status,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_exists": os.path.isfile(self.checkpoint_path),
            "checkpoint_size_kb": self.checkpoint_size_kb,
            "device": str(self.device),
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "parameter_count": self.parameter_count,
            "action_masking_supported": True,
            "warmup_completed": self.warmup_completed
        }

model_manager = ModelManager()
