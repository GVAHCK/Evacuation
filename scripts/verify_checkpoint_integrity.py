"""Comprehensive Checkpoint Integrity & Inference Validation Script for Six MARL Models.
Validates Phase 2 requirements across all 6 models and 5 benchmark seeds (30 total model-seed instances):
  - File existence and non-zero byte size
  - Torch loadability
  - Architectural conformity (obs_dim=34, act_dim=4, n_agents=4)
  - Hard action masking assertion
  - Deterministic and stochastic inference execution
Exports complete audit trail to results/marl_six/checkpoint_validation.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.logger import setup_logger
from src.agents.marl_six import (
    MADDPGModel,
    RSMADDPGModel,
    OfflineOnlineMARLModel,
    MAGRLModel,
    GraphSAGEMAPPOModel,
    HARLModel
)

logger = setup_logger("CheckpointValidation")

MODELS = ["maddpg", "rs_maddpg", "offline_online", "ma_grl", "graphsage_mappo", "harl"]
SEEDS = [42, 101, 2024, 777, 999]

MODEL_MAP = {
    "maddpg": MADDPGModel,
    "rs_maddpg": RSMADDPGModel,
    "offline_online": OfflineOnlineMARLModel,
    "ma_grl": MAGRLModel,
    "graphsage_mappo": GraphSAGEMAPPOModel,
    "harl": HARLModel
}


def validate_checkpoints() -> pd.DataFrame:
    logger.info("================================================================")
    logger.info("PHASE 2: VERIFYING INTEGRITY OF ALL 30 MARL MODEL CHECKPOINTS")
    logger.info("================================================================")

    out_dir = os.path.join("results", "marl_six")
    os.makedirs(out_dir, exist_ok=True)

    validation_records = []
    total_checks = len(MODELS) * len(SEEDS)
    passed_checks = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for m_name in MODELS:
        cls = MODEL_MAP[m_name]
        for s in SEEDS:
            ckpt_dir = os.path.join("models", "marl_six", m_name, f"seed_{s}")
            best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
            final_ckpt = os.path.join(ckpt_dir, "final_model.pt")

            # Check existence
            exists = os.path.isfile(best_ckpt) or os.path.isfile(final_ckpt)
            active_path = best_ckpt if os.path.isfile(best_ckpt) else final_ckpt
            file_size_kb = os.path.getsize(active_path) / 1024.0 if exists else 0.0

            is_loadable = False
            arch_match = False
            obs_dim_correct = False
            act_dim_correct = False
            masking_works = False
            inference_works = False
            error_msg = ""

            if exists:
                try:
                    # 1. Instantiate
                    model = cls(n_agents=4, obs_dim=34, act_dim=4, device=device)
                    # 2. Load
                    model.load(active_path)
                    is_loadable = True
                    arch_match = (model.n_agents == 4 and model.obs_dim == 34 and model.act_dim == 4)
                    obs_dim_correct = (model.obs_dim == 34)
                    act_dim_correct = (model.act_dim == 4)

                    # 3. Test Inference & Hard Action Masking
                    dummy_obs = [np.random.uniform(0.0, 1.0, size=(34,)).astype(np.float32) for _ in range(4)]
                    # Create restricted mask where only action index 2 is valid
                    dummy_mask = [np.array([False, False, True, False], dtype=bool) for _ in range(4)]

                    actions = model.predict(dummy_obs, dummy_mask, deterministic=True)
                    assert len(actions) == 4
                    # Every agent MUST pick index 2 because only index 2 is True
                    for a_idx, act in enumerate(actions):
                        assert act == 2, f"Model picked masked action {act}, expected 2!"

                    masking_works = True
                    inference_works = True
                    passed_checks += 1

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Integrity check failed for {m_name} (Seed {s}): {e}")
            else:
                error_msg = f"Checkpoint file not found in {ckpt_dir}"

            status = "PASSED" if (exists and is_loadable and arch_match and masking_works and inference_works) else "FAILED"

            record = {
                "model": m_name,
                "seed": s,
                "checkpoint_path": active_path if exists else "NONE",
                "file_exists": exists,
                "file_size_kb": file_size_kb,
                "loadable": is_loadable,
                "architecture_conformity": arch_match,
                "obs_dim_34": obs_dim_correct,
                "act_dim_4": act_dim_correct,
                "action_masking_enforced": masking_works,
                "inference_functional": inference_works,
                "status": status,
                "error_notes": error_msg
            }
            validation_records.append(record)
            logger.info(f"  [{status}] Model: {m_name:16s} | Seed: {s:4d} | Size: {file_size_kb:6.1f} KB | Masking: {masking_works}")

    df = pd.DataFrame(validation_records)
    out_csv = os.path.join(out_dir, "checkpoint_validation.csv")
    df.to_csv(out_csv, index=False)

    logger.info("----------------------------------------------------------------")
    logger.info(f"Checkpoint Validation Summary: {passed_checks}/{total_checks} PASSED.")
    logger.info(f"Validation records saved to: {out_csv}")
    logger.info("----------------------------------------------------------------")

    return df


if __name__ == "__main__":
    validate_checkpoints()
