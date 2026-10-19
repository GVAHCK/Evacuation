"""Unit tests for the EvacuationEnv Gymnasium environment."""

import pytest
import numpy as np
from src.environment.evacuation_env import EvacuationEnv


def test_env_initialization():
    """Verify EvacuationEnv initializes with correct spaces."""
    env = EvacuationEnv(use_traci=False)
    assert env.action_space.n == 4
    assert env.observation_space.shape == (34,)
    assert env.observation_space.dtype == np.float32
    env.close()


def test_env_reset():
    """Verify reset returns valid normalized observation and info dictionary."""
    env = EvacuationEnv(use_traci=False)
    obs, info = env.reset(seed=42)

    assert obs.shape == (34,)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert "scenario_id" in info
    assert "hazard_exposure_index" in info
    assert "action_mask" in info
    assert len(info["action_mask"]) == 4
    env.close()


def test_env_step():
    """Verify step transitions, computes rewards, and produces reward breakdown."""
    env = EvacuationEnv(use_traci=False)
    obs, info = env.reset(seed=42)

    # Pick first action from mask
    mask = env.action_masks()
    action = int(np.where(mask)[0][0]) if np.any(mask) else 0

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (34,)
    assert np.all(next_obs >= 0.0) and np.all(next_obs <= 1.0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # Check reward decomposition keys
    assert "progress_reward" in step_info
    assert "hazard_penalty" in step_info
    assert "closure_penalty" in step_info
    assert "congestion_penalty" in step_info
    assert "total_reward" in step_info

    env.close()


def test_action_masking():
    """Verify action masking properly masks candidate transitions."""
    env = EvacuationEnv(use_traci=False)
    obs, info = env.reset(seed=123)
    mask = env.action_masks()

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert len(mask) == env.max_neighbors
    # At least one action should be permitted
    assert np.any(mask)
    env.close()
