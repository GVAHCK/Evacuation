"""Unit and integration tests for the Six MARL Models and Multi-Agent Evacuation Environment."""

import os
import shutil
import tempfile
import numpy as np
import pytest
import torch

from src.envs.multi_agent_evacuation_env import MultiAgentEvacuationEnv
from src.agents.marl_six import (
    MADDPGModel,
    RSMADDPGModel,
    OfflineOnlineMARLModel,
    MAGRLModel,
    GraphSAGEMAPPOModel,
    HARLModel
)


@pytest.fixture(scope="module")
def env_and_state():
    """Create a shared multi-agent environment instance for testing."""
    env = MultiAgentEvacuationEnv(network_size="small", n_agents=4, seed=42)
    obs_n, info = env.reset(seed=42)
    masks_n = env.get_action_masks()
    state = env.get_global_state()
    yield env, obs_n, masks_n, state
    env.close()


def test_multi_agent_environment_lifecycle(env_and_state):
    """Verify multi-agent environment observation dimensions, masking, and stepping."""
    env, obs_n, masks_n, state = env_and_state

    assert len(obs_n) == 4, "Must have 4 agents"
    assert len(masks_n) == 4, "Must have 4 action masks"

    for i in range(4):
        assert obs_n[i].shape == (34,), f"Agent {i} obs shape must be 34, got {obs_n[i].shape}"
        assert masks_n[i].shape == (4,), f"Agent {i} mask shape must be 4, got {masks_n[i].shape}"
        assert masks_n[i].sum() >= 1, f"Agent {i} must have at least 1 valid successor"

    # Step with valid actions
    valid_actions = [int(np.random.choice(np.where(masks_n[i])[0])) for i in range(4)]
    next_obs_n, rewards, dones, truncated, info = env.step(valid_actions)

    assert len(next_obs_n) == 4
    assert len(rewards) == 4
    assert "mean_hei" in info
    assert "inter_agent_congestion" in info
    assert "bottleneck_conflicts" in info
    assert "route_overlap_ratio" in info
    assert "coordination_efficiency" in info


def test_maddpg_lifecycle(env_and_state):
    """Test MADDPG forward, action selection, masking, replay buffer, update, save/load."""
    _, obs_n, masks_n, state = env_and_state
    model = MADDPGModel(n_agents=4, obs_dim=34, act_dim=4, actor_hidden_dim=64, critic_hidden_dim=64, device="cpu")

    actions = model.select_actions(obs_n, masks_n, explore=True)
    for i in range(4):
        assert masks_n[i][actions[i]], f"MADDPG Agent {i} selected masked action {actions[i]}"

    next_state = state + 0.01
    next_obs_n = [o + 0.01 for o in obs_n]
    next_masks_n = masks_n

    # Fill small buffer
    for _ in range(60):
        model.buffer.add(
            obs_n, state, actions, [0.5, 0.5, 0.5, 0.5],
            next_obs_n, next_state, [False, False, False, False],
            masks_n, next_masks_n
        )

    loss_dict = model.update(batch_size=32)
    assert "critic_loss" in loss_dict
    assert "actor_loss" in loss_dict

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "maddpg.pt")
        model.save(save_path)
        assert os.path.isfile(save_path)
        model.load(save_path)


def test_rs_maddpg_lifecycle(env_and_state):
    """Test RS-MADDPG with MA-PER, GARC relational critic, and hazard barrier shaping."""
    _, obs_n, masks_n, state = env_and_state
    model = RSMADDPGModel(n_agents=4, obs_dim=34, act_dim=4, actor_hidden_dim=64, critic_hidden_dim=64, device="cpu")

    actions = model.select_actions(obs_n, masks_n, explore=True)
    for i in range(4):
        assert masks_n[i][actions[i]], f"RS-MADDPG Agent {i} selected masked action {actions[i]}"

    next_state = state + 0.01
    next_obs_n = [o + 0.01 for o in obs_n]
    next_masks_n = masks_n

    for _ in range(60):
        model.buffer.add(
            obs_n, state, actions, [0.5, 0.5, 0.5, 0.5],
            next_obs_n, next_state, [False, False, False, False],
            masks_n, next_masks_n,
            hazard_dists=[100.0, 150.0, 80.0, 200.0],
            next_hazard_dists=[110.0, 160.0, 85.0, 210.0]
        )

    loss_dict = model.update(batch_size=32)
    assert "critic_loss" in loss_dict
    assert "actor_loss" in loss_dict

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "rs_maddpg.pt")
        model.save(save_path)
        assert os.path.isfile(save_path)
        model.load(save_path)


def test_offline_online_lifecycle(env_and_state):
    """Test Offline-to-Online MARL with Offline CQL and KL-regularized online fine-tuning."""
    _, obs_n, masks_n, state = env_and_state
    model = OfflineOnlineMARLModel(n_agents=4, obs_dim=34, act_dim=4, hidden_dim=64, device="cpu")

    actions = model.select_actions(obs_n, masks_n, deterministic=True)
    for i in range(4):
        assert masks_n[i][actions[i]], f"Offline-Online Agent {i} selected masked action {actions[i]}"

    next_state = state + 0.01
    next_obs_n = [o + 0.01 for o in obs_n]
    next_masks_n = masks_n

    for _ in range(60):
        model.store_offline_transition(
            obs_n, state, actions, [0.5, 0.5, 0.5, 0.5],
            next_obs_n, next_state, [False, False, False, False],
            masks_n, next_masks_n
        )

    cql_losses = model.update_offline(batch_size=32)
    assert "critic_loss" in cql_losses
    assert "cql_penalty" in cql_losses

    model.freeze_for_online()

    for _ in range(60):
        model.store_online_transition(
            obs_n, state, actions, [0.5, 0.5, 0.5, 0.5],
            next_obs_n, next_state, [False, False, False, False],
            masks_n, next_masks_n
        )

    online_losses = model.update_online(batch_size=32)
    assert "online_actor_loss" in online_losses
    assert "kl_loss" in online_losses

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "offline_online.pt")
        model.save(save_path)
        assert os.path.isfile(save_path)
        model.load(save_path)


def test_ma_grl_lifecycle(env_and_state):
    """Test MA-GRL with EgoGraphGNN decentralized actor and graph-structured critic."""
    _, obs_n, masks_n, state = env_and_state
    model = MAGRLModel(n_agents=4, obs_dim=34, act_dim=4, gnn_hidden_dim=32, critic_hidden_dim=64, device="cpu")

    actions = model.select_actions(obs_n, masks_n, explore=True)
    for i in range(4):
        assert masks_n[i][actions[i]], f"MA-GRL Agent {i} selected masked action {actions[i]}"

    next_state = state + 0.01
    next_obs_n = [o + 0.01 for o in obs_n]
    next_masks_n = masks_n

    for _ in range(60):
        model.buffer.add(
            obs_n, state, actions, [0.5, 0.5, 0.5, 0.5],
            next_obs_n, next_state, [False, False, False, False],
            masks_n, next_masks_n
        )

    losses = model.update(batch_size=32)
    assert "critic_loss" in losses
    assert "actor_loss" in losses

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "ma_grl.pt")
        model.save(save_path)
        assert os.path.isfile(save_path)
        model.load(save_path)


def test_graphsage_mappo_lifecycle(env_and_state):
    """Test GraphSAGE-MAPPO with inductive GraphSAGE neighbor aggregation and PPO clipping."""
    _, obs_n, masks_n, state = env_and_state
    model = GraphSAGEMAPPOModel(n_agents=4, obs_dim=34, act_dim=4, graphsage_hidden_dim=32, mlp_hidden_dim=64, ppo_epochs=2, device="cpu")

    actions, log_probs = model.select_actions(obs_n, masks_n, deterministic=False)
    for i in range(4):
        assert masks_n[i][actions[i]], f"GraphSAGE-MAPPO Agent {i} selected masked action {actions[i]}"

    val = model.get_value(state)
    assert isinstance(val, float)

    for _ in range(20):
        model.buffer.add(
            obs_n=obs_n,
            state=state,
            actions_n=actions,
            log_probs_n=log_probs,
            rewards_n=[0.5, 0.5, 0.5, 0.5],
            dones_n=[False, False, False, False],
            masks_n=masks_n,
            value=val
        )

    losses = model.update(next_state=state)
    assert "critic_loss" in losses
    assert "actor_loss" in losses

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "graphsage_mappo.pt")
        model.save(save_path)
        assert os.path.isfile(save_path)
        model.load(save_path)


def test_harl_lifecycle(env_and_state):
    """Test HARL with heterogeneous cohort actors and sequential HAPPO policy updates."""
    _, obs_n, masks_n, state = env_and_state
    model = HARLModel(n_agents=4, obs_dim=34, act_dim=4, actor_hidden_dim=64, critic_hidden_dim=64, ppo_epochs=2, device="cpu")

    actions, log_probs = model.select_actions(obs_n, masks_n, deterministic=False)
    for i in range(4):
        assert masks_n[i][actions[i]], f"HARL Agent {i} selected masked action {actions[i]}"

    val = model.get_value(state)
    assert isinstance(val, float)

    for _ in range(20):
        model.buffer.add(
            obs_n=obs_n,
            state=state,
            actions_n=actions,
            log_probs_n=log_probs,
            rewards_n=[0.5, 0.5, 0.5, 0.5],
            dones_n=[False, False, False, False],
            masks_n=masks_n,
            value=val
        )

    losses = model.update(next_state=state)
    assert "critic_loss" in losses
    assert "actor_loss" in losses

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "harl.pt")
        model.save(save_path)
        assert os.path.isfile(save_path)
        model.load(save_path)
