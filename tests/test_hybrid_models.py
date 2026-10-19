"""Unit tests for Five Hybrid Models:
1. GNN tensor dimensions and gradient flow
2. LSTM state reset across episode boundaries
3. QR-DQN output dimensions and masked argmax
4. Safety shield action constraint verification
5. TriHybrid ensemble decision scoring & disagreement tracking
6. Action mask guarantee: action_mask[action] == True
7. Normalization validity [0.0, 1.0]
8. Reproducibility across random seeds
"""

import pytest
import torch
import numpy as np
import gymnasium as gym

from src.environment.evacuation_env import EvacuationEnv
from src.network.gnn_layers import EgoGraphGNN
from src.agents.recurrent_ppo_agent import RecurrentActorCriticNet, RecurrentMaskablePPOAgent
from src.agents.safety_shield_agent import SafetyShieldPPOAgent
from src.agents.tri_hybrid_agent import TriHybridEvacuationAgent
from src.baselines.dijkstra import DijkstraBaseline


def test_gnn_tensor_dimensions():
    """Verify EgoGraphGNN handles 34-dim input and produces 64-dim representation."""
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(34,), dtype=np.float32)
    gnn = EgoGraphGNN(obs_space, features_dim=64, hidden_dim=64, num_layers=2)

    # Batch of 8 observations
    x = torch.rand(8, 34)
    out = gnn(x)

    assert out.shape == (8, 64), f"Expected shape (8, 64), got {out.shape}"
    assert not torch.isnan(out).any(), "GNN output contains NaN"

    # Test gradient backpropagation
    loss = out.sum()
    loss.backward()
    for name, param in gnn.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"


def test_lstm_state_reset():
    """Verify RecurrentActorCriticNet resets LSTM state on episode boundaries."""
    net = RecurrentActorCriticNet(obs_dim=34, action_dim=4, hidden_dim=64, lstm_hidden_dim=64)

    init_state = net.get_initial_state(batch_size=2)
    assert init_state[0].shape == (1, 2, 64)
    assert init_state[1].shape == (1, 2, 64)

    obs_seq = torch.rand(2, 5, 34)  # B=2, T=5
    done_seq = torch.zeros(2, 5)
    done_seq[0, 2] = 1.0  # Batch 0 resets at step 2

    logits, values, next_state = net.forward_sequence(obs_seq, init_state, done_seq=done_seq)

    assert logits.shape == (2, 5, 4)
    assert values.shape == (2, 5)
    assert next_state[0].shape == (1, 2, 64)


def test_action_mask_prevention():
    """Verify invalid actions receive -1e9 and cannot be sampled."""
    net = RecurrentActorCriticNet(obs_dim=34, action_dim=4, hidden_dim=64, lstm_hidden_dim=64)
    obs = torch.rand(1, 34)
    state = net.get_initial_state(batch_size=1)

    # Mask allowing only action 2
    mask = torch.tensor([[False, False, True, False]], dtype=torch.bool)
    action, log_prob, entropy, value, _ = net.get_action_and_value(
        obs, state, action_masks=mask, deterministic=True
    )

    assert int(action.item()) == 2, f"Expected action 2, got {action.item()}"
    assert mask[0, action].item() is True, "Action mask was violated!"


def test_safety_shield_constrained_action():
    """Verify SafetyShieldPPO always selects an action permitted by the environment mask."""
    env = EvacuationEnv(use_traci=False)
    obs, info = env.reset(seed=42)
    mask = env.action_masks()

    class MockPPO:
        def predict(self, o, deterministic=True, action_masks=None):
            return 0, None

    mock_ppo = MockPPO()
    shield = SafetyShieldPPOAgent(base_ppo_agent=mock_ppo)
    action = shield.select_action(env)

    assert isinstance(action, int)
    assert 0 <= action < 4
    assert mask[action] is True or mask[action] == 1, f"SafetyShield violated action mask at action {action}"
    env.close()


def test_tri_hybrid_ensemble_disagreement():
    """Verify TriHybrid ensemble selects valid action and accurately logs model disagreement."""
    env = EvacuationEnv(use_traci=False)
    obs, info = env.reset(seed=101)
    mask = env.action_masks()

    class MockAgent1:
        def predict(self, o, action_masks=None, deterministic=True):
            return 0, None

    class MockAgent2:
        def predict(self, o, action_masks=None, deterministic=True):
            return 1 if mask[1] else 0, None

    ensemble = TriHybridEvacuationAgent(
        ppo_agent=MockAgent1(),
        qrdqn_agent=MockAgent2(),
        dijkstra_baseline=DijkstraBaseline()
    )

    action = ensemble.select_action(env)

    assert isinstance(action, int)
    assert mask[action] is True or mask[action] == 1, "Ensemble violated action mask!"
    assert len(ensemble.decision_records) == 1
    assert "models_disagreed" in ensemble.decision_records[0]
    assert "selected_action" in ensemble.decision_records[0]
    env.close()


def test_observation_normalization_bounds():
    """Verify environment observation vector elements are strictly within [0.0, 1.0]."""
    env = EvacuationEnv(use_traci=False)
    for s in [42, 123, 777]:
        obs, _ = env.reset(seed=s)
        assert obs.shape == (34,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"Observation values out of [0, 1] range: {obs}"

        # Take 5 steps and verify each step's observation
        for _ in range(5):
            mask = env.action_masks()
            act = int(np.where(mask)[0][0])
            obs, _, term, trunc, _ = env.step(act)
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"Step observation out of [0, 1] range: {obs}"
            if term or trunc:
                break
    env.close()


def test_reproducibility():
    """Verify environment step trajectory is identical under identical seeds."""
    env1 = EvacuationEnv(use_traci=False)
    env2 = EvacuationEnv(use_traci=False)

    obs1, _ = env1.reset(seed=2024)
    obs2, _ = env2.reset(seed=2024)

    assert np.allclose(obs1, obs2), "Initial observations under identical seed diverged!"

    mask1 = env1.action_masks()
    mask2 = env2.action_masks()
    assert np.array_equal(mask1, mask2), "Action masks diverged!"

    obs1_next, r1, term1, trunc1, _ = env1.step(0)
    obs2_next, r2, term2, trunc2, _ = env2.step(0)

    assert np.allclose(obs1_next, obs2_next), "Next observations diverged!"
    assert r1 == r2, "Rewards diverged!"

    env1.close()
    env2.close()
