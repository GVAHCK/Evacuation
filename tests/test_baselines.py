"""Unit tests for Dijkstra, A*, and Random baselines."""

import pytest
from src.environment.evacuation_env import EvacuationEnv
from src.baselines.dijkstra import DijkstraBaseline
from src.baselines.astar import AStarBaseline
from src.baselines.random_routing import RandomValidRoutingBaseline


def test_dijkstra_baseline():
    """Verify Dijkstra baseline selects valid action and runs episode."""
    env = EvacuationEnv(use_traci=False)
    dijkstra = DijkstraBaseline()
    result = dijkstra.run_episode(env)

    assert "model" in result
    assert result["model"] == "Dijkstra"
    assert "success" in result
    assert "total_reward" in result
    assert result["steps"] > 0
    env.close()


def test_astar_baseline():
    """Verify A* baseline selects valid action and runs episode."""
    env = EvacuationEnv(use_traci=False)
    astar = AStarBaseline()
    result = astar.run_episode(env)

    assert "model" in result
    assert result["model"] == "A*"
    assert "success" in result
    assert "total_reward" in result
    assert result["steps"] > 0
    env.close()


def test_random_baseline():
    """Verify Random baseline selects valid action and runs episode."""
    env = EvacuationEnv(use_traci=False)
    rand_bot = RandomValidRoutingBaseline(seed=42)
    result = rand_bot.run_episode(env)

    assert "model" in result
    assert result["model"] == "Random"
    assert "success" in result
    assert "total_reward" in result
    assert result["steps"] > 0
    env.close()
