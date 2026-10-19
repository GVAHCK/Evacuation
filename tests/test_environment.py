"""Unit tests to verify environment setup, dependencies, configurations, and SUMO binaries."""

import os
import sys
import yaml
import pytest


def test_core_imports():
    """Verify that all core reinforcement learning, spatial, and simulation libraries import."""
    import torch
    import gymnasium
    import stable_baselines3
    import networkx
    import osmnx
    import traci
    import sumolib
    import scipy
    import pandas
    import numpy

    assert torch.__version__ is not None
    assert gymnasium.__version__ is not None
    assert stable_baselines3.__version__ is not None
    assert sumolib.__version__ is not None


def test_pytorch_cuda():
    """Verify PyTorch detects CUDA device."""
    import torch

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA Device: {device_name}")
        assert "RTX" in device_name or "GeForce" in device_name or len(device_name) > 0


def test_sumo_binaries():
    """Verify that SUMO and netconvert binaries exist and can be invoked."""
    import sumolib

    sumo_bin = sumolib.checkBinary("sumo")
    netconvert_bin = sumolib.checkBinary("netconvert")

    assert os.path.isfile(sumo_bin), f"sumo binary not found at {sumo_bin}"
    assert os.path.isfile(netconvert_bin), f"netconvert binary not found at {netconvert_bin}"


def test_config_files():
    """Verify that YAML configuration files exist and parse cleanly."""
    config_files = [
        "configs/default.yaml",
        "configs/ppo.yaml",
        "configs/dqn.yaml"
    ]
    for cfg_path in config_files:
        assert os.path.isfile(cfg_path), f"Missing config: {cfg_path}"
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), f"Config {cfg_path} did not parse into a dictionary"
            assert len(data) > 0, f"Config {cfg_path} is empty"


def test_reward_config_structure():
    """Verify reward configuration keys in default.yaml."""
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "environment" in cfg
    assert "reward" in cfg["environment"]
    reward_cfg = cfg["environment"]["reward"]

    expected_keys = [
        "progress_weight",
        "arrival_bonus",
        "hazard_penalty_weight",
        "closure_penalty",
        "congestion_penalty_weight",
        "delay_penalty",
        "dead_end_penalty",
        "timeout_penalty"
    ]
    for k in expected_keys:
        assert k in reward_cfg, f"Reward config missing key: {k}"
        assert isinstance(reward_cfg[k], (int, float))
