import sys
import os
sys.path.insert(0, r"c:\Users\venka\OneDrive\Desktop\FinalYear")
import pandas as pd
print("1. Pandas:", pd.__version__, flush=True)

# Append .venv to sys.path after standard site-packages
sys.path.append(r"c:\Users\venka\OneDrive\Desktop\FinalYear\.venv\Lib\site-packages")

import torch
print("2. PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())

import networkx as nx
print("3. NetworkX:", nx.__version__)

import gymnasium as gym
print("4. Gymnasium:", gym.__version__)

import traci
print("5. TraCI:", traci.__file__)

from src.agents.marl_six import OfflineOnlineMARLModel
print("6. OfflineOnlineMARLModel successfully imported!")

# Test loading the checkpoint
checkpoint_path = r"models\marl_six\offline_online\seed_42\best_model.pt"
model = OfflineOnlineMARLModel(n_agents=4, obs_dim=34, act_dim=4)
model.load(checkpoint_path)
print("7. MODEL CHECKPOINT LOADED SUCCESSFULLY!")
