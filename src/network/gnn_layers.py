"""Topology-Aware Ego-Graph Message Passing Neural Network (GNN) Feature Extractor
for Reinforcement Learning Evacuation Routing.

Natively processes 34-dimensional observation batches into relational graph embeddings
without external C++ graph library dependencies, ensuring high throughput on CPU and GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class EgoGraphGNN(BaseFeaturesExtractor):
    """Message-Passing Graph Neural Network encoder over the agent's local ego-network.

    Input observation: 34 dimensions:
      - 14 Root Node & Global features (coordinates, remaining distance, hazard, traffic)
      - 20 Candidate Successor features (4 candidate slots x 5 attributes each:
        [is_valid, dest_alignment_cos, dist_gain_norm, hazard_intensity, congestion])

    Message Passing:
      1. Candidate feature projection: (B, 4, 5) -> (B, 4, H)
      2. Root node projection: (B, 14) -> (B, H)
      3. Message Layer 1 (Candidates -> Root): Aggregates candidate state into root context.
      4. Message Layer 2 (Root -> Candidates): Context broadcast to candidate branches.
      5. Readout Layer (Global/Local Pooling): Mean and Max pooling over open candidates,
         concatenated with root embedding into a final normalized latent representation.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        hidden_dim: int = 64,
        num_candidates: int = 4,
        candidate_feature_dim: int = 5,
        num_layers: int = 2
    ):
        super().__init__(observation_space, features_dim)
        self.num_candidates = num_candidates
        self.candidate_feature_dim = candidate_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        root_in_dim = observation_space.shape[0] - (num_candidates * candidate_feature_dim)
        assert root_in_dim == 14, f"Expected 14 root features, got {root_in_dim}"

        # Projections
        self.root_proj = nn.Sequential(
            nn.Linear(root_in_dim, hidden_dim),
            nn.GELU()
        )

        self.cand_proj = nn.Sequential(
            nn.Linear(candidate_feature_dim, hidden_dim),
            nn.GELU()
        )

        # Message Passing Layer 1: Candidates to Root
        self.cand_to_root = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.root_update_1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )

        # Message Passing Layer 2: Root to Candidates
        self.root_to_cand = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.cand_update_2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )

        # Optional Layer 3: Inter-candidate cross-attention / aggregation
        if self.num_layers >= 3:
            self.cand_self_msg = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            )

        # Readout MLP with residual skip connection from raw observation features
        readout_dim = hidden_dim * 3 + observation_space.shape[0]  # root + mean_pooled + max_pooled + raw_obs
        self.readout = nn.Sequential(
            nn.Linear(readout_dim, features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through message-passing GNN layers.

        :param observations: (B, 34) tensor of normalized state features
        :return: (B, features_dim) topology-aware latent feature vector
        """
        batch_size = observations.shape[0]

        # 1. Split root features (first 14) and candidate features (next 20)
        root_feats = observations[:, :14]
        cand_raw = observations[:, 14:].view(batch_size, self.num_candidates, self.candidate_feature_dim)

        # Candidate validity mask: is_valid is slot attribute 0
        valid_mask = cand_raw[:, :, 0:1]  # (B, 4, 1)

        # 2. Initial embeddings
        h_root = self.root_proj(root_feats)  # (B, hidden_dim)
        h_cand = self.cand_proj(cand_raw)    # (B, 4, hidden_dim)

        # 3. Message Passing: Candidates -> Root
        cand_msgs = self.cand_to_root(h_cand) * valid_mask  # zero out invalid actions
        # Masked average pooling across valid candidate branches
        valid_counts = torch.clamp(valid_mask.sum(dim=1), min=1.0)  # (B, 1)
        pooled_cand_msg = cand_msgs.sum(dim=1) / valid_counts         # (B, hidden_dim)

        h_root_1 = self.root_update_1(torch.cat([h_root, pooled_cand_msg], dim=-1))  # (B, hidden_dim)

        # 4. Message Passing: Root -> Candidates
        root_broadcast = self.root_to_cand(h_root_1).unsqueeze(1).expand(-1, self.num_candidates, -1)
        h_cand_1 = self.cand_update_2(torch.cat([h_cand, root_broadcast], dim=-1)) * valid_mask

        # 5. Optional Layer 3 refinement
        if self.num_layers >= 3:
            h_cand_1 = h_cand_1 + self.cand_self_msg(h_cand_1) * valid_mask

        # 6. Readout: Aggregate candidate representations and root context
        mean_pooled = (h_cand_1 * valid_mask).sum(dim=1) / valid_counts
        # For max-pooling, set invalid candidates to a very low value
        masked_cand = h_cand_1.clone()
        masked_cand[valid_mask.expand(-1, -1, self.hidden_dim) == 0] = -1e4
        max_pooled = torch.max(masked_cand, dim=1)[0]
        # Replace if all were invalid
        max_pooled = torch.where(max_pooled < -1e3, torch.zeros_like(max_pooled), max_pooled)

        combined = torch.cat([h_root_1, mean_pooled, max_pooled, observations], dim=-1)
        latent = self.readout(combined)                                   # (B, features_dim)
        return latent
