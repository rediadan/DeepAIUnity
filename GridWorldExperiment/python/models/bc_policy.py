from __future__ import annotations

import torch
from torch import nn

from models.cnn_encoder import CNNEncoder


class BCPolicy(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        grid_size: int = 24,
        grid_height: int | None = None,
        grid_width: int | None = None,
        action_count: int = 4,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.encoder = CNNEncoder(
            input_channels=input_channels,
            grid_size=grid_size,
            grid_height=grid_height,
            grid_width=grid_width,
            feature_dim=feature_dim,
        )
        self.policy_head = nn.Linear(feature_dim, action_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.policy_head(features)
