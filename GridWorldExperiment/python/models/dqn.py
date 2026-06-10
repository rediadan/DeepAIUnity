from __future__ import annotations

import torch
from torch import nn

from models.cnn_encoder import CNNEncoder


class CNNDQN(nn.Module):
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
        self.grid_height = grid_size if grid_height is None else grid_height
        self.grid_width = grid_size if grid_width is None else grid_width
        self.encoder = CNNEncoder(
            input_channels=input_channels,
            grid_size=grid_size,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            feature_dim=feature_dim,
        )
        self.q_head = nn.Linear(feature_dim, action_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.q_head(features)


class MLPDQN(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        grid_size: int = 24,
        grid_height: int | None = None,
        grid_width: int | None = None,
        action_count: int = 4,
        hidden_size: int = 256,
    ):
        super().__init__()
        self.grid_height = grid_size if grid_height is None else grid_height
        self.grid_width = grid_size if grid_width is None else grid_width
        input_size = input_channels * self.grid_height * self.grid_width
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
