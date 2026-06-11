from __future__ import annotations

import torch
from torch import nn

from models.cnn_encoder import CNNEncoder


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        group_count = min(8, out_channels)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(group_count, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count, out_channels),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(group_count, out_channels),
            )
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.skip(x))


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


class ResNetDQN(nn.Module):
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
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, stride=2),
            nn.AdaptiveAvgPool2d((4, 6)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 6, feature_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(feature_dim, action_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.q_head(features)


class NatureDQNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        grid_size: int = 24,
        grid_height: int | None = None,
        grid_width: int | None = None,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.grid_height = grid_size if grid_height is None else grid_height
        self.grid_width = grid_size if grid_width is None else grid_width
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 6, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.conv(x))


class NatureDQN(nn.Module):
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
        self.encoder = NatureDQNEncoder(
            input_channels=input_channels,
            grid_size=grid_size,
            grid_height=grid_height,
            grid_width=grid_width,
            feature_dim=feature_dim,
        )
        self.q_head = nn.Linear(feature_dim, action_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.q_head(features)


class DuelingHead(nn.Module):
    def __init__(self, feature_dim: int, action_count: int):
        super().__init__()
        self.value = nn.Linear(feature_dim, 1)
        self.advantage = nn.Linear(feature_dim, action_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DuelingCNNDQN(nn.Module):
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
        self.q_head = DuelingHead(feature_dim, action_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.q_head(features)


class DuelingNatureDQN(nn.Module):
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
        self.encoder = NatureDQNEncoder(
            input_channels=input_channels,
            grid_size=grid_size,
            grid_height=grid_height,
            grid_width=grid_width,
            feature_dim=feature_dim,
        )
        self.q_head = DuelingHead(feature_dim, action_count)

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
