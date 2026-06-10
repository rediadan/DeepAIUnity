from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 10,
        grid_size: int = 24,
        grid_height: int | None = None,
        grid_width: int | None = None,
        feature_dim: int = 256,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.grid_height = grid_size if grid_height is None else grid_height
        self.grid_width = grid_size if grid_width is None else grid_width
        self.feature_dim = feature_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
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
