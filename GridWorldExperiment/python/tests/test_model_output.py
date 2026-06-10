from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from models.bc_policy import BCPolicy
from models.dqn import CNNDQN, MLPDQN


def test_bc_policy_output_shape():
    model = BCPolicy(input_channels=10, grid_size=24, action_count=4)
    x = torch.zeros((2, 10, 24, 24), dtype=torch.float32)
    assert model(x).shape == (2, 4)


def test_cnn_dqn_output_shape():
    model = CNNDQN(input_channels=10, grid_size=24, action_count=4)
    x = torch.zeros((2, 10, 24, 24), dtype=torch.float32)
    assert model(x).shape == (2, 4)


def test_mlp_dqn_output_shape():
    model = MLPDQN(input_channels=10, grid_size=24, action_count=4)
    x = torch.zeros((2, 10, 24, 24), dtype=torch.float32)
    assert model(x).shape == (2, 4)
