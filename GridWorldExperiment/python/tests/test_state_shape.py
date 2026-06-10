from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.grid_env import GridWorldEnv


def test_state_shape_and_dtype():
    env = GridWorldEnv(PROJECT_ROOT / "configs" / "env_config.json")
    state = env.reset(seed=999)
    assert state.shape == (10, 24, 24)
    assert str(state.dtype) == "float32"


def test_state_has_single_player_cell():
    env = GridWorldEnv(PROJECT_ROOT / "configs" / "env_config.json")
    state = env.reset(seed=999)
    assert state[0].sum() == 1.0


def test_dynamic_observation_channels_are_present():
    env = GridWorldEnv(PROJECT_ROOT / "configs" / "env_config.json")
    state = env.reset(seed=999)
    assert state[7].sum() >= 1.0
    assert state[8].sum() >= 1.0
    assert state[9].sum() == 1.0
