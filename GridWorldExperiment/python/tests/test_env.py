from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.grid_env import GridWorldEnv


def test_reset_is_deterministic_for_same_seed():
    env = GridWorldEnv(PROJECT_ROOT / "configs" / "env_config.json")
    first = env.reset(seed=123)
    second = env.reset(seed=123)
    assert np.array_equal(first, second)


def test_step_returns_expected_fields():
    env = GridWorldEnv(PROJECT_ROOT / "configs" / "env_config.json")
    env.reset(seed=42)
    result = env.step(0)
    assert result.state.shape == (10, 24, 24)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert "event" in result.info
    assert "score" in result.info
    assert "hit_trap" in result.info
