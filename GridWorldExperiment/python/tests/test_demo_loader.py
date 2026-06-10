from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from data.demo_loader import split_by_episode, transitions_to_arrays


def test_transitions_to_arrays_shapes():
    state = [0.0] * (10 * 24 * 24)
    rows = [
        {
            "episode": 0,
            "step": 0,
            "state": state,
            "action": 1,
            "reward": 0.5,
            "next_state": state,
            "done": False,
        },
        {
            "episode": 1,
            "step": 0,
            "state": state,
            "action": 2,
            "reward": -1.0,
            "next_state": state,
            "done": True,
        },
    ]
    arrays = transitions_to_arrays(rows, (10, 24, 24))
    assert arrays["states"].shape == (2, 10, 24, 24)
    assert arrays["actions"].tolist() == [1, 2]


def test_split_by_episode_keeps_episode_boundaries():
    state = [0.0] * (10 * 24 * 24)
    rows = []
    for episode in range(4):
        for step in range(2):
            rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "state": state,
                    "action": step % 4,
                    "reward": 0.0,
                    "next_state": state,
                    "done": step == 1,
                }
            )
    arrays = transitions_to_arrays(rows, (10, 24, 24))
    train, val = split_by_episode(arrays, validation_ratio=0.25, seed=1)
    assert train["actions"].shape[0] == 6
    assert val["actions"].shape[0] == 2
