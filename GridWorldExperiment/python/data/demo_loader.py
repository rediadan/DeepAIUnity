from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.env_config import load_config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exception:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exception
    return rows


def _collect_jsonl_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))
    return [input_path]


def load_transitions(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _collect_jsonl_files(input_path):
        rows.extend(_read_jsonl(path))
    return rows


def transitions_to_arrays(rows: list[dict[str, Any]], state_shape: tuple[int, int, int]):
    if not rows:
        raise ValueError("No transition rows were loaded.")

    expected_length = int(np.prod(state_shape))
    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []
    step_indices = []

    for index, row in enumerate(rows):
        state = np.asarray(row["state"], dtype=np.float32)
        next_state = np.asarray(row["next_state"], dtype=np.float32)
        if state.size != expected_length:
            raise ValueError(f"state length mismatch at row {index}: expected {expected_length}, got {state.size}")
        if next_state.size != expected_length:
            raise ValueError(f"next_state length mismatch at row {index}: expected {expected_length}, got {next_state.size}")

        states.append(state.reshape(state_shape))
        next_states.append(next_state.reshape(state_shape))
        actions.append(int(row["action"]))
        rewards.append(float(row["reward"]))
        dones.append(bool(row["done"]))
        episode_ids.append(int(row.get("episode", 0)))
        step_indices.append(int(row.get("step", index)))

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
    }


def split_by_episode(arrays: dict[str, np.ndarray], validation_ratio: float = 0.2, seed: int = 42):
    episode_ids = np.unique(arrays["episode_ids"])
    if episode_ids.size < 2:
        raise ValueError("At least two episodes are required for train/validation split.")

    rng = np.random.default_rng(seed)
    shuffled = episode_ids.copy()
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
    validation_episodes = set(int(episode) for episode in shuffled[:validation_count])
    validation_mask = np.asarray([int(episode) in validation_episodes for episode in arrays["episode_ids"]])
    train_mask = ~validation_mask

    return (
        {key: value[train_mask] for key, value in arrays.items()},
        {key: value[validation_mask] for key, value in arrays.items()},
    )


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    action_counts = defaultdict(int)
    for action in arrays["actions"]:
        action_counts[int(action)] += 1
    return {
        "samples": int(arrays["actions"].shape[0]),
        "episodes": int(np.unique(arrays["episode_ids"]).size),
        "action_counts": dict(sorted(action_counts.items())),
        "reward_mean": float(np.mean(arrays["rewards"])) if arrays["rewards"].size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Unity/Python GridWorld JSONL demonstrations to NPZ.")
    parser.add_argument("--input", required=True, help="JSONL file or directory")
    parser.add_argument("--config", default="configs/env_config.json")
    parser.add_argument("--output-dir", default="datasets/demonstrations")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    state_shape = (config.channel_count, config.grid_size, config.grid_size)
    rows = load_transitions(Path(args.input))
    arrays = transitions_to_arrays(rows, state_shape)
    train_arrays, val_arrays = split_by_episode(arrays, validation_ratio=args.validation_ratio, seed=args.seed)

    output_dir = Path(args.output_dir)
    save_npz(output_dir / "demo_train.npz", train_arrays)
    save_npz(output_dir / "demo_val.npz", val_arrays)

    print("Loaded:", summarize(arrays))
    print("Train:", summarize(train_arrays))
    print("Validation:", summarize(val_arrays))


if __name__ == "__main__":
    main()
