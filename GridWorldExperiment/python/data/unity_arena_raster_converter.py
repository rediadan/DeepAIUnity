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

from envs.arena_raster_env import (
    CANONICAL_ARENA_CHANNELS,
    ArenaRasterConfig,
    RectObstacle,
    apply_target_highlight,
    semantic_channels_to_rgb,
)


STATIC_WALLS = [
    RectObstacle((0.0, 5.35), (20.5, 0.35)),
    RectObstacle((0.0, -5.35), (20.5, 0.35)),
    RectObstacle((-10.15, 0.0), (0.35, 10.7)),
    RectObstacle((10.15, 0.0), (0.35, 10.7)),
    RectObstacle((-4.26, 0.65), (5.56, 0.42)),
    RectObstacle((4.28, 0.65), (5.44, 0.42)),
    RectObstacle((-6.12, -0.81), (3.2, 0.42)),
    RectObstacle((6.08, -0.81), (3.2, 0.42)),
    RectObstacle((-3.89, -1.65), (3.2, 0.42)),
    RectObstacle((4.11, -1.65), (3.2, 0.42)),
    RectObstacle((-2.27, -1.03), (1.2, 0.42)),
    RectObstacle((2.36, -1.03), (1.2, 0.42)),
    RectObstacle((-6.78, 2.17), (0.5, 2.6)),
    RectObstacle((6.78, 2.17), (0.5, 2.6)),
]

DOORS = [
    RectObstacle((0.0, 0.75), (2.84625, 0.43875)),
    RectObstacle((0.0, -0.79), (2.84625, 0.4)),
]

SWITCHES = [
    np.asarray([-5.1, 1.46], dtype=np.float32),
    np.asarray([5.15, 1.46], dtype=np.float32),
    np.asarray([-3.85, -1.06], dtype=np.float32),
    np.asarray([3.87, -1.02], dtype=np.float32),
]


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


def _vector2(payload: dict[str, Any] | None, default: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    if not isinstance(payload, dict):
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(payload.get("x", default[0])), float(payload.get("y", default[1]))], dtype=np.float32)


class UnityArenaRasterizer:
    def __init__(self, config: ArenaRasterConfig):
        self.config = config
        self.channels = {name: index for index, name in enumerate(CANONICAL_ARENA_CHANNELS)}
        self.cell_centers = self._build_cell_centers()
        self.static_wall_mask = self._rect_mask(STATIC_WALLS)
        self.walkable_mask = 1.0 - self.static_wall_mask
        self.door_masks = [self._rect_mask([door]) for door in DOORS]

    @property
    def state_shape(self) -> tuple[int, int, int]:
        return (len(CANONICAL_ARENA_CHANNELS), self.config.grid_height, self.config.grid_width)

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        state = np.zeros(self.state_shape, dtype=np.float32)
        self_position = _vector2(observation.get("selfPosition"), (-8.8, 0.0))
        opponent_position = _vector2(observation.get("opponentPosition"), (8.8, 0.0))
        item_position = _vector2(observation.get("itemPosition"), (0.0, 0.0))
        base_position = _vector2(observation.get("basePosition"), (0.0, -4.45))

        self._write_position(state, "self", self_position)
        self._write_position(state, "opponent", opponent_position)
        self._write_position(state, "item", item_position)
        self._write_position(state, "base", base_position)

        state[self.channels["wall"]] = self.static_wall_mask
        state[self.channels["walkable"]] = self.walkable_mask

        self._write_doors(state, observation, self_position)
        self._write_switches(state, observation, self_position)
        self._write_moving_obstacle(state, observation, self_position)
        self._write_disc(state, "danger", opponent_position, 0.9)

        if bool(observation.get("selfHasItem", False)):
            state[self.channels["self_has_item"], :, :] = 1.0
        if bool(observation.get("opponentHasItem", False)):
            state[self.channels["opponent_has_item"], :, :] = 1.0

        return state

    def encode_rgb(self, observation: dict[str, Any]) -> np.ndarray:
        opponent_position = _vector2(observation.get("opponentPosition"), (8.8, 0.0))
        item_position = _vector2(observation.get("itemPosition"), (0.0, 0.0))
        base_position = _vector2(observation.get("basePosition"), (0.0, -4.45))

        if bool(observation.get("selfHasItem", False)):
            target_position = base_position
            target_type = "base"
        elif bool(observation.get("opponentHasItem", False)):
            target_position = opponent_position
            target_type = "opponent"
        else:
            target_position = item_position
            target_type = "item"

        return apply_target_highlight(
            semantic_channels_to_rgb(self.encode(observation)),
            target_position,
            target_type,
            self.config.world_min,
            self.config.world_max,
        )

    def _write_doors(self, state: np.ndarray, observation: dict[str, Any], self_position: np.ndarray) -> None:
        door_position = self_position + _vector2(observation.get("doorDelta"))
        nearest_index = int(np.argmin([np.linalg.norm(door_position - np.asarray(door.center, dtype=np.float32)) for door in DOORS]))
        for index, door_mask in enumerate(self.door_masks):
            if index == nearest_index and bool(observation.get("doorOpen", False)):
                continue
            state[self.channels["door_closed"]] = np.maximum(state[self.channels["door_closed"]], door_mask)

    def _write_switches(self, state: np.ndarray, observation: dict[str, Any], self_position: np.ndarray) -> None:
        switch_position = self_position + _vector2(observation.get("switchDelta"))
        nearest_index = int(np.argmin([np.linalg.norm(switch_position - switch) for switch in SWITCHES]))
        for index, switch in enumerate(SWITCHES):
            value = 1.0 if index == nearest_index and bool(observation.get("switchActive", False)) else 0.65
            self._write_position(state, "switch", switch, value)

    def _write_moving_obstacle(self, state: np.ndarray, observation: dict[str, Any], self_position: np.ndarray) -> None:
        moving_delta = _vector2(observation.get("movingObstacleDelta"))
        if np.linalg.norm(moving_delta) <= 1e-6:
            return
        obstacle_position = self_position + moving_delta
        self._write_position(state, "moving_obstacle", obstacle_position, 1.0)
        self._write_disc(state, "danger", obstacle_position, self.config.danger_radius)

    def _world_to_cell(self, position: np.ndarray) -> tuple[int, int]:
        normalized = (position - self.config.world_min) / (self.config.world_max - self.config.world_min)
        x = int(np.clip(np.floor(normalized[0] * self.config.grid_width), 0, self.config.grid_width - 1))
        y = int(np.clip(np.floor((1.0 - normalized[1]) * self.config.grid_height), 0, self.config.grid_height - 1))
        return y, x

    def _cell_center(self, y: int, x: int) -> np.ndarray:
        size = self.config.world_max - self.config.world_min
        normalized_y = 1.0 - (y + 0.5) / self.config.grid_height
        return self.config.world_min + np.asarray(
            [(x + 0.5) / self.config.grid_width, normalized_y],
            dtype=np.float32,
        ) * size

    def _build_cell_centers(self) -> np.ndarray:
        xs = (np.arange(self.config.grid_width, dtype=np.float32) + 0.5) / self.config.grid_width
        ys = 1.0 - (np.arange(self.config.grid_height, dtype=np.float32) + 0.5) / self.config.grid_height
        grid_x, grid_y = np.meshgrid(xs, ys)
        size = self.config.world_max - self.config.world_min
        centers = np.stack(
            [
                self.config.world_min[0] + grid_x * size[0],
                self.config.world_min[1] + grid_y * size[1],
            ],
            axis=-1,
        )
        return centers.astype(np.float32)

    def _write_position(self, state: np.ndarray, channel_name: str, position: np.ndarray, value: float = 1.0) -> None:
        y, x = self._world_to_cell(position)
        state[self.channels[channel_name], y, x] = value

    def _write_disc(self, state: np.ndarray, channel_name: str, center: np.ndarray, radius: float) -> None:
        channel = self.channels[channel_name]
        distance = np.linalg.norm(self.cell_centers - center.reshape(1, 1, 2), axis=-1)
        values = np.where(distance <= radius, 1.0 - distance / max(radius, 1e-6), 0.0)
        state[channel] = np.maximum(state[channel], values.astype(np.float32))

    def _rect_mask(self, rects: list[RectObstacle]) -> np.ndarray:
        mask = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.float32)
        x = self.cell_centers[:, :, 0]
        y = self.cell_centers[:, :, 1]
        for rect in rects:
            mask = np.maximum(
                mask,
                ((x >= rect.min_x) & (x <= rect.max_x) & (y >= rect.min_y) & (y <= rect.max_y)).astype(np.float32),
            )
        return mask


def load_dqn_transitions(input_path: Path, actor_side: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for path in _collect_jsonl_files(input_path):
        for row in _read_jsonl(path):
            if row.get("transitionType") != "DqnTransition":
                continue
            if actor_side.lower() != "all" and str(row.get("actorSide", "")).lower() != actor_side.lower():
                continue
            transitions.append(row)
            if max_rows is not None and len(transitions) >= max_rows:
                return transitions
    return transitions


def transitions_to_arrays(
    transitions: list[dict[str, Any]],
    rasterizer: UnityArenaRasterizer,
    output_format: str,
) -> dict[str, np.ndarray]:
    if not transitions:
        raise ValueError("No DqnTransition rows were found.")

    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []
    step_indices = []

    per_episode_step = defaultdict(int)
    rgb_frame_stacks: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for row in transitions:
        round_index = int(row.get("roundIndex", 0))
        actor_side = str(row.get("actorSide", "Left"))
        episode_key = (actor_side, round_index)
        step_index = per_episode_step[episode_key]
        per_episode_step[episode_key] += 1

        if output_format == "semantic_rgb":
            states.append(rasterizer.encode_rgb(row["state"]))
            next_states.append(rasterizer.encode_rgb(row["nextState"]))
        elif output_format == "semantic_rgb_frame_stack_4":
            state_rgb = rasterizer.encode_rgb(row["state"])
            next_state_rgb = rasterizer.encode_rgb(row["nextState"])
            stack = rgb_frame_stacks[episode_key]
            if step_index == 0 or not stack:
                stack[:] = [state_rgb.copy() for _ in range(4)]

            states.append(np.concatenate(stack, axis=0).astype(np.float32))
            next_stack = stack[1:] + [next_state_rgb.copy()]
            next_states.append(np.concatenate(next_stack, axis=0).astype(np.float32))
            stack[:] = next_stack
        else:
            states.append(rasterizer.encode(row["state"]))
            next_states.append(rasterizer.encode(row["nextState"]))
        actions.append(int(row["action"]))
        rewards.append(float(row["reward"]))
        done = bool(row.get("done", False))
        dones.append(done)
        episode_ids.append(round_index)
        step_indices.append(step_index)
        if done:
            rgb_frame_stacks.pop(episode_key, None)

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
    }


def split_by_episode(arrays: dict[str, np.ndarray], validation_ratio: float, seed: int):
    episode_ids = np.unique(arrays["episode_ids"])
    if episode_ids.size < 2:
        return arrays, {key: value[:0] for key, value in arrays.items()}

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
        "episodes": int(np.unique(arrays["episode_ids"]).size) if arrays["episode_ids"].size else 0,
        "state_shape": list(arrays["states"].shape[1:]),
        "action_counts": dict(sorted(action_counts.items())),
        "reward_mean": float(np.mean(arrays["rewards"])) if arrays["rewards"].size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Unity Arena DqnTransition JSONL logs to raster NPZ.")
    parser.add_argument("--input", required=True, help="JSONL file or directory containing Unity round logs")
    parser.add_argument("--config", default="configs/arena_raster_config.json")
    parser.add_argument("--output-dir", default="datasets/arena_demonstrations")
    parser.add_argument("--actor-side", default="Left", help="Left, Right, or All")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, help="Optional cap for smoke tests")
    parser.add_argument(
        "--output-format",
        choices=["semantic_channels", "semantic_rgb", "semantic_rgb_frame_stack_4"],
        default="semantic_channels",
        help=(
            "semantic_channels outputs 12xHxW, semantic_rgb outputs Atari-style 3xHxW, "
            "semantic_rgb_frame_stack_4 outputs 12xHxW from four RGB frames."
        ),
    )
    args = parser.parse_args()

    config = ArenaRasterConfig(args.config)
    rasterizer = UnityArenaRasterizer(config)
    transitions = load_dqn_transitions(Path(args.input), args.actor_side, args.max_rows)
    arrays = transitions_to_arrays(transitions, rasterizer, args.output_format)
    train_arrays, val_arrays = split_by_episode(arrays, args.validation_ratio, args.seed)

    output_dir = Path(args.output_dir)
    save_npz(output_dir / "demo_train.npz", train_arrays)
    save_npz(output_dir / "demo_val.npz", val_arrays)

    print("Loaded:", summarize(arrays))
    print("Train:", summarize(train_arrays))
    print("Validation:", summarize(val_arrays))


if __name__ == "__main__":
    main()
