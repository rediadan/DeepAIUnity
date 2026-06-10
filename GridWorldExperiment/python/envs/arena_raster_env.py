from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
import json

import numpy as np


class ArenaRasterAction(IntEnum):
    IDLE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    SHOVE = 5


ACTION_DELTAS: dict[ArenaRasterAction, np.ndarray] = {
    ArenaRasterAction.IDLE: np.asarray([0.0, 0.0], dtype=np.float32),
    ArenaRasterAction.UP: np.asarray([0.0, 1.0], dtype=np.float32),
    ArenaRasterAction.DOWN: np.asarray([0.0, -1.0], dtype=np.float32),
    ArenaRasterAction.LEFT: np.asarray([-1.0, 0.0], dtype=np.float32),
    ArenaRasterAction.RIGHT: np.asarray([1.0, 0.0], dtype=np.float32),
    ArenaRasterAction.SHOVE: np.asarray([0.0, 0.0], dtype=np.float32),
}


CANONICAL_ARENA_CHANNELS = (
    "self",
    "opponent",
    "item",
    "base",
    "wall",
    "walkable",
    "door_closed",
    "switch",
    "moving_obstacle",
    "danger",
    "self_has_item",
    "opponent_has_item",
)


def semantic_channels_to_rgb(semantic: np.ndarray) -> np.ndarray:
    """Project the 12 semantic planes into an Atari-style 3-channel screen tensor."""
    _, height, width = semantic.shape
    rgb = np.zeros((3, height, width), dtype=np.float32)
    self_has_item = bool(np.max(semantic[10]) > 0.5)
    opponent_has_item = bool(np.max(semantic[11]) > 0.5)

    walkable = semantic[5] > 0.5
    rgb[:, walkable] = 0.08

    wall = semantic[4] > 0.5
    rgb[:, wall] = 0.26

    door = semantic[6] > 0.5
    rgb[0, door] = 0.95
    rgb[1, door] = 0.45
    rgb[2, door] = 0.08

    danger = np.clip(semantic[9], 0.0, 1.0)
    danger_mask = danger > 0.0
    rgb[0, danger_mask] = np.maximum(rgb[0, danger_mask], 0.55 + 0.35 * danger[danger_mask])
    rgb[1, danger_mask] *= 1.0 - 0.45 * danger[danger_mask]
    rgb[2, danger_mask] *= 1.0 - 0.45 * danger[danger_mask]

    switch = semantic[7]
    switch_mask = switch > 0.0
    rgb[0, switch_mask] = np.maximum(rgb[0, switch_mask], 0.05)
    rgb[1, switch_mask] = np.maximum(rgb[1, switch_mask], 0.55 + 0.4 * switch[switch_mask])
    rgb[2, switch_mask] = np.maximum(rgb[2, switch_mask], 0.7)

    moving = semantic[8] > 0.5
    rgb[0, moving] = 0.9
    rgb[1, moving] = 0.05
    rgb[2, moving] = 1.0

    base = semantic[3] > 0.5
    rgb[0, base] = 0.1
    rgb[1, base] = 0.95
    rgb[2, base] = 0.25

    item = semantic[2] > 0.5
    rgb[0, item] = 1.0
    rgb[1, item] = 0.9
    rgb[2, item] = 0.05

    opponent = semantic[1] > 0.5
    rgb[0, opponent] = 1.0
    rgb[1, opponent] = 0.55 if opponent_has_item else 0.06
    rgb[2, opponent] = 0.04 if opponent_has_item else 0.1

    self_mask = semantic[0] > 0.5
    rgb[0, self_mask] = 0.05 if self_has_item else 0.08
    rgb[1, self_mask] = 0.95 if self_has_item else 0.45
    rgb[2, self_mask] = 1.0

    if self_has_item:
        rgb[:, height - 3 : height, 0:6] = np.asarray([0.05, 0.95, 1.0], dtype=np.float32).reshape(3, 1, 1)
    if opponent_has_item:
        rgb[:, height - 3 : height, width - 6 : width] = np.asarray([1.0, 0.55, 0.04], dtype=np.float32).reshape(3, 1, 1)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def apply_target_highlight(
    rgb: np.ndarray,
    target_position: np.ndarray,
    target_type: str,
    world_min: np.ndarray,
    world_max: np.ndarray,
    radius_cells: int = 2,
) -> np.ndarray:
    """Draw a small target marker so CNN-DQN can distinguish the current objective."""
    output = rgb.copy()
    _, height, width = output.shape
    normalized = (target_position - world_min) / (world_max - world_min)
    center_x = int(np.clip(np.floor(normalized[0] * width), 0, width - 1))
    center_y = int(np.clip(np.floor((1.0 - normalized[1]) * height), 0, height - 1))
    color = {
        "base": np.asarray([0.05, 1.0, 0.15], dtype=np.float32),
        "opponent": np.asarray([1.0, 0.15, 0.05], dtype=np.float32),
        "item": np.asarray([1.0, 0.95, 0.05], dtype=np.float32),
        "switch": np.asarray([0.05, 0.95, 1.0], dtype=np.float32),
    }.get(target_type, np.asarray([1.0, 0.95, 0.05], dtype=np.float32))

    radius = max(1, int(radius_cells))
    for y in range(center_y - radius, center_y + radius + 1):
        for x in range(center_x - radius, center_x + radius + 1):
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            if abs(y - center_y) + abs(x - center_x) > radius + 1:
                continue
            output[:, y, x] = color
    return np.clip(output, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class RectObstacle:
    center: tuple[float, float]
    size: tuple[float, float]

    @property
    def min_x(self) -> float:
        return self.center[0] - self.size[0] * 0.5

    @property
    def max_x(self) -> float:
        return self.center[0] + self.size[0] * 0.5

    @property
    def min_y(self) -> float:
        return self.center[1] - self.size[1] * 0.5

    @property
    def max_y(self) -> float:
        return self.center[1] + self.size[1] * 0.5


@dataclass(frozen=True)
class ArenaRasterStepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict[str, int | float | bool | str]


class ArenaRasterConfig:
    def __init__(self, path: str | Path = "configs/arena_raster_config.json"):
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.env_version = str(payload["env_version"])
        self.world_min = np.asarray(payload["world_min"], dtype=np.float32)
        self.world_max = np.asarray(payload["world_max"], dtype=np.float32)
        self.grid_height = int(payload["grid_height"])
        self.grid_width = int(payload["grid_width"])
        self.grid_size = self.grid_height
        self.channels = tuple(str(channel) for channel in payload["channels"])
        self.observation_format = str(payload.get("observation_format", "semantic_channels"))
        self.frame_stack = max(1, int(payload.get("frame_stack", 1)))
        self.actions = tuple(str(action) for action in payload["actions"])
        self.max_steps = int(payload["max_steps"])
        self.move_step = float(payload["move_step"])
        self.actor_radius = float(payload["actor_radius"])
        self.item_radius = float(payload["item_radius"])
        self.base_radius = float(payload["base_radius"])
        self.shove_radius = float(payload["shove_radius"])
        self.enemy_speed = float(payload["enemy_speed"])
        self.moving_obstacle_speed = float(payload["moving_obstacle_speed"])
        self.danger_radius = float(payload["danger_radius"])
        self.target_score = int(payload["target_score"])
        self.random_seed = int(payload["random_seed"])
        self.rewards = payload["rewards"]
        if self.observation_format == "semantic_rgb_frame_stack_4" and self.channel_count != 3 * self.frame_stack:
            raise ValueError(
                "semantic_rgb_frame_stack_4 expects channel count to match 3 * frame_stack "
                f"({3 * self.frame_stack}), got {self.channel_count}."
            )

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def action_count(self) -> int:
        return len(self.actions)


class ArenaRasterEnv:
    """Python mirror of the Unity capture-the-flag Arena for CNN-DQN experiments."""

    def __init__(self, config: ArenaRasterConfig | str | Path | None = None):
        self.config = config if isinstance(config, ArenaRasterConfig) else ArenaRasterConfig(config or "configs/arena_raster_config.json")
        self.rng = np.random.default_rng(self.config.random_seed)
        self.static_walls = self._build_static_walls()
        self.doors = [
            RectObstacle((0.0, 0.75), (2.84625, 0.43875)),
            RectObstacle((0.0, -0.79), (2.84625, 0.4)),
        ]
        self.static_wall_mask = self._rect_mask(self.static_walls)
        self.walkable_mask = 1.0 - self.static_wall_mask
        self.door_masks = [self._rect_mask([door]) for door in self.doors]
        self.switches = [
            np.asarray([-5.1, 1.46], dtype=np.float32),
            np.asarray([5.15, 1.46], dtype=np.float32),
            np.asarray([-3.85, -1.06], dtype=np.float32),
            np.asarray([3.87, -1.02], dtype=np.float32),
        ]
        self.switch_to_door = [0, 0, 1, 1]
        self.moving_paths = [
            (np.asarray([-6.71, -0.08], dtype=np.float32), np.asarray([-1.96, -0.08], dtype=np.float32)),
            (np.asarray([6.71, -0.08], dtype=np.float32), np.asarray([1.96, -0.08], dtype=np.float32)),
        ]
        self.frame_stack_frames: list[np.ndarray] = []
        self.item_spawn_points = [
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 3.25], dtype=np.float32),
            np.asarray([0.0, -2.3], dtype=np.float32),
        ]
        self.base = np.asarray([0.0, -4.45], dtype=np.float32)
        self.reset(self.config.random_seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.last_seed = self.config.random_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.last_seed)
        self.player = np.asarray([-8.8, 0.0], dtype=np.float32)
        self.opponent = np.asarray([8.8, 0.0], dtype=np.float32)
        self.item = self.item_spawn_points[int(self.rng.integers(0, len(self.item_spawn_points)))].copy()
        self.self_has_item = False
        self.opponent_has_item = False
        self.item_held_by: str | None = None
        self.door_timers = [0, 0]
        self.moving_obstacles = [path[0].copy() for path in self.moving_paths]
        self.moving_dirs = [1, 1]
        self.score = 0
        self.steps = 0
        self.done = False
        self.previous_target_distance = self._target_distance()
        self.previous_target_type = self._target_type()
        self.frame_stack_frames = []
        return self.get_state()

    def step(self, action: int | ArenaRasterAction) -> ArenaRasterStepResult:
        if self.done:
            return ArenaRasterStepResult(self.get_state(), 0.0, True, self._info("already_done"))

        action = ArenaRasterAction(int(action))
        self.steps += 1
        reward = float(self.config.rewards["step"])
        event = "move"
        hit_wall = False
        forced_drop = False
        previous_target_distance = self.previous_target_distance
        previous_target_type = self.previous_target_type

        if action == ArenaRasterAction.SHOVE:
            if np.linalg.norm(self.player - self.opponent) <= self.config.shove_radius and self.opponent_has_item:
                forced_drop = True
                self.opponent_has_item = False
                self.item_held_by = None
                self.item = self.opponent.copy()
                reward += float(self.config.rewards["forced_drop"])
                event = "forced_drop"
            else:
                reward += float(self.config.rewards.get("invalid_shove", 0.0))
                event = "invalid_shove"
        else:
            candidate = self.player + ACTION_DELTAS[action] * self.config.move_step
            if self._blocked(candidate):
                hit_wall = True
                reward += float(self.config.rewards["wall_hit"])
                event = "wall_hit"
            else:
                self.player = candidate

        self._update_switches()
        self._move_opponent_rule_based()
        self._move_obstacles()

        if self._touching(self.player, self.item, self.config.actor_radius + self.config.item_radius) and self.item_held_by is None:
            self.self_has_item = True
            self.item_held_by = "self"
            reward += float(self.config.rewards["item_pickup"])
            event = "item_pickup"

        if self._touching(self.opponent, self.item, self.config.actor_radius + self.config.item_radius) and self.item_held_by is None:
            self.opponent_has_item = True
            self.item_held_by = "opponent"
            event = "opponent_pickup"

        if self.self_has_item:
            self.item = self.player.copy()
        elif self.opponent_has_item:
            self.item = self.opponent.copy()

        if any(self._touching(self.player, obstacle, 0.48) for obstacle in self.moving_obstacles):
            reward += float(self.config.rewards["obstacle_hit"])
            event = "obstacle_hit"

        if self.self_has_item and self._touching(self.player, self.base, self.config.base_radius):
            reward += float(self.config.rewards["deliver"])
            self.score = 1
            self.done = True
            event = "deliver"
        elif self.opponent_has_item and self._touching(self.opponent, self.base, self.config.base_radius):
            reward += float(self.config.rewards["opponent_deliver"])
            self.done = True
            event = "opponent_deliver"
        elif self.steps >= self.config.max_steps:
            self.done = True
            event = "timeout"

        current_distance = self._target_distance()
        progress = (previous_target_distance - current_distance) * self._target_progress_scale(previous_target_type)
        progress_clamp = float(self.config.rewards.get("target_progress_clamp", 0.15))
        reward += float(np.clip(progress, -progress_clamp, progress_clamp))
        self.previous_target_distance = current_distance
        self.previous_target_type = self._target_type()

        info = self._info(event)
        info.update({"hit_wall": hit_wall, "forced_drop": forced_drop})
        return ArenaRasterStepResult(self.get_state(), float(reward), self.done, info)

    def get_state(self) -> np.ndarray:
        semantic = self._get_semantic_channels()
        if self.config.observation_format == "semantic_rgb":
            return self._current_target_rgb(semantic)
        if self.config.observation_format == "semantic_rgb_frame_stack_4":
            return self._push_rgb_frame(self._current_target_rgb(semantic))
        return semantic

    def _current_target_rgb(self, semantic: np.ndarray) -> np.ndarray:
        return apply_target_highlight(
            semantic_channels_to_rgb(semantic),
            self._target_position(),
            self._target_type(),
            self.config.world_min,
            self.config.world_max,
        )

    def _push_rgb_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.frame_stack <= 1:
            return frame

        if not self.frame_stack_frames:
            self.frame_stack_frames = [frame.copy() for _ in range(self.config.frame_stack)]
        else:
            self.frame_stack_frames.append(frame.copy())
            self.frame_stack_frames = self.frame_stack_frames[-self.config.frame_stack :]

        return np.concatenate(self.frame_stack_frames, axis=0).astype(np.float32)

    def _get_semantic_channels(self) -> np.ndarray:
        channels = {name: index for index, name in enumerate(CANONICAL_ARENA_CHANNELS)}
        state = np.zeros((len(CANONICAL_ARENA_CHANNELS), self.config.grid_height, self.config.grid_width), dtype=np.float32)
        self._write_position(state, channels["self"], self.player)
        self._write_position(state, channels["opponent"], self.opponent)
        self._write_position(state, channels["item"], self.item)
        self._write_position(state, channels["base"], self.base)
        state[channels["wall"]] = self.static_wall_mask
        state[channels["walkable"]] = self.walkable_mask
        for door_mask, timer in zip(self.door_masks, self.door_timers):
            if timer <= 0:
                state[channels["door_closed"]] = np.maximum(state[channels["door_closed"]], door_mask)
        for switch in self.switches:
            self._write_position(state, channels["switch"], switch)
        for obstacle in self.moving_obstacles:
            self._write_position(state, channels["moving_obstacle"], obstacle)
            self._write_disc(state, channels["danger"], obstacle, self.config.danger_radius)
        self._write_disc(state, channels["danger"], self.opponent, 0.9)
        if self.self_has_item:
            state[channels["self_has_item"], :, :] = 1.0
        if self.opponent_has_item:
            state[channels["opponent_has_item"], :, :] = 1.0
        return state

    def _move_opponent_rule_based(self) -> None:
        target = self.base if self.opponent_has_item else (self.player if self.self_has_item else self.item)
        direction = target - self.opponent
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            return
        candidates = [
            direction / norm,
            np.asarray([np.sign(direction[0]), 0.0], dtype=np.float32),
            np.asarray([0.0, np.sign(direction[1])], dtype=np.float32),
            np.asarray([-direction[1], direction[0]], dtype=np.float32),
            np.asarray([direction[1], -direction[0]], dtype=np.float32),
        ]
        scored_candidates: list[tuple[float, np.ndarray]] = []
        for candidate_direction in candidates:
            candidate_norm = float(np.linalg.norm(candidate_direction))
            if candidate_norm <= 1e-6:
                continue
            candidate = self.opponent + candidate_direction / candidate_norm * self.config.enemy_speed
            if self._blocked(candidate):
                continue
            distance_after_move = float(np.linalg.norm(target - candidate))
            scored_candidates.append((distance_after_move, candidate))

        if scored_candidates:
            scored_candidates.sort(key=lambda value: value[0])
            self.opponent = scored_candidates[0][1].astype(np.float32)

    def _move_obstacles(self) -> None:
        for index, (point_a, point_b) in enumerate(self.moving_paths):
            target = point_b if self.moving_dirs[index] > 0 else point_a
            direction = target - self.moving_obstacles[index]
            norm = float(np.linalg.norm(direction))
            if norm <= self.config.moving_obstacle_speed:
                self.moving_obstacles[index] = target.copy()
                self.moving_dirs[index] *= -1
            else:
                self.moving_obstacles[index] += direction / norm * self.config.moving_obstacle_speed
        self.door_timers = [max(0, timer - 1) for timer in self.door_timers]

    def _update_switches(self) -> None:
        for index, switch in enumerate(self.switches):
            if self._touching(self.player, switch, 0.55) or self._touching(self.opponent, switch, 0.55):
                self.door_timers[self.switch_to_door[index]] = 12

    def _blocked(self, position: np.ndarray) -> bool:
        if np.any(position < self.config.world_min + self.config.actor_radius) or np.any(position > self.config.world_max - self.config.actor_radius):
            return True
        blockers = self.static_walls + [door for door, timer in zip(self.doors, self.door_timers) if timer <= 0]
        return any(self._circle_intersects_rect(position, self.config.actor_radius, wall) for wall in blockers)

    def _target_distance(self) -> float:
        if self.self_has_item:
            return float(np.linalg.norm(self.player - self.base))
        if self.opponent_has_item:
            return float(np.linalg.norm(self.player - self.opponent))
        return float(np.linalg.norm(self.player - self.item))

    def _target_position(self) -> np.ndarray:
        if self.self_has_item:
            return self.base
        if self.opponent_has_item:
            return self.opponent
        return self.item

    def _target_type(self) -> str:
        if self.self_has_item:
            return "base"
        if self.opponent_has_item:
            return "opponent"
        return "item"

    def _target_progress_scale(self, target_type: str) -> float:
        if target_type == "base":
            return float(self.config.rewards.get("target_progress_base", self.config.rewards["target_progress"]))
        if target_type == "opponent":
            return float(self.config.rewards.get("target_progress_opponent", self.config.rewards["target_progress"]))
        return float(self.config.rewards.get("target_progress_item", self.config.rewards["target_progress"]))

    def _info(self, event: str) -> dict[str, int | float | bool | str]:
        return {"event": event, "score": self.score, "steps": self.steps, "seed": self.last_seed}

    def _world_to_cell(self, position: np.ndarray) -> tuple[int, int]:
        normalized = (position - self.config.world_min) / (self.config.world_max - self.config.world_min)
        x = int(np.clip(normalized[0] * self.config.grid_width, 0, self.config.grid_width - 1))
        y = int(np.clip((1.0 - normalized[1]) * self.config.grid_height, 0, self.config.grid_height - 1))
        return y, x

    def _cell_center(self, y: int, x: int) -> np.ndarray:
        size = self.config.world_max - self.config.world_min
        normalized_y = 1.0 - (y + 0.5) / self.config.grid_height
        return self.config.world_min + np.asarray(
            [(x + 0.5) / self.config.grid_width, normalized_y],
            dtype=np.float32,
        ) * size

    def _write_position(self, state: np.ndarray, channel: int, position: np.ndarray) -> None:
        y, x = self._world_to_cell(position)
        state[channel, y, x] = 1.0

    def _write_disc(self, state: np.ndarray, channel: int, center: np.ndarray, radius: float) -> None:
        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                distance = float(np.linalg.norm(self._cell_center(y, x) - center))
                if distance <= radius:
                    state[channel, y, x] = max(state[channel, y, x], 1.0 - distance / max(radius, 1e-6))

    def _write_rects(self, state: np.ndarray, channel: int, rects: list[RectObstacle]) -> None:
        state[channel] = np.maximum(state[channel], self._rect_mask(rects))

    def _rect_mask(self, rects: list[RectObstacle]) -> np.ndarray:
        mask = np.zeros((self.config.grid_height, self.config.grid_width), dtype=np.float32)
        for y in range(self.config.grid_height):
            for x in range(self.config.grid_width):
                point = self._cell_center(y, x)
                if any(rect.min_x <= point[0] <= rect.max_x and rect.min_y <= point[1] <= rect.max_y for rect in rects):
                    mask[y, x] = 1.0
        return mask

    @staticmethod
    def _touching(left: np.ndarray, right: np.ndarray, radius: float) -> bool:
        return float(np.linalg.norm(left - right)) <= radius

    @staticmethod
    def _circle_intersects_rect(position: np.ndarray, radius: float, rect: RectObstacle) -> bool:
        closest = np.asarray([
            np.clip(position[0], rect.min_x, rect.max_x),
            np.clip(position[1], rect.min_y, rect.max_y),
        ], dtype=np.float32)
        return float(np.linalg.norm(position - closest)) <= radius

    @staticmethod
    def _build_static_walls() -> list[RectObstacle]:
        return [
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
