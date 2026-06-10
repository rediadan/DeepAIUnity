from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from envs.env_config import GridWorldConfig, load_config


class AgentAction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS: dict[AgentAction, np.ndarray] = {
    AgentAction.UP: np.asarray([0.0, 1.0], dtype=np.float32),
    AgentAction.DOWN: np.asarray([0.0, -1.0], dtype=np.float32),
    AgentAction.LEFT: np.asarray([-1.0, 0.0], dtype=np.float32),
    AgentAction.RIGHT: np.asarray([1.0, 0.0], dtype=np.float32),
}


@dataclass(frozen=True)
class WallRect:
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
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict[str, int | float | bool | str]


class GridWorldEnv:
    """Continuous top-down world rasterized into CNN-friendly feature maps."""

    def __init__(self, config: GridWorldConfig | str | Path | None = None):
        if config is None:
            self.config = load_config()
        elif isinstance(config, GridWorldConfig):
            self.config = config
        else:
            self.config = load_config(config)

        self.rng = np.random.default_rng(self.config.random_seed)
        self.player = np.zeros(2, dtype=np.float32)
        self.coins: list[np.ndarray] = []
        self.enemies: list[np.ndarray] = []
        self.enemy_previous: list[np.ndarray] = []
        self.enemy_predicted: list[np.ndarray] = []
        self.walls: list[WallRect] = []
        self.traps: list[np.ndarray] = []
        self.visited: set[tuple[int, int]] = set()
        self.score = 0
        self.steps = 0
        self.done = False
        self.last_seed = self.config.random_seed
        self.reset(self.config.random_seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.last_seed = self.config.random_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(self.last_seed)
        self.score = 0
        self.steps = 0
        self.done = False
        self.coins = []
        self.enemies = []
        self.enemy_previous = []
        self.enemy_predicted = []
        self.traps = []
        self.visited = set()
        self.player = np.zeros(2, dtype=np.float32)

        self.walls = self._generate_walls()
        self.traps = [self._sample_open_position(self.config.trap_radius) for _ in range(self._trap_count())]
        self.player = self._sample_open_position(self.config.player_radius)
        self.visited.add(self._to_grid(self.player))

        coin_count = int(self.rng.integers(self.config.coin_count.min, self.config.coin_count.max + 1))
        enemy_count = int(self.rng.integers(self.config.enemy_count.min, self.config.enemy_count.max + 1))
        self.coins = [self._sample_open_position(self.config.coin_radius) for _ in range(coin_count)]
        self.enemies = [self._sample_open_position(self.config.enemy_radius) for _ in range(enemy_count)]
        self.enemy_previous = [enemy.copy() for enemy in self.enemies]
        self.enemy_predicted = self._predict_enemies()
        return self.get_state()

    def step(self, action: int | AgentAction) -> StepResult:
        if self.done:
            return StepResult(self.get_state(), 0.0, True, self._info("already_done"))

        try:
            agent_action = AgentAction(int(action))
        except ValueError as exception:
            raise ValueError(f"Unsupported action: {action}") from exception

        self.steps += 1
        reward = self.config.reward.step
        event = "move"
        hit_wall = False
        hit_trap = False
        collected_coin = False
        death = False

        next_position = self.candidate_position(agent_action)
        if self.is_position_blocked(next_position, self.config.player_radius):
            hit_wall = True
            event = "wall_hit"
            reward += self.config.reward.wall_hit
        else:
            self.player = next_position
            self.visited.add(self._to_grid(self.player))

        if self._touches_any(self.player, self.config.player_radius, self.traps, self.config.trap_radius):
            hit_trap = True
            event = "trap"
            reward += self.config.reward.trap

        collected_index = self._first_touching_index(
            self.player,
            self.config.player_radius,
            self.coins,
            self.config.coin_radius,
        )
        if collected_index is not None:
            collected_coin = True
            event = "coin"
            self.coins.pop(collected_index)
            self.score += 1
            reward += self.config.reward.coin

        death = self._touches_any(self.player, self.config.player_radius, self.enemies, self.config.enemy_radius)
        if not death and self.config.moving_enemies:
            self._move_enemies()
            death = self._touches_any(self.player, self.config.player_radius, self.enemies, self.config.enemy_radius)
        else:
            self.enemy_previous = [enemy.copy() for enemy in self.enemies]
            self.enemy_predicted = self._predict_enemies()

        if death:
            event = "death"
            reward += self.config.reward.death
            self.done = True
        elif self.score >= self.config.target_score:
            event = "clear"
            reward += self.config.reward.clear_bonus
            self.done = True
        elif self.steps >= self.config.max_steps:
            event = "timeout"
            self.done = True

        info = self._info(event)
        info.update(
            {
                "hit_wall": hit_wall,
                "hit_trap": hit_trap,
                "collected_coin": collected_coin,
                "death": death,
            }
        )
        return StepResult(self.get_state(), float(reward), self.done, info)

    def get_state(self) -> np.ndarray:
        size = self.config.grid_size
        channels = {name: index for index, name in enumerate(self.config.channels)}
        state = np.zeros((self.config.channel_count, size, size), dtype=np.float32)

        self._write_position(state, channels["player"], self.player, 1.0)
        self._write_positions(state, channels["coin"], self.coins, 1.0)
        self._write_positions(state, channels["enemy"], self.enemies, 1.0)
        self._write_walls(state, channels["wall"])
        self._write_positions(state, channels["trap"], self.traps, 1.0)
        self._write_grid_cells(state, channels["visited"], self.visited, 1.0)
        self._write_danger(state, channels["danger"])
        self._write_positions(state, channels["enemy_previous"], self.enemy_previous, 1.0)
        self._write_positions(state, channels["enemy_predicted"], self.enemy_predicted, 1.0)
        target = self.nearest_coin()
        if target is not None:
            self._write_position(state, channels["target_coin"], target, 1.0)
        return state

    def candidate_position(self, action: int | AgentAction) -> np.ndarray:
        agent_action = AgentAction(int(action))
        return self.player + ACTION_DELTAS[agent_action] * self.config.move_step

    def is_position_blocked(self, position: np.ndarray, radius: float) -> bool:
        if not self._inside_world(position, radius):
            return True
        return any(self._circle_intersects_wall(position, radius, wall) for wall in self.walls)

    def nearest_coin(self) -> np.ndarray | None:
        if not self.coins:
            return None
        return min(self.coins, key=lambda coin: self.distance(self.player, coin))

    @staticmethod
    def distance(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm(left - right))

    def _generate_walls(self) -> list[WallRect]:
        walls: list[WallRect] = []
        target_count = self._wall_count()
        attempts = 0
        while len(walls) < target_count and attempts < target_count * 80:
            attempts += 1
            long_axis = self.rng.random() < 0.5
            width = float(self.rng.uniform(1.1, 2.4) if long_axis else self.rng.uniform(0.35, 0.75))
            height = float(self.rng.uniform(0.35, 0.75) if long_axis else self.rng.uniform(1.1, 2.4))
            margin = max(width, height) * 0.5 + 0.4
            center = (
                float(self.rng.uniform(margin, self.config.world_size - margin)),
                float(self.rng.uniform(margin, self.config.world_size - margin)),
            )
            wall = WallRect(center=center, size=(width, height))
            if not self._wall_blocks_spawn_area(wall):
                walls.append(wall)
        return walls

    def _wall_count(self) -> int:
        return max(3, int(round(self.config.grid_size * self.config.grid_size * self.config.wall_density / 16.0)))

    def _trap_count(self) -> int:
        return max(2, int(round(self.config.grid_size * self.config.grid_size * self.config.trap_density / 6.0)))

    def _wall_blocks_spawn_area(self, wall: WallRect) -> bool:
        center = np.asarray(wall.center, dtype=np.float32)
        world_center = np.asarray([self.config.world_size * 0.5, self.config.world_size * 0.5], dtype=np.float32)
        return self.distance(center, world_center) < 0.8

    def _sample_open_position(self, radius: float) -> np.ndarray:
        for _ in range(2000):
            candidate = self.rng.uniform(radius, self.config.world_size - radius, size=2).astype(np.float32)
            if self._is_occupied(candidate, radius):
                continue
            return candidate
        raise RuntimeError("Not enough open space to generate continuous GridWorld episode.")

    def _is_occupied(self, position: np.ndarray, radius: float) -> bool:
        if self.is_position_blocked(position, radius):
            return True
        if self._touches_any(position, radius, self.traps, self.config.trap_radius + 0.2):
            return True
        if self._touches_any(position, radius, self.coins, self.config.coin_radius + 0.25):
            return True
        if self._touches_any(position, radius, self.enemies, self.config.enemy_radius + 0.35):
            return True
        if self.player.any() and self.distance(position, self.player) < radius + self.config.player_radius + 0.35:
            return True
        return False

    def _move_enemies(self) -> None:
        self.enemy_previous = [enemy.copy() for enemy in self.enemies]
        moved: list[np.ndarray] = []
        for enemy in self.enemies:
            direction = self.player - enemy
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                moved.append(enemy.copy())
                continue
            candidate = enemy + direction / norm * self.config.enemy_speed
            if self.is_position_blocked(candidate, self.config.enemy_radius):
                moved.append(enemy.copy())
            else:
                moved.append(candidate.astype(np.float32))
        self.enemies = moved
        self.enemy_predicted = self._predict_enemies()

    def _predict_enemies(self) -> list[np.ndarray]:
        predicted: list[np.ndarray] = []
        for enemy in self.enemies:
            direction = self.player - enemy
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-6:
                predicted.append(enemy.copy())
                continue
            candidate = enemy + direction / norm * self.config.enemy_speed
            if self.is_position_blocked(candidate, self.config.enemy_radius):
                predicted.append(enemy.copy())
            else:
                predicted.append(candidate.astype(np.float32))
        return predicted

    def _inside_world(self, position: np.ndarray, radius: float) -> bool:
        return bool(
            radius <= position[0] <= self.config.world_size - radius
            and radius <= position[1] <= self.config.world_size - radius
        )

    @staticmethod
    def _circle_intersects_wall(position: np.ndarray, radius: float, wall: WallRect) -> bool:
        closest_x = float(np.clip(position[0], wall.min_x, wall.max_x))
        closest_y = float(np.clip(position[1], wall.min_y, wall.max_y))
        dx = float(position[0] - closest_x)
        dy = float(position[1] - closest_y)
        return dx * dx + dy * dy <= radius * radius

    @staticmethod
    def _touches_any(position: np.ndarray, radius: float, targets: list[np.ndarray], target_radius: float) -> bool:
        return any(float(np.linalg.norm(position - target)) <= radius + target_radius for target in targets)

    @staticmethod
    def _first_touching_index(
        position: np.ndarray,
        radius: float,
        targets: list[np.ndarray],
        target_radius: float,
    ) -> int | None:
        for index, target in enumerate(targets):
            if float(np.linalg.norm(position - target)) <= radius + target_radius:
                return index
        return None

    def _to_grid(self, position: np.ndarray) -> tuple[int, int]:
        size = self.config.grid_size
        scale = size / self.config.world_size
        x = int(np.clip(position[0] * scale, 0, size - 1))
        y = int(np.clip(position[1] * scale, 0, size - 1))
        return y, x

    def _cell_center_world(self, row: int, col: int) -> np.ndarray:
        cell_size = self.config.world_size / self.config.grid_size
        return np.asarray([(col + 0.5) * cell_size, (row + 0.5) * cell_size], dtype=np.float32)

    def _write_positions(self, state: np.ndarray, channel: int, positions: list[np.ndarray], value: float) -> None:
        for position in positions:
            self._write_position(state, channel, position, value)

    def _write_position(self, state: np.ndarray, channel: int, position: np.ndarray, value: float) -> None:
        row, col = self._to_grid(position)
        state[channel, row, col] = value

    @staticmethod
    def _write_grid_cells(
        state: np.ndarray,
        channel: int,
        cells: set[tuple[int, int]],
        value: float,
    ) -> None:
        for row, col in cells:
            state[channel, row, col] = value

    def _write_walls(self, state: np.ndarray, channel: int) -> None:
        size = self.config.grid_size
        for row in range(size):
            for col in range(size):
                center = self._cell_center_world(row, col)
                if any(wall.min_x <= center[0] <= wall.max_x and wall.min_y <= center[1] <= wall.max_y for wall in self.walls):
                    state[channel, row, col] = 1.0

    def _write_danger(self, state: np.ndarray, channel: int) -> None:
        radius_world = max(1, self.config.danger_radius) * self.config.world_size / self.config.grid_size
        for enemy in self.enemies:
            center_row, center_col = self._to_grid(enemy)
            search = max(1, self.config.danger_radius)
            for row in range(max(0, center_row - search), min(self.config.grid_size, center_row + search + 1)):
                for col in range(max(0, center_col - search), min(self.config.grid_size, center_col + search + 1)):
                    distance = self.distance(enemy, self._cell_center_world(row, col))
                    if distance <= radius_world:
                        state[channel, row, col] = max(state[channel, row, col], 1.0 - distance / (radius_world + 1e-6))

    def _info(self, event: str) -> dict[str, int | float | bool | str]:
        return {
            "event": event,
            "score": self.score,
            "steps": self.steps,
            "seed": self.last_seed,
            "coins_left": len(self.coins),
            "enemies": len(self.enemies),
            "player_x": float(self.player[0]),
            "player_y": float(self.player[1]),
        }
