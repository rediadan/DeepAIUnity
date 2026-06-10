from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CountRange:
    min: int
    max: int


@dataclass(frozen=True)
class RewardConfig:
    coin: float
    death: float
    trap: float
    wall_hit: float
    step: float
    clear_bonus: float


@dataclass(frozen=True)
class GridWorldConfig:
    env_version: str
    world_size: float
    grid_size: int
    move_step: float
    player_radius: float
    coin_radius: float
    enemy_radius: float
    trap_radius: float
    enemy_speed: float
    channels: tuple[str, ...]
    actions: tuple[str, ...]
    max_steps: int
    target_score: int
    coin_count: CountRange
    enemy_count: CountRange
    wall_density: float
    trap_density: float
    danger_radius: int
    moving_enemies: bool
    reward: RewardConfig
    random_seed: int

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    @property
    def action_count(self) -> int:
        return len(self.actions)


def _count_range(payload: dict[str, Any]) -> CountRange:
    return CountRange(min=int(payload["min"]), max=int(payload["max"]))


def _reward_config(payload: dict[str, Any]) -> RewardConfig:
    return RewardConfig(
        coin=float(payload["coin"]),
        death=float(payload["death"]),
        trap=float(payload["trap"]),
        wall_hit=float(payload["wall_hit"]),
        step=float(payload["step"]),
        clear_bonus=float(payload["clear_bonus"]),
    )


def load_config(path: str | Path = "configs/env_config.json") -> GridWorldConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return GridWorldConfig(
        env_version=str(payload["env_version"]),
        world_size=float(payload.get("world_size", 12.0)),
        grid_size=int(payload["grid_size"]),
        move_step=float(payload.get("move_step", 0.25)),
        player_radius=float(payload.get("player_radius", 0.2)),
        coin_radius=float(payload.get("coin_radius", 0.22)),
        enemy_radius=float(payload.get("enemy_radius", 0.24)),
        trap_radius=float(payload.get("trap_radius", 0.28)),
        enemy_speed=float(payload.get("enemy_speed", 0.16)),
        channels=tuple(str(channel) for channel in payload["channels"]),
        actions=tuple(str(action) for action in payload["actions"]),
        max_steps=int(payload["max_steps"]),
        target_score=int(payload["target_score"]),
        coin_count=_count_range(payload["coin_count"]),
        enemy_count=_count_range(payload["enemy_count"]),
        wall_density=float(payload["wall_density"]),
        trap_density=float(payload["trap_density"]),
        danger_radius=int(payload["danger_radius"]),
        moving_enemies=bool(payload["moving_enemies"]),
        reward=_reward_config(payload["reward"]),
        random_seed=int(payload["random_seed"]),
    )
