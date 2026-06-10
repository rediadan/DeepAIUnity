from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.grid_env import ACTION_DELTAS, AgentAction, GridWorldEnv
from data.demo_loader import save_npz, summarize


def _flatten_state(state: np.ndarray) -> list[float]:
    return state.astype(np.float32).reshape(-1).tolist()


def _nearest_coin_action(env: GridWorldEnv) -> int:
    nearest = env.nearest_coin()
    if nearest is None:
        return int(np.random.default_rng(env.steps + env.last_seed).integers(0, 4))

    ranked_actions = []
    for action, delta in ACTION_DELTAS.items():
        candidate = env.player + delta * env.config.move_step
        distance = env.distance(candidate, nearest)
        wall_penalty = 1000.0 if env.is_position_blocked(candidate, env.config.player_radius) else 0.0
        trap_penalty = 6.0 if any(env.distance(candidate, trap) <= env.config.player_radius + env.config.trap_radius for trap in env.traps) else 0.0
        enemy_penalty = 10.0 if any(env.distance(candidate, enemy) <= env.config.player_radius + env.config.enemy_radius + 0.25 for enemy in env.enemies) else 0.0
        danger_penalty = sum(max(0.0, 1.0 - env.distance(candidate, enemy) / 1.5) for enemy in env.enemies)
        ranked_actions.append((distance + wall_penalty + trap_penalty + enemy_penalty + danger_penalty, int(action)))

    ranked_actions.sort(key=lambda item: (item[0], item[1]))
    return ranked_actions[0][1]


def generate_demonstrations(
    output_path: Path,
    episodes: int,
    seed_start: int,
    config_path: Path,
) -> None:
    env = GridWorldEnv(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in range(episodes):
            state = env.reset(seed=seed_start + episode)
            done = False
            step = 0
            while not done:
                action = _nearest_coin_action(env)
                result = env.step(action)
                row = {
                    "episode": episode,
                    "step": step,
                    "state": _flatten_state(state),
                    "action": action,
                    "reward": result.reward,
                    "next_state": _flatten_state(result.state),
                    "done": result.done,
                    "event": result.info["event"],
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                state = result.state
                done = result.done
                step += 1


def generate_demonstration_npz(
    output_dir: Path,
    episodes: int,
    seed_start: int,
    config_path: Path,
    validation_ratio: float,
    seed: int,
) -> None:
    env = GridWorldEnv(config_path)
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    episode_ids = []
    step_indices = []

    for episode in range(episodes):
        state = env.reset(seed=seed_start + episode)
        done = False
        step = 0
        while not done:
            action = _nearest_coin_action(env)
            result = env.step(action)
            states.append(state.astype(np.float32))
            actions.append(int(action))
            rewards.append(float(result.reward))
            next_states.append(result.state.astype(np.float32))
            dones.append(bool(result.done))
            episode_ids.append(episode)
            step_indices.append(step)
            state = result.state
            done = result.done
            step += 1

    arrays = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "step_indices": np.asarray(step_indices, dtype=np.int64),
    }
    rng = np.random.default_rng(seed)
    episode_values = np.unique(arrays["episode_ids"])
    shuffled = episode_values.copy()
    rng.shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_ratio)))
    validation_episodes = set(int(episode) for episode in shuffled[:validation_count])
    validation_mask = np.asarray([int(episode) in validation_episodes for episode in arrays["episode_ids"]])
    train_mask = ~validation_mask
    train_arrays = {key: value[train_mask] for key, value in arrays.items()}
    val_arrays = {key: value[validation_mask] for key, value in arrays.items()}

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "demo_train.npz", train_arrays)
    save_npz(output_dir / "demo_val.npz", val_arrays)
    print("Loaded:", summarize(arrays))
    print("Train:", summarize(train_arrays))
    print("Validation:", summarize(val_arrays))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rule-based GridWorld demonstration JSONL.")
    parser.add_argument("--output", default="datasets/demonstrations/demo_raw.jsonl")
    parser.add_argument("--npz-output-dir")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--config", default="configs/env_config.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.npz_output_dir:
        generate_demonstration_npz(
            output_dir=Path(args.npz_output_dir),
            episodes=args.episodes,
            seed_start=args.seed_start,
            config_path=Path(args.config),
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
        print(f"Saved {args.episodes} demonstration episodes to {args.npz_output_dir}")
    else:
        generate_demonstrations(
            output_path=Path(args.output),
            episodes=args.episodes,
            seed_start=args.seed_start,
            config_path=Path(args.config),
        )
        print(f"Saved {args.episodes} demonstration episodes to {args.output}")


if __name__ == "__main__":
    main()
