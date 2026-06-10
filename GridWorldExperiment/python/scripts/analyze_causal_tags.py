from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ACTION_NAMES = ["up", "down", "left", "right"]
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _player_position(state: np.ndarray) -> tuple[int, int]:
    positions = np.argwhere(state[0] > 0.5)
    if positions.size == 0:
        return 0, 0
    return int(positions[0][0]), int(positions[0][1])


def _near_channel(state: np.ndarray, channel: int, position: tuple[int, int], radius: int = 2) -> bool:
    row, col = position
    height, width = state.shape[1], state.shape[2]
    row_min = max(0, row - radius)
    row_max = min(height, row + radius + 1)
    col_min = max(0, col - radius)
    col_max = min(width, col + radius + 1)
    return bool(np.any(state[channel, row_min:row_max, col_min:col_max] > 0.0))


def build_tags(state: np.ndarray, action: int, step_index: int, max_steps: int) -> list[str]:
    tags: list[str] = []
    position = _player_position(state)
    if _near_channel(state, 1, position, radius=3):
        tags.append("near_coin")
    if _near_channel(state, 2, position, radius=3):
        tags.append("near_enemy")
    if _near_channel(state, 4, position, radius=2):
        tags.append("near_trap")
    if state[5].sum() > max(8, state.shape[1] * state.shape[2] * 0.25):
        tags.append("revisiting_area")
    if step_index >= int(max_steps * 0.75):
        tags.append("time_pressure")

    delta = ACTION_DELTAS[int(action)]
    target = (position[0] + delta[0], position[1] + delta[1])
    if (
        target[0] < 0
        or target[1] < 0
        or target[0] >= state.shape[1]
        or target[1] >= state.shape[2]
        or state[3, target[0], target[1]] > 0.5
    ):
        tags.append("wall_blocking")

    if state[6, position[0], position[1]] > 0.25:
        tags.append("danger_zone")
    if state.shape[0] > 8 and _near_channel(state, 8, position, radius=3):
        tags.append("predicted_enemy_path")
    if state.shape[0] > 9 and _near_channel(state, 9, position, radius=4):
        tags.append("near_target_coin")
    return tags or ["neutral"]


def build_results(reward: float, done: bool) -> list[str]:
    results = []
    if reward >= 0.9:
        results.append("positive_reward")
    if reward <= -0.4:
        results.append("negative_reward")
    if done:
        results.append("episode_done")
    if not results:
        results.append("neutral_step")
    return results


def analyze(dataset_path: Path, output_dir: Path, max_steps: int) -> None:
    data = np.load(dataset_path)
    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    rewards = data["rewards"].astype(np.float32)
    dones = data["dones"].astype(bool)
    step_indices = data["step_indices"].astype(np.int64) if "step_indices" in data.files else np.arange(actions.shape[0])

    tag_stats = defaultdict(lambda: {"count": 0, "reward_sum": 0.0, "results": Counter()})
    edge_counts: Counter[tuple[str, str]] = Counter()
    action_counts: Counter[tuple[str, str]] = Counter()

    for state, action, reward, done, step_index in zip(states, actions, rewards, dones, step_indices):
        tags = build_tags(state, int(action), int(step_index), max_steps)
        results = build_results(float(reward), bool(done))
        action_name = ACTION_NAMES[int(action)]
        for tag in tags:
            stats = tag_stats[tag]
            stats["count"] += 1
            stats["reward_sum"] += float(reward)
            action_counts[(tag, action_name)] += 1
            for result in results:
                stats["results"][result] += 1
                edge_counts[(tag, result)] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tag_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["tag", "count", "avg_reward", "positive_reward_rate", "negative_reward_rate", "done_rate"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for tag, stats in sorted(tag_stats.items()):
            count = max(int(stats["count"]), 1)
            writer.writerow(
                {
                    "tag": tag,
                    "count": stats["count"],
                    "avg_reward": stats["reward_sum"] / count,
                    "positive_reward_rate": stats["results"]["positive_reward"] / count,
                    "negative_reward_rate": stats["results"]["negative_reward"] / count,
                    "done_rate": stats["results"]["episode_done"] / count,
                }
            )

    with (output_dir / "causal_edges.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cause_tag", "result", "count"])
        writer.writeheader()
        for (tag, result), count in sorted(edge_counts.items()):
            writer.writerow({"cause_tag": tag, "result": result, "count": count})

    with (output_dir / "tag_action_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tag", "action", "count"])
        writer.writeheader()
        for (tag, action), count in sorted(action_counts.items()):
            writer.writerow({"tag": tag, "action": action, "count": count})

    print(f"Saved causal tag analysis to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GridWorld causal tag statistics from NPZ transitions.")
    parser.add_argument("--input", default="datasets/demonstrations/demo_train.npz")
    parser.add_argument("--output-dir", default="runs/causal")
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    analyze(Path(args.input), Path(args.output_dir), args.max_steps)


if __name__ == "__main__":
    main()
