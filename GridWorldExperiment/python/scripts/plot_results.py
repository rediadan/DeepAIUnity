from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SERIES = [
    ("MLP-DQN", Path("runs/dqn_mlp/metrics.csv")),
    ("CNN-DQN", Path("runs/dqn_pure/metrics.csv")),
    ("BC Init + CNN-DQN", Path("runs/dqn_bc_init/metrics.csv")),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except ValueError:
        return default


def plot_metric(output_dir: Path, metric: str, title: str, ylabel: str, output_name: str) -> None:
    plt.figure(figsize=(8, 5))
    has_data = False
    for label, path in SERIES:
        rows = read_rows(path)
        if not rows:
            continue
        episodes = [read_float(row, "episode") for row in rows]
        values = [read_float(row, metric) for row in rows]
        plt.plot(episodes, values, marker="o", label=label)
        has_data = True

    if not has_data:
        print(f"Skip {output_name}: no metrics found")
        return

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / output_name, dpi=160)
    plt.close()
    print(f"Saved {output_dir / output_name}")


def plot_time_to_threshold(output_dir: Path, threshold: float) -> None:
    labels = []
    values = []
    for label, path in SERIES:
        rows = read_rows(path)
        if not rows:
            continue
        reached_episode = None
        for row in rows:
            if read_float(row, "eval_average_return") >= threshold:
                reached_episode = read_float(row, "episode")
                break
        labels.append(label)
        values.append(reached_episode if reached_episode is not None else 0)

    if not labels:
        print("Skip time_to_threshold: no metrics found")
        return

    plt.figure(figsize=(8, 5))
    colors = ["#5A8DEE" if value > 0 else "#C7CDD8" for value in values]
    plt.bar(labels, values, color=colors)
    plt.title(f"Time to Threshold (return >= {threshold})")
    plt.ylabel("Episode reached, 0 means not reached")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_dir / "time_to_threshold.png", dpi=160)
    plt.close()
    print(f"Saved {output_dir / 'time_to_threshold.png'}")


def plot_evaluation_summary(output_dir: Path, evaluation_path: Path) -> None:
    rows = read_rows(evaluation_path)
    if not rows:
        print("Skip final evaluation chart: no evaluation summary found")
        return
    labels = [row["agent"] for row in rows]
    returns = [read_float(row, "average_return") for row in rows]
    success = [read_float(row, "success_rate") for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(labels, returns, color="#4C78A8")
    axes[0].set_title("Final Average Return")
    axes[0].set_ylabel("Average Return")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(labels, success, color="#59A14F")
    axes[1].set_title("Final Success Rate")
    axes[1].set_ylabel("Success Rate")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "final_evaluation_bar_chart.png", dpi=160)
    plt.close(fig)
    print(f"Saved {output_dir / 'final_evaluation_bar_chart.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GridWorld experiment results.")
    parser.add_argument("--output-dir", default="runs/plots")
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--evaluation", default="runs/evaluation_summary.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    plot_metric(output_dir, "eval_average_return", "Average Return Over Episodes", "Average Return", "average_return_curve.png")
    plot_metric(output_dir, "eval_success_rate", "Success Rate Over Episodes", "Success Rate", "success_rate_curve.png")
    plot_time_to_threshold(output_dir, args.threshold)
    plot_evaluation_summary(output_dir, Path(args.evaluation))


if __name__ == "__main__":
    main()
