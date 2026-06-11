from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.grid_env import GridWorldEnv
from models.bc_policy import BCPolicy
from models.dqn import CNNDQN, DuelingCNNDQN, DuelingNatureDQN, MLPDQN, NatureDQN, ResNetDQN
from rl.dqn_trainer import evaluate_model


def _build_model_from_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model_type = checkpoint.get("model_type", "cnn")
    input_channels = int(checkpoint.get("input_channels", 10))
    grid_size = int(checkpoint.get("grid_size", 24))
    grid_height = int(checkpoint.get("grid_height", grid_size))
    grid_width = int(checkpoint.get("grid_width", grid_size))
    action_count = int(checkpoint.get("action_count", 4))
    feature_dim = int(checkpoint.get("feature_dim", 256))
    model_type = str(model_type).lower()
    if model_type == "mlp":
        model = MLPDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("nature", "nature_cnn", "nature_dqn"):
        model = NatureDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("dueling", "dueling_cnn"):
        model = DuelingCNNDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("dueling_nature", "dueling_nature_cnn"):
        model = DuelingNatureDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("resnet", "residual_cnn"):
        model = ResNetDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    else:
        model = CNNDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _build_bc_policy(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    model = BCPolicy(
        input_channels=int(checkpoint.get("input_channels", 10)),
        grid_size=int(checkpoint.get("grid_size", 24)),
        grid_height=int(checkpoint.get("grid_height", checkpoint.get("grid_size", 24))),
        grid_width=int(checkpoint.get("grid_width", checkpoint.get("grid_size", 24))),
        action_count=int(checkpoint.get("action_count", 4)),
        feature_dim=int(checkpoint.get("feature_dim", 256)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _evaluate_bc(env: GridWorldEnv, model: BCPolicy, episodes: int, seed_start: int, device: torch.device):
    class BCAsDQN(torch.nn.Module):
        def __init__(self, policy: BCPolicy):
            super().__init__()
            self.policy = policy

        def forward(self, x):
            return self.policy(x)

    return evaluate_model(env, BCAsDQN(model), episodes, seed_start, device)


def _row(agent: str, stats: dict[str, float]) -> dict[str, str | float]:
    return {"agent": agent, **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GridWorld agents on fixed seeds.")
    parser.add_argument("--config", default="configs/env_config.json")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--output", default="runs/evaluation_summary.csv")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    env = GridWorldEnv(args.config)
    rows = [
        _row("Random", evaluate_model(env, None, args.episodes, args.seed_start, device, policy="random")),
        _row("Rule-Based", evaluate_model(env, None, args.episodes, args.seed_start, device, policy="rule")),
    ]

    candidates = [
        ("MLP-DQN", Path("exports/dqn_mlp.pth"), "dqn"),
        ("CNN-DQN", Path("exports/dqn_pure.pth"), "dqn"),
        ("BC Init + CNN-DQN", Path("exports/dqn_bc_init.pth"), "dqn"),
    ]
    for name, path, kind in candidates:
        if not path.exists():
            print(f"Skip missing checkpoint: {name} ({path})")
            continue
        if kind == "bc":
            stats = _evaluate_bc(env, _build_bc_policy(path, device), args.episodes, args.seed_start, device)
        else:
            stats = evaluate_model(env, _build_model_from_checkpoint(path, device), args.episodes, args.seed_start, device)
        rows.append(_row(name, stats))
        print(f"Evaluated {name}: return={stats['average_return']:.3f}, success={stats['success_rate']:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
