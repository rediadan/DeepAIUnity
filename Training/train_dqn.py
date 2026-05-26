import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ACTION_COUNT = 6
MODEL_TYPES = ("mlp", "deep_mlp", "dueling", "conv1d")


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as exception:
        if exception.name != "numpy._core":
            raise

        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
        return torch.load(path, map_location="cpu", weights_only=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DqnReplayDataset(Dataset):
    def __init__(self, dataset_path: Path):
        data = np.load(dataset_path)
        self.feature_mean = data["feature_mean"].astype(np.float32)
        self.feature_std = data["feature_std"].astype(np.float32)
        state = data["state"].astype(np.float32)
        next_state = data["next_state"].astype(np.float32)
        self.state = ((state - self.feature_mean) / self.feature_std).astype(np.float32)
        self.next_state = ((next_state - self.feature_mean) / self.feature_std).astype(np.float32)
        self.action = data["action"].astype(np.int64)
        self.reward = data["reward"].astype(np.float32)
        self.done = data["done"].astype(np.float32)
        self.base_feature_count = int(data["base_feature_count"]) if "base_feature_count" in data.files else 21
        self.sequence_length = int(data["sequence_length"]) if "sequence_length" in data.files else max(1, state.shape[1] // self.base_feature_count)
        self.feature_version = str(data["feature_version"]) if "feature_version" in data.files else "v1"

    def __len__(self) -> int:
        return len(self.state)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.state[index]),
            torch.tensor(self.action[index], dtype=torch.long),
            torch.tensor(self.reward[index], dtype=torch.float32),
            torch.from_numpy(self.next_state[index]),
            torch.tensor(self.done[index], dtype=torch.float32),
        )


class DqnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, action_count: int = ACTION_COUNT):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden_size // 2, action_count)

    def forward(self, x):
        return self.q_head(self.backbone(x))


class DeepMlpDqnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, action_count: int = ACTION_COUNT):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_count),
        )

    def forward(self, x):
        return self.network(x)


class DuelingDqnModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, action_count: int = ACTION_COUNT):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_count),
        )

    def forward(self, x):
        hidden = self.backbone(x)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class Conv1dDqnModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        action_count: int = ACTION_COUNT,
        sequence_length: int = 4,
        base_feature_count: int = 38,
    ):
        super().__init__()
        if input_size != sequence_length * base_feature_count:
            raise ValueError(
                "Conv1D DQN expects input_size == sequence_length * base_feature_count. "
                f"got input_size={input_size}, sequence_length={sequence_length}, base_feature_count={base_feature_count}"
            )

        self.sequence_length = sequence_length
        self.base_feature_count = base_feature_count
        self.conv = nn.Sequential(
            nn.Conv1d(base_feature_count, hidden_size, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_count),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        sequence = x.view(batch_size, self.sequence_length, self.base_feature_count).transpose(1, 2)
        return self.head(self.conv(sequence))


def build_model(
    model_type: str,
    input_size: int,
    hidden_size: int,
    action_count: int,
    sequence_length: int,
    base_feature_count: int,
) -> nn.Module:
    if model_type == "mlp":
        return DqnModel(input_size=input_size, hidden_size=hidden_size, action_count=action_count)
    if model_type == "deep_mlp":
        return DeepMlpDqnModel(input_size=input_size, hidden_size=hidden_size, action_count=action_count)
    if model_type == "dueling":
        return DuelingDqnModel(input_size=input_size, hidden_size=hidden_size, action_count=action_count)
    if model_type == "conv1d":
        return Conv1dDqnModel(
            input_size=input_size,
            hidden_size=hidden_size,
            action_count=action_count,
            sequence_length=sequence_length,
            base_feature_count=base_feature_count,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    gamma: float
    target_update_interval: int
    model_type: str
    seed: int


def load_bc_backbone_if_available(model: nn.Module, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = load_torch_checkpoint(checkpoint_path)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in source_state.items()
        if key.startswith("backbone.") and key in model_state and model_state[key].shape == value.shape
    }
    model_state.update(compatible)
    model.load_state_dict(model_state)
    print(f"Loaded {len(compatible)} BC backbone tensors from {checkpoint_path}")


def evaluate(model, loader, device, gamma: float):
    model.eval()
    total_loss = 0.0
    total_count = 0
    mse = nn.SmoothL1Loss()

    with torch.no_grad():
        for state, action, reward, next_state, done in loader:
            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)

            q_values = model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            next_q = model(next_state).max(dim=1).values
            target = reward + gamma * (1.0 - done) * next_q
            loss = mse(q_values, target)
            total_loss += loss.item() * state.size(0)
            total_count += state.size(0)

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline DQN model from Deep AI Arena transition logs.")
    parser.add_argument("--dataset", required=True, help="Path to preprocessed DQN NPZ dataset")
    parser.add_argument("--output", required=True, help="Path to save trained DQN checkpoint (.pt)")
    parser.add_argument("--bc-checkpoint", help="Optional BC checkpoint used to initialize the shared backbone")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--target-update-interval", type=int, default=5)
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", help="Optional CSV path for epoch metrics")
    parser.add_argument("--summary-output", help="Optional JSON path for final run summary")
    args = parser.parse_args()

    if args.bc_checkpoint and args.model_type != "mlp":
        parser.error("--bc-checkpoint is only supported with --model-type mlp because the backbone shapes must match.")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
        model_type=args.model_type,
        seed=args.seed,
    )
    set_seed(config.seed)

    dataset = DqnReplayDataset(Path(args.dataset))
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.model_type,
        input_size=dataset.state.shape[1],
        hidden_size=config.hidden_size,
        action_count=ACTION_COUNT,
        sequence_length=dataset.sequence_length,
        base_feature_count=dataset.base_feature_count,
    ).to(device)
    target_model = build_model(
        config.model_type,
        input_size=dataset.state.shape[1],
        hidden_size=config.hidden_size,
        action_count=ACTION_COUNT,
        sequence_length=dataset.sequence_length,
        base_feature_count=dataset.base_feature_count,
    ).to(device)
    load_bc_backbone_if_available(model, Path(args.bc_checkpoint) if args.bc_checkpoint else None)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.SmoothL1Loss()
    metrics_rows: list[dict[str, float | int | str]] = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_count = 0

        for state, action, reward, next_state, done in train_loader:
            state = state.to(device)
            action = action.to(device)
            reward = reward.to(device)
            next_state = next_state.to(device)
            done = done.to(device)

            q_values = model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = target_model(next_state).max(dim=1).values
                target = reward + config.gamma * (1.0 - done) * next_q

            loss = loss_fn(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * state.size(0)
            total_count += state.size(0)

        if (epoch + 1) % config.target_update_interval == 0:
            target_model.load_state_dict(model.state_dict())

        valid_loss = evaluate(model, valid_loader, device, config.gamma)
        train_loss = total_loss / max(total_count, 1)
        metrics_rows.append(
            {
                "epoch": epoch + 1,
                "model_type": config.model_type,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
        )
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": dataset.state.shape[1],
            "hidden_size": config.hidden_size,
            "action_count": ACTION_COUNT,
            "feature_mean": dataset.feature_mean,
            "feature_std": dataset.feature_std,
            "gamma": config.gamma,
            "model_type": config.model_type,
            "base_feature_count": dataset.base_feature_count,
            "sequence_length": dataset.sequence_length,
            "feature_version": dataset.feature_version,
            "seed": config.seed,
        },
        output_path,
    )
    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "model_type", "train_loss", "valid_loss"])
            writer.writeheader()
            writer.writerows(metrics_rows)

        print(f"Saved metrics to {metrics_path}")

    summary_payload = {
        "model_type": config.model_type,
        "output": str(output_path),
        "dataset": str(Path(args.dataset)),
        "feature_version": dataset.feature_version,
        "input_size": int(dataset.state.shape[1]),
        "base_feature_count": int(dataset.base_feature_count),
        "sequence_length": int(dataset.sequence_length),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "hidden_size": config.hidden_size,
        "gamma": config.gamma,
        "seed": config.seed,
        "final_train_loss": float(metrics_rows[-1]["train_loss"]) if metrics_rows else None,
        "final_valid_loss": float(metrics_rows[-1]["valid_loss"]) if metrics_rows else None,
        "bc_checkpoint": args.bc_checkpoint,
    }
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(f"Saved summary to {summary_path}")

    print(f"Saved DQN checkpoint to {output_path}")


if __name__ == "__main__":
    main()
