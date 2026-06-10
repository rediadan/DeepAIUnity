from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from models.bc_policy import BCPolicy


class DemonstrationDataset(Dataset):
    def __init__(self, path: Path):
        data = np.load(path)
        self.states = data["states"].astype(np.float32)
        self.actions = data["actions"].astype(np.int64)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int):
        return torch.from_numpy(self.states[index]), torch.tensor(self.actions[index], dtype=torch.long)


@dataclass(frozen=True)
class BCMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: BCPolicy, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    confusion = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            logits = model(states)
            loss = criterion(logits, actions)
            predictions = logits.argmax(dim=1)

            total_loss += float(loss.item()) * states.size(0)
            total_correct += int((predictions == actions).sum().item())
            total_count += int(states.size(0))

            for target, prediction in zip(actions.cpu().numpy(), predictions.cpu().numpy()):
                confusion[int(target), int(prediction)] += 1

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1), confusion


def write_metrics(path: Path, rows: list[BCMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "val_loss": row.val_loss,
                    "val_accuracy": row.val_accuracy,
                }
            )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    train_dataset = DemonstrationDataset(Path(args.train))
    val_dataset = DemonstrationDataset(Path(args.val))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = BCPolicy(
        input_channels=args.input_channels,
        grid_size=args.grid_size,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        action_count=args.action_count,
        feature_dim=args.feature_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    rows: list[BCMetrics] = []
    best_accuracy = -1.0
    best_confusion = np.zeros((args.action_count, args.action_count), dtype=np.int64)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for states, actions in train_loader:
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * states.size(0)
            train_count += int(states.size(0))

        val_loss, val_accuracy, confusion = evaluate(model, val_loader, device)
        train_loss = train_loss_sum / max(train_count, 1)
        rows.append(BCMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_accuracy=val_accuracy))

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_confusion = confusion.copy()
            checkpoint_path = Path(args.output)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_channels": args.input_channels,
                    "grid_size": args.grid_size,
                    "grid_height": args.grid_height or args.grid_size,
                    "grid_width": args.grid_width or args.grid_size,
                    "action_count": args.action_count,
                    "feature_dim": args.feature_dim,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )
            encoder_path = Path(args.encoder_output)
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.encoder.state_dict(), encoder_path)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

    write_metrics(Path(args.metrics), rows)
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("Confusion matrix rows=true actions, cols=predicted actions")
    print(best_confusion)
    print(f"Saved checkpoint to {args.output}")
    print(f"Saved encoder to {args.encoder_output}")
    print(f"Saved metrics to {args.metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BC policy from GridWorld demonstrations.")
    parser.add_argument("--train", default="datasets/demonstrations/demo_train.npz")
    parser.add_argument("--val", default="datasets/demonstrations/demo_val.npz")
    parser.add_argument("--output", default="exports/bc_policy.pth")
    parser.add_argument("--encoder-output", default="exports/bc_encoder.pth")
    parser.add_argument("--metrics", default="runs/bc/metrics.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-channels", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=24)
    parser.add_argument("--grid-height", type=int)
    parser.add_argument("--grid-width", type=int)
    parser.add_argument("--action-count", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
