import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class ArenaDataset(Dataset):
    def __init__(self, dataset_path: Path):
        data = np.load(dataset_path)
        x = data["x"].astype(np.float32)
        self.feature_mean = data["feature_mean"].astype(np.float32)
        self.feature_std = data["feature_std"].astype(np.float32)
        self.x = ((x - self.feature_mean) / self.feature_std).astype(np.float32)
        self.move = data["move"].astype(np.int64)
        self.jump = data["jump"].astype(np.float32)
        self.drop = data["drop"].astype(np.float32)
        self.shove = data["shove"].astype(np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index]),
            torch.tensor(self.move[index], dtype=torch.long),
            torch.tensor(self.jump[index], dtype=torch.float32),
            torch.tensor(self.drop[index], dtype=torch.float32),
            torch.tensor(self.shove[index], dtype=torch.float32),
        )


class BehaviorCloningModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.move_head = nn.Linear(hidden_size // 2, 3)
        self.jump_head = nn.Linear(hidden_size // 2, 1)
        self.drop_head = nn.Linear(hidden_size // 2, 1)
        self.shove_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        hidden = self.backbone(x)
        return {
            "move_logits": self.move_head(hidden),
            "jump_logit": self.jump_head(hidden),
            "drop_logit": self.drop_head(hidden),
            "shove_logit": self.shove_head(hidden),
        }


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int


def compute_loss(model_output, move_target, jump_target, drop_target, shove_target):
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    move_loss = ce(model_output["move_logits"], move_target)
    jump_loss = bce(model_output["jump_logit"].squeeze(-1), jump_target)
    drop_loss = bce(model_output["drop_logit"].squeeze(-1), drop_target)
    shove_loss = bce(model_output["shove_logit"].squeeze(-1), shove_target)
    total = move_loss + jump_loss + drop_loss + shove_loss
    return total, {
        "move": move_loss.item(),
        "jump": jump_loss.item(),
        "drop": drop_loss.item(),
        "shove": shove_loss.item(),
    }


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_move_correct = 0
    total_count = 0

    with torch.no_grad():
        for x, move, jump, drop, shove in loader:
            x = x.to(device)
            move = move.to(device)
            jump = jump.to(device)
            drop = drop.to(device)
            shove = shove.to(device)

            output = model(x)
            loss, _ = compute_loss(output, move, jump, drop, shove)
            total_loss += loss.item() * x.size(0)
            total_move_correct += (output["move_logits"].argmax(dim=1) == move).sum().item()
            total_count += x.size(0)

    return {
        "loss": total_loss / max(total_count, 1),
        "move_acc": total_move_correct / max(total_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train behavior cloning model for Deep AI Arena.")
    parser.add_argument("--dataset", required=True, help="Path to preprocessed NPZ dataset")
    parser.add_argument("--output", required=True, help="Path to save trained checkpoint (.pt)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
    )

    dataset = ArenaDataset(Path(args.dataset))
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningModel(input_size=dataset.x.shape[1], hidden_size=config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        for x, move, jump, drop, shove in train_loader:
            x = x.to(device)
            move = move.to(device)
            jump = jump.to(device)
            drop = drop.to(device)
            shove = shove.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss, _ = compute_loss(output, move, jump, drop, shove)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, valid_loader, device)
        print(
            f"epoch={epoch + 1} "
            f"val_loss={metrics['loss']:.4f} "
            f"move_acc={metrics['move_acc']:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": dataset.x.shape[1],
            "hidden_size": config.hidden_size,
            "feature_mean": dataset.feature_mean,
            "feature_std": dataset.feature_std,
        },
        output_path,
    )
    print(f"Saved model checkpoint to {output_path}")


if __name__ == "__main__":
    main()
