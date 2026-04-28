import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ACTION_COUNT = 6


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


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    hidden_size: int
    gamma: float
    target_update_interval: int


def load_bc_backbone_if_available(model: DqnModel, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        gamma=args.gamma,
        target_update_interval=args.target_update_interval,
    )

    dataset = DqnReplayDataset(Path(args.dataset))
    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DqnModel(input_size=dataset.state.shape[1], hidden_size=config.hidden_size).to(device)
    target_model = DqnModel(input_size=dataset.state.shape[1], hidden_size=config.hidden_size).to(device)
    load_bc_backbone_if_available(model, Path(args.bc_checkpoint) if args.bc_checkpoint else None)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.SmoothL1Loss()

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
        print(
            f"epoch={epoch + 1} "
            f"train_loss={total_loss / max(total_count, 1):.4f} "
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
        },
        output_path,
    )
    print(f"Saved DQN checkpoint to {output_path}")


if __name__ == "__main__":
    main()
