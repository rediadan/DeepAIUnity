from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.arena_raster_env import ArenaRasterConfig
from rl.dqn_trainer import (
    DQNTrainConfig,
    batch_to_tensors,
    build_model,
    expert_ratio_for_progress,
    sample_training_batch,
    set_seed,
)
from rl.epsilon_scheduler import LinearEpsilonScheduler
from rl.expert_replay_buffer import ExpertReplayBuffer
from rl.replay_buffer import ReplayBuffer


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def torch_load_checkpoint(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_live_config(args: argparse.Namespace, payload: dict[str, Any]) -> tuple[ArenaRasterConfig, DQNTrainConfig, int]:
    env_config = ArenaRasterConfig(payload["env"]["config_path"])
    dqn_payload = payload["dqn"]
    epsilon_payload = payload["epsilon"]
    expert_payload = payload["expert_replay"]
    model_payload = payload["model"]

    use_expert_replay = bool(expert_payload.get("enabled", False)) or bool(args.use_expert_replay)
    config = DQNTrainConfig(
        model_type=str(model_payload.get("model_type", "cnn")),
        total_episodes=0,
        max_steps_per_episode=int(dqn_payload["max_steps_per_episode"]),
        batch_size=args.batch_size or int(dqn_payload["batch_size"]),
        gamma=float(dqn_payload["gamma"]),
        learning_rate=args.lr or float(dqn_payload["learning_rate"]),
        replay_size=args.replay_size or int(dqn_payload["replay_size"]),
        min_replay_size=args.min_replay_size or int(dqn_payload["min_replay_size"]),
        target_update_interval=args.target_update_interval or int(dqn_payload["target_update_interval"]),
        train_every_steps=args.train_every_steps or int(dqn_payload["train_every_steps"]),
        gradient_clip_norm=float(dqn_payload["gradient_clip_norm"]),
        epsilon_start=float(epsilon_payload["start"]),
        epsilon_end=float(epsilon_payload["end"]),
        epsilon_decay_steps=args.epsilon_decay_steps or int(epsilon_payload["decay_steps"]),
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        use_expert_replay=use_expert_replay,
        expert_dataset_path=args.expert_dataset or str(expert_payload["dataset_path"]),
        expert_ratio_start=float(expert_payload.get("ratio_start", 0.5)),
        expert_ratio_mid=float(expert_payload.get("ratio_mid", 0.25)),
        expert_ratio_end=float(expert_payload.get("ratio_end", 0.1)),
        seed=args.seed or int(payload["seed"]),
        cpu=args.cpu,
    )
    return env_config, config, int(model_payload.get("feature_dim", 256))


class LiveDqnTrainer:
    ACTION_NAMES = ("Idle", "Up", "Down", "Left", "Right", "Shove")

    def __init__(
        self,
        env_config: ArenaRasterConfig,
        config: DQNTrainConfig,
        feature_dim: int,
        init_checkpoint: str | None,
        save_interval_seconds: float,
        metrics_interval_steps: int,
        console_log_interval_steps: int,
        expert_bc_weight: float,
        expert_pretrain_steps: int,
        expert_pretrain_batch_size: int,
    ):
        set_seed(config.seed)
        random.seed(config.seed)
        self.env_config = env_config
        self.config = config
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
        self.state_shape = (env_config.channel_count, env_config.grid_height, env_config.grid_width)
        self.model = build_model(
            config.model_type,
            env_config.channel_count,
            env_config.grid_size,
            env_config.action_count,
            feature_dim,
            grid_height=env_config.grid_height,
            grid_width=env_config.grid_width,
        ).to(self.device)
        self.target_model = build_model(
            config.model_type,
            env_config.channel_count,
            env_config.grid_size,
            env_config.action_count,
            feature_dim,
            grid_height=env_config.grid_height,
            grid_width=env_config.grid_width,
        ).to(self.device)

        self.loaded_checkpoint = False
        if init_checkpoint:
            self.load_checkpoint(Path(init_checkpoint))
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.replay = ReplayBuffer(config.replay_size, self.state_shape, seed=config.seed)
        self.expert_replay = (
            ExpertReplayBuffer(config.expert_dataset_path, seed=config.seed)
            if config.use_expert_replay
            else None
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.expert_action_loss_fn = nn.CrossEntropyLoss()
        self.expert_bc_weight = max(0.0, float(expert_bc_weight))
        self.expert_pretrain_steps = max(0, int(expert_pretrain_steps))
        self.expert_pretrain_batch_size = max(1, int(expert_pretrain_batch_size))
        self.epsilon_schedule = LinearEpsilonScheduler(
            config.epsilon_start,
            config.epsilon_end,
            config.epsilon_decay_steps,
        )
        self.rng = np.random.default_rng(config.seed)

        self.global_step = 0
        self.episode_count = 0
        self.action_counts: Counter[int] = Counter()
        self.agent_episode_rewards: dict[str, float] = defaultdict(float)
        self.recent_episode_rewards = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)
        self.recent_expert_action_losses = deque(maxlen=100)
        self.best_episode_reward: float | None = None
        self.save_interval_seconds = float(save_interval_seconds)
        self.metrics_interval_steps = max(1, int(metrics_interval_steps))
        self.console_log_interval_steps = max(0, int(console_log_interval_steps))
        self.last_save_time = time.monotonic()
        self.last_metrics_step = -1
        self.last_console_log_step = 0
        self.last_console_action_counts: Counter[int] = Counter()

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"
        self.checkpoint_path = Path(config.checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        if self.expert_replay is not None and self.expert_pretrain_steps > 0:
            self.pretrain_on_expert()
            self.target_model.load_state_dict(self.model.state_dict())

    def load_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch_load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.loaded_checkpoint = True

    def handle_message(self, payload: dict[str, Any], connection_status: str) -> dict[str, Any] | None:
        message_type = payload.get("type")
        if message_type == "hello":
            print(
                "Unity connected:",
                payload.get("envId"),
                payload.get("observationFormat"),
                payload.get("shape"),
                flush=True,
            )
            return None
        if message_type == "observe":
            self.handle_observe(payload, connection_status)
            return None
        if message_type == "act":
            return self.handle_act(payload)
        return None

    def handle_observe(self, payload: dict[str, Any], connection_status: str) -> None:
        state = self.reshape_state(payload["state"])
        next_state = self.reshape_state(payload["nextState"])
        action = int(np.clip(int(payload["action"]), 0, self.env_config.action_count - 1))
        reward = float(payload["reward"])
        done = bool(payload.get("done", False))
        agent_id = str(payload.get("agentId", "Unknown"))

        self.replay.add(state, action, reward, next_state, done)
        self.global_step += 1
        self.action_counts[action] += 1
        self.agent_episode_rewards[agent_id] += reward

        if done:
            self.episode_count += 1
            episode_reward = self.agent_episode_rewards[agent_id]
            self.recent_episode_rewards.append(episode_reward)
            self.best_episode_reward = (
                episode_reward
                if self.best_episode_reward is None
                else max(self.best_episode_reward, episode_reward)
            )
            self.agent_episode_rewards[agent_id] = 0.0

        loss = self.train_if_ready()
        if loss is not None:
            self.recent_losses.append(loss)

        if self.global_step % self.config.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        now = time.monotonic()
        if now - self.last_save_time >= self.save_interval_seconds:
            self.save_checkpoint()
            self.last_save_time = now

        if self.global_step - self.last_metrics_step >= self.metrics_interval_steps:
            self.write_metrics(connection_status)
            self.last_metrics_step = self.global_step

        self.maybe_write_console_summary(connection_status)

    def handle_act(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.reshape_state(payload["state"])
        epsilon = self.epsilon_schedule.value(self.global_step) if bool(payload.get("epsilonAllowed", True)) else 0.0
        action = self.select_action(state, epsilon)
        return {
            "type": "action",
            "agentId": str(payload.get("agentId", "Unknown")),
            "action": int(action),
            "epsilon": float(epsilon),
            "globalStep": int(self.global_step),
        }

    def reshape_state(self, flat_values: list[float]) -> np.ndarray:
        state = np.asarray(flat_values, dtype=np.float32)
        expected = int(np.prod(self.state_shape))
        if state.size != expected:
            raise ValueError(f"Expected state length {expected}, received {state.size}.")
        return state.reshape(self.state_shape)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.env_config.action_count))
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.model(tensor)
            return int(q_values.argmax(dim=1).item())

    def train_if_ready(self) -> float | None:
        if len(self.replay) < self.config.min_replay_size:
            return None
        if self.global_step % self.config.train_every_steps != 0:
            return None

        progress = min(1.0, self.global_step / max(self.config.epsilon_decay_steps, 1))
        expert_ratio = expert_ratio_for_progress(progress, self.config) if self.expert_replay is not None else 0.0
        batch = sample_training_batch(self.replay, self.config.batch_size, self.expert_replay, expert_ratio)
        states, actions, rewards, next_states, dones = batch_to_tensors(batch, self.device)
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1).values
            target_q = rewards + self.config.gamma * (1.0 - dones) * next_q
        td_loss = self.loss_fn(q_values, target_q)
        loss = td_loss
        expert_action_loss_value: float | None = None
        if self.expert_replay is not None and self.expert_bc_weight > 0.0:
            expert_batch = self.expert_replay.sample(min(self.config.batch_size, self.expert_pretrain_batch_size))
            expert_states, expert_actions, _, _, _ = batch_to_tensors(expert_batch, self.device)
            expert_logits = self.model(expert_states)
            expert_action_loss = self.expert_action_loss_fn(expert_logits, expert_actions)
            loss = loss + self.expert_bc_weight * expert_action_loss
            expert_action_loss_value = float(expert_action_loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        if expert_action_loss_value is not None:
            self.recent_expert_action_losses.append(expert_action_loss_value)
        return float(loss.item())

    def pretrain_on_expert(self) -> None:
        if self.expert_replay is None:
            return

        print(
            f"Expert action pretrain: steps={self.expert_pretrain_steps}, "
            f"batch={self.expert_pretrain_batch_size}, dataset={len(self.expert_replay)}",
            flush=True,
        )
        self.model.train()
        for step in range(1, self.expert_pretrain_steps + 1):
            batch = self.expert_replay.sample(self.expert_pretrain_batch_size)
            states, actions, _, _, _ = batch_to_tensors(batch, self.device)
            logits = self.model(states)
            loss = self.expert_action_loss_fn(logits, actions)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.optimizer.step()
            if step == 1 or step == self.expert_pretrain_steps or step % 250 == 0:
                print(f"  pretrain step {step}/{self.expert_pretrain_steps}: action_loss={loss.item():.5f}", flush=True)

    def save_checkpoint(self) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_type": self.config.model_type,
                "input_channels": self.env_config.channel_count,
                "grid_size": self.env_config.grid_size,
                "grid_height": self.env_config.grid_height,
                "grid_width": self.env_config.grid_width,
                "action_count": self.env_config.action_count,
                "feature_dim": self.feature_dim,
                "live_training": True,
                "loaded_checkpoint": self.loaded_checkpoint,
                "expert_replay": self.config.use_expert_replay,
                "expert_bc_weight": self.expert_bc_weight,
                "expert_pretrain_steps": self.expert_pretrain_steps,
                "global_step": self.global_step,
                "episode_count": self.episode_count,
            },
            self.checkpoint_path,
        )
        print(f"Saved checkpoint to {self.checkpoint_path}", flush=True)

    def format_action_distribution(self, counts: Counter[int]) -> str:
        total = sum(counts.values())
        if total <= 0:
            return "none"

        parts: list[str] = []
        for action in range(self.env_config.action_count):
            label = self.ACTION_NAMES[action] if action < len(self.ACTION_NAMES) else f"A{action}"
            ratio = counts.get(action, 0) / total * 100.0
            parts.append(f"{label}:{ratio:.1f}%")
        return " ".join(parts)

    def maybe_write_console_summary(self, connection_status: str) -> None:
        if self.console_log_interval_steps <= 0:
            return
        if self.global_step - self.last_console_log_step < self.console_log_interval_steps:
            return

        recent_rewards = list(self.recent_episode_rewards)
        mean_reward_100 = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        max_reward_100 = float(max(recent_rewards)) if recent_rewards else 0.0
        best_reward = self.best_episode_reward if self.best_episode_reward is not None else 0.0
        running_reward = float(sum(self.agent_episode_rewards.values()))
        loss = float(np.mean(self.recent_losses)) if self.recent_losses else 0.0
        expert_action_loss = (
            float(np.mean(self.recent_expert_action_losses))
            if self.recent_expert_action_losses
            else 0.0
        )

        recent_action_counts: Counter[int] = Counter()
        for action in range(self.env_config.action_count):
            delta = self.action_counts[action] - self.last_console_action_counts[action]
            if delta > 0:
                recent_action_counts[action] = delta

        print(
            "[LiveDQN] "
            f"step={self.global_step} "
            f"episodes={self.episode_count} "
            f"replay={len(self.replay)} "
            f"eps={self.epsilon_schedule.value(self.global_step):.4f} "
            f"loss={loss:.5f} "
            f"reward100_mean={mean_reward_100:.3f} "
            f"reward100_max={max_reward_100:.3f} "
            f"best_reward={best_reward:.3f} "
            f"running_reward={running_reward:.3f} "
            f"expert_loss={expert_action_loss:.5f} "
            f"status={connection_status} "
            f"actions_recent={self.format_action_distribution(recent_action_counts)}",
            flush=True,
        )
        self.last_console_log_step = self.global_step
        self.last_console_action_counts = Counter(self.action_counts)

    def write_metrics(self, connection_status: str) -> None:
        row = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "replay_size": len(self.replay),
            "epsilon": self.epsilon_schedule.value(self.global_step),
            "loss": float(np.mean(self.recent_losses)) if self.recent_losses else 0.0,
            "mean_reward_100": float(np.mean(self.recent_episode_rewards)) if self.recent_episode_rewards else 0.0,
            "expert_action_loss": float(np.mean(self.recent_expert_action_losses))
            if self.recent_expert_action_losses
            else 0.0,
            "expert_bc_weight": self.expert_bc_weight,
            "expert_replay_size": len(self.expert_replay) if self.expert_replay is not None else 0,
            "action_counts": json.dumps(dict(sorted(self.action_counts.items())), ensure_ascii=False),
            "connection_status": connection_status,
        }
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def serve_forever(args: argparse.Namespace) -> None:
    env_config, train_config, feature_dim = build_live_config(args, load_yaml(Path(args.config)))
    trainer = LiveDqnTrainer(
        env_config,
        train_config,
        feature_dim,
        init_checkpoint=args.init_checkpoint,
        save_interval_seconds=args.save_interval_seconds,
        metrics_interval_steps=args.metrics_interval_steps,
        console_log_interval_steps=args.console_log_interval_steps,
        expert_bc_weight=args.expert_bc_weight,
        expert_pretrain_steps=args.expert_pretrain_steps,
        expert_pretrain_batch_size=args.expert_pretrain_batch_size,
    )

    print(
        f"Live CNN-DQN listening on {args.host}:{args.port}, "
        f"model_type={train_config.model_type}, device={trainer.device}, state_shape={trainer.state_shape}",
        flush=True,
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)

        try:
            while True:
                connection, address = server.accept()
                print(f"Unity client connected from {address}", flush=True)
                with connection:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    reader = connection.makefile("r", encoding="utf-8", newline="\n")
                    writer = connection.makefile("w", encoding="utf-8", newline="\n")
                    for line in reader:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            response = trainer.handle_message(json.loads(line), "connected")
                        except Exception as exception:
                            print(f"Live DQN message failed: {exception}", flush=True)
                            continue
                        if response is not None:
                            writer.write(json.dumps(response, separators=(",", ":")) + "\n")
                            writer.flush()
                print("Unity client disconnected", flush=True)
        except KeyboardInterrupt:
            print("Stopping live trainer...", flush=True)
        finally:
            trainer.save_checkpoint()
            trainer.write_metrics("stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-DQN live from Unity TCP JSONL transitions.")
    parser.add_argument("--config", default="configs/train_arena_semantic_screen_stack4_dqn.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--output-dir", default="runs/arena_live_stack4_dqn")
    parser.add_argument("--checkpoint-path", default="exports/arena_live_semantic_screen_stack4_dqn.pth")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--use-expert-replay", action="store_true")
    parser.add_argument("--expert-dataset")
    parser.add_argument("--expert-bc-weight", type=float, default=0.0)
    parser.add_argument("--expert-pretrain-steps", type=int, default=0)
    parser.add_argument("--expert-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-replay-size", type=int)
    parser.add_argument("--replay-size", type=int)
    parser.add_argument("--target-update-interval", type=int)
    parser.add_argument("--train-every-steps", type=int)
    parser.add_argument("--epsilon-decay-steps", type=int)
    parser.add_argument("--save-interval-seconds", type=float, default=300.0)
    parser.add_argument("--metrics-interval-steps", type=int, default=100)
    parser.add_argument(
        "--console-log-interval-steps",
        type=int,
        default=1000,
        help="Print a live training summary every N observed transitions. Set 0 to disable.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    serve_forever(args)


if __name__ == "__main__":
    main()
