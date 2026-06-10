from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from envs.grid_env import ACTION_DELTAS, GridWorldEnv
from models.dqn import CNNDQN, MLPDQN
from rl.epsilon_scheduler import LinearEpsilonScheduler
from rl.expert_replay_buffer import ExpertReplayBuffer
from rl.replay_buffer import ReplayBatch, ReplayBuffer, concat_batches


@dataclass(frozen=True)
class DQNTrainConfig:
    model_type: str = "cnn"
    total_episodes: int = 2000
    max_steps_per_episode: int = 200
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 1e-4
    replay_size: int = 100000
    min_replay_size: int = 2000
    target_update_interval: int = 1000
    train_every_steps: int = 4
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    eval_interval_episodes: int = 50
    eval_episodes: int = 30
    output_dir: str = "runs/dqn_pure"
    checkpoint_path: str = "exports/dqn_pure.pth"
    init_encoder_path: str | None = None
    use_expert_replay: bool = False
    expert_dataset_path: str = "datasets/demonstrations/demo_train.npz"
    expert_ratio_start: float = 0.5
    expert_ratio_mid: float = 0.25
    expert_ratio_end: float = 0.1
    seed: int = 42
    cpu: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    input_channels: int,
    grid_size: int,
    action_count: int,
    feature_dim: int,
    grid_height: int | None = None,
    grid_width: int | None = None,
) -> nn.Module:
    if model_type == "cnn":
        return CNNDQN(
            input_channels=input_channels,
            grid_size=grid_size,
            grid_height=grid_height,
            grid_width=grid_width,
            action_count=action_count,
            feature_dim=feature_dim,
        )
    if model_type == "mlp":
        return MLPDQN(
            input_channels=input_channels,
            grid_size=grid_size,
            grid_height=grid_height,
            grid_width=grid_width,
            action_count=action_count,
            hidden_size=feature_dim,
        )
    raise ValueError(f"Unsupported DQN model_type: {model_type}")


def load_encoder_if_available(model: nn.Module, init_encoder_path: str | None) -> bool:
    if not init_encoder_path:
        return False
    if not hasattr(model, "encoder"):
        raise ValueError("Encoder initialization is only supported for CNN DQN models.")
    path = Path(init_encoder_path)
    if not path.exists():
        raise FileNotFoundError(path)
    state_dict = torch.load(path, map_location="cpu")
    model.encoder.load_state_dict(state_dict)
    return True


def greedy_rule_action(env: GridWorldEnv) -> int:
    nearest = env.nearest_coin()
    if nearest is None:
        return 0

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


def select_action(
    model: nn.Module,
    state: np.ndarray,
    epsilon: float,
    action_count: int,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, action_count))
    with torch.no_grad():
        tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        q_values = model(tensor)
        return int(q_values.argmax(dim=1).item())


def batch_to_tensors(batch: ReplayBatch, device: torch.device):
    return (
        torch.from_numpy(batch.states).to(device),
        torch.from_numpy(batch.actions).long().to(device),
        torch.from_numpy(batch.rewards).to(device),
        torch.from_numpy(batch.next_states).to(device),
        torch.from_numpy(batch.dones).to(device),
    )


def expert_ratio_for_progress(progress: float, config: DQNTrainConfig) -> float:
    if progress < 0.2:
        return config.expert_ratio_start
    if progress < 0.6:
        return config.expert_ratio_mid
    return config.expert_ratio_end


def sample_training_batch(
    replay: ReplayBuffer,
    batch_size: int,
    expert_replay: ExpertReplayBuffer | None,
    expert_ratio: float,
) -> ReplayBatch:
    if expert_replay is None or expert_ratio <= 0.0:
        return replay.sample(batch_size)

    expert_count = min(batch_size - 1, max(1, int(round(batch_size * expert_ratio))))
    agent_count = max(1, batch_size - expert_count)
    return concat_batches(replay.sample(agent_count), expert_replay.sample(expert_count))


def evaluate_model(
    env: GridWorldEnv,
    model: nn.Module | None,
    episodes: int,
    seed_start: int,
    device: torch.device,
    policy: str = "dqn",
) -> dict[str, float]:
    returns = []
    successes = []
    coins = []
    deaths = []
    wall_hits = []
    trap_hits = []
    survival_steps = []
    for episode in range(episodes):
        state = env.reset(seed=seed_start + episode)
        done = False
        total_reward = 0.0
        wall_hit_count = 0
        trap_hit_count = 0
        death = False
        while not done:
            if policy == "random":
                action = int(env.rng.integers(0, env.config.action_count))
            elif policy == "rule":
                action = greedy_rule_action(env)
            else:
                if model is None:
                    raise ValueError("DQN policy requires a model.")
                action = select_action(model, state, 0.0, env.config.action_count, device, env.rng)
            result = env.step(action)
            total_reward += result.reward
            wall_hit_count += 1 if result.info.get("hit_wall", False) else 0
            trap_hit_count += 1 if result.info.get("hit_trap", False) else 0
            death = death or bool(result.info.get("death", False))
            state = result.state
            done = result.done

        returns.append(total_reward)
        successes.append(1.0 if env.score >= env.config.target_score else 0.0)
        coins.append(float(env.score))
        deaths.append(1.0 if death else 0.0)
        wall_hits.append(float(wall_hit_count))
        trap_hits.append(float(trap_hit_count))
        survival_steps.append(float(env.steps))

    return {
        "average_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "average_coins": float(np.mean(coins)),
        "death_rate": float(np.mean(deaths)),
        "wall_hit_count": float(np.mean(wall_hits)),
        "trap_hit_count": float(np.mean(trap_hits)),
        "average_survival_steps": float(np.mean(survival_steps)),
    }


def train_dqn(env: GridWorldEnv, config: DQNTrainConfig, feature_dim: int = 256) -> None:
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    grid_height = int(getattr(env.config, "grid_height", env.config.grid_size))
    grid_width = int(getattr(env.config, "grid_width", env.config.grid_size))
    state_shape = (env.config.channel_count, grid_height, grid_width)
    model = build_model(
        config.model_type,
        env.config.channel_count,
        env.config.grid_size,
        env.config.action_count,
        feature_dim,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    target_model = build_model(
        config.model_type,
        env.config.channel_count,
        env.config.grid_size,
        env.config.action_count,
        feature_dim,
        grid_height=grid_height,
        grid_width=grid_width,
    )
    encoder_loaded = load_encoder_if_available(model, config.init_encoder_path)
    target_model.load_state_dict(model.state_dict())
    model.to(device)
    target_model.to(device)
    target_model.eval()

    replay = ReplayBuffer(config.replay_size, state_shape, seed=config.seed)
    expert_replay = ExpertReplayBuffer(config.expert_dataset_path, seed=config.seed) if config.use_expert_replay else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    epsilon_schedule = LinearEpsilonScheduler(config.epsilon_start, config.epsilon_end, config.epsilon_decay_steps)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    rows: list[dict[str, float | int | str | bool]] = []
    global_step = 0
    recent_returns: list[float] = []
    recent_losses: list[float] = []
    best_eval_return = -float("inf")

    for episode in range(1, config.total_episodes + 1):
        state = env.reset(seed=config.seed + episode)
        episode_return = 0.0
        done = False
        step = 0
        while not done and step < config.max_steps_per_episode:
            epsilon = epsilon_schedule.value(global_step)
            action = select_action(model, state, epsilon, env.config.action_count, device, rng)
            result = env.step(action)
            replay.add(state, action, result.reward, result.state, result.done)
            state = result.state
            episode_return += result.reward
            done = result.done
            step += 1
            global_step += 1

            if len(replay) >= config.min_replay_size and global_step % config.train_every_steps == 0:
                progress = episode / max(config.total_episodes, 1)
                expert_ratio = expert_ratio_for_progress(progress, config) if expert_replay is not None else 0.0
                batch = sample_training_batch(replay, config.batch_size, expert_replay, expert_ratio)
                states, actions, rewards, next_states, dones = batch_to_tensors(batch, device)
                q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_model(next_states).max(dim=1).values
                    target_q = rewards + config.gamma * (1.0 - dones) * next_q
                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                recent_losses.append(float(loss.item()))

            if global_step % config.target_update_interval == 0:
                target_model.load_state_dict(model.state_dict())

        recent_returns.append(float(episode_return))
        if len(recent_returns) > 100:
            recent_returns.pop(0)
        if len(recent_losses) > 100:
            recent_losses = recent_losses[-100:]

        should_eval = episode == 1 or episode % config.eval_interval_episodes == 0 or episode == config.total_episodes
        if should_eval:
            eval_stats = evaluate_model(
                env,
                model,
                episodes=config.eval_episodes,
                seed_start=100000 + episode * 100,
                device=device,
            )
            mean_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
            row = {
                "episode": episode,
                "global_step": global_step,
                "epsilon": epsilon_schedule.value(global_step),
                "episode_return": float(episode_return),
                "moving_average_return": float(np.mean(recent_returns)),
                "loss": mean_loss,
                "eval_average_return": eval_stats["average_return"],
                "eval_success_rate": eval_stats["success_rate"],
                "eval_average_coins": eval_stats["average_coins"],
                "eval_death_rate": eval_stats["death_rate"],
                "eval_wall_hit_count": eval_stats["wall_hit_count"],
                "eval_average_survival_steps": eval_stats["average_survival_steps"],
                "model_type": config.model_type,
                "encoder_loaded": encoder_loaded,
                "expert_replay": config.use_expert_replay,
            }
            rows.append(row)
            print(
                f"episode={episode} step={global_step} "
                f"eval_return={eval_stats['average_return']:.3f} "
                f"success={eval_stats['success_rate']:.3f} epsilon={row['epsilon']:.3f}"
            )

            if eval_stats["average_return"] >= best_eval_return:
                best_eval_return = eval_stats["average_return"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_type": config.model_type,
                        "input_channels": env.config.channel_count,
                        "grid_size": env.config.grid_size,
                        "grid_height": grid_height,
                        "grid_width": grid_width,
                        "action_count": env.config.action_count,
                        "feature_dim": feature_dim,
                        "encoder_loaded": encoder_loaded,
                        "expert_replay": config.use_expert_replay,
                        "best_eval_return": best_eval_return,
                    },
                    checkpoint_path,
                )

            write_metrics(metrics_path, rows)

    print(f"Saved best checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


def write_metrics(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
