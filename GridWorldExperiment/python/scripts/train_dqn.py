from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from envs.grid_env import GridWorldEnv
from rl.dqn_trainer import DQNTrainConfig, train_dqn


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_config(args: argparse.Namespace, payload: dict) -> tuple[GridWorldEnv, DQNTrainConfig, int]:
    env_path = Path(payload["env"]["config_path"])
    env = GridWorldEnv(env_path)
    model_payload = payload["model"]
    dqn_payload = payload["dqn"]
    epsilon_payload = payload["epsilon"]
    logging_payload = payload["logging"]
    expert_payload = payload["expert_replay"]

    init_encoder_path = args.init_encoder_path
    use_expert_replay = args.use_expert_replay
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path
    model_type = args.model_type or str(model_payload.get("model_type", "cnn"))

    if args.mode == "mlp_pure":
        model_type = "mlp"
        output_dir = output_dir or "runs/dqn_mlp"
        checkpoint_path = checkpoint_path or "exports/dqn_mlp.pth"
    elif args.mode == "cnn_pure":
        model_type = "cnn"
        output_dir = output_dir or "runs/dqn_pure"
        checkpoint_path = checkpoint_path or "exports/dqn_pure.pth"
    elif args.mode == "pure":
        output_dir = output_dir or "runs/dqn_pure"
        checkpoint_path = checkpoint_path or "exports/dqn_pure.pth"
    elif args.mode == "bc_init":
        model_type = "cnn"
        init_encoder_path = init_encoder_path or "exports/bc_encoder.pth"
        output_dir = output_dir or "runs/dqn_bc_init"
        checkpoint_path = checkpoint_path or "exports/dqn_bc_init.pth"
    elif args.mode == "expert":
        init_encoder_path = init_encoder_path or "exports/bc_encoder.pth"
        use_expert_replay = True
        output_dir = output_dir or "runs/dqn_bc_expert_replay"
        checkpoint_path = checkpoint_path or "exports/dqn_bc_expert_replay.pth"
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    config = DQNTrainConfig(
        model_type=model_type,
        total_episodes=args.episodes or int(dqn_payload["total_episodes"]),
        max_steps_per_episode=int(dqn_payload["max_steps_per_episode"]),
        batch_size=args.batch_size or int(dqn_payload["batch_size"]),
        gamma=float(dqn_payload["gamma"]),
        learning_rate=args.lr or float(dqn_payload["learning_rate"]),
        replay_size=int(dqn_payload["replay_size"]),
        min_replay_size=args.min_replay_size or int(dqn_payload["min_replay_size"]),
        target_update_interval=int(dqn_payload["target_update_interval"]),
        train_every_steps=int(dqn_payload["train_every_steps"]),
        gradient_clip_norm=float(dqn_payload["gradient_clip_norm"]),
        epsilon_start=float(epsilon_payload["start"]),
        epsilon_end=float(epsilon_payload["end"]),
        epsilon_decay_steps=int(epsilon_payload["decay_steps"]),
        eval_interval_episodes=args.eval_interval or int(logging_payload["eval_interval_episodes"]),
        eval_episodes=args.eval_episodes or int(logging_payload["eval_episodes"]),
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        init_encoder_path=init_encoder_path,
        use_expert_replay=use_expert_replay,
        expert_dataset_path=args.expert_dataset_path or str(expert_payload["dataset_path"]),
        expert_ratio_start=float(expert_payload["ratio_start"]),
        expert_ratio_mid=float(expert_payload["ratio_mid"]),
        expert_ratio_end=float(expert_payload["ratio_end"]),
        seed=args.seed or int(payload["seed"]),
        cpu=args.cpu,
    )
    return env, config, int(model_payload.get("feature_dim", 256))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GridWorld DQN variants.")
    parser.add_argument("--config", default="configs/train_dqn.yaml")
    parser.add_argument("--mode", choices=["pure", "mlp_pure", "cnn_pure", "bc_init", "expert"], default="cnn_pure")
    parser.add_argument("--model-type", choices=["cnn", "mlp"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min-replay-size", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init-encoder-path")
    parser.add_argument("--use-expert-replay", action="store_true")
    parser.add_argument("--expert-dataset-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    env, config, feature_dim = _build_config(args, _load_yaml(Path(args.config)))
    train_dqn(env, config, feature_dim=feature_dim)


if __name__ == "__main__":
    main()
