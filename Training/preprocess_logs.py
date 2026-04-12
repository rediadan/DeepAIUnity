import argparse
import json
from pathlib import Path

import numpy as np


def build_feature_vector(observation: dict) -> list[float]:
    return [
        observation["selfPosition"]["x"],
        observation["selfPosition"]["y"],
        observation["opponentPosition"]["x"],
        observation["opponentPosition"]["y"],
        observation["itemPosition"]["x"],
        observation["itemPosition"]["y"],
        observation["selfVelocity"]["x"],
        observation["selfVelocity"]["y"],
        observation["opponentVelocity"]["x"],
        observation["opponentVelocity"]["y"],
        observation["selfBasePosition"]["x"],
        observation["selfBasePosition"]["y"],
        observation["opponentBasePosition"]["x"],
        observation["opponentBasePosition"]["y"],
        1.0 if observation["selfHasItem"] else 0.0,
        1.0 if observation["opponentHasItem"] else 0.0,
        1.0 if observation["isGrounded"] else 0.0,
        1.0 if observation["opponentGrounded"] else 0.0,
        float(observation["itemLane"]),
    ]


MOVE_TO_INDEX = {
    "Idle": 0,
    "MoveLeft": 1,
    "MoveRight": 2,
}


def parse_step_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if "observation" not in payload or "action" not in payload:
                continue

            rows.append(payload)

    return rows


def downsample_idle_rows(rows: list[dict], keep_ratio: float) -> list[dict]:
    if keep_ratio >= 1.0:
        return rows

    rng = np.random.default_rng(42)
    filtered: list[dict] = []
    for row in rows:
        action = row["action"]
        is_idle = (
            action["moveAction"] == "Idle"
            and not action["jumpPressed"]
            and not action["dropPressed"]
            and not action.get("shovePressed", False)
        )
        if not is_idle or rng.random() <= keep_ratio:
            filtered.append(row)

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Deep AI Arena logs into a numpy dataset.")
    parser.add_argument("--input", required=True, help="Path to arena_training_log.jsonl")
    parser.add_argument("--output", required=True, help="Output NPZ dataset path")
    parser.add_argument("--idle-keep-ratio", type=float, default=0.25, help="Ratio of idle-only rows to keep")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_step_rows(input_path)
    if not rows:
        raise RuntimeError(f"No step rows found in {input_path}")

    rows = downsample_idle_rows(rows, args.idle_keep_ratio)

    features = []
    move_targets = []
    jump_targets = []
    drop_targets = []
    shove_targets = []

    for row in rows:
        observation = row["observation"]
        action = row["action"]
        features.append(build_feature_vector(observation))
        move_targets.append(MOVE_TO_INDEX[action["moveAction"]])
        jump_targets.append(1.0 if action["jumpPressed"] else 0.0)
        drop_targets.append(1.0 if action["dropPressed"] else 0.0)
        shove_targets.append(1.0 if action.get("shovePressed", False) else 0.0)

    x = np.asarray(features, dtype=np.float32)
    move = np.asarray(move_targets, dtype=np.int64)
    jump = np.asarray(jump_targets, dtype=np.float32)
    drop = np.asarray(drop_targets, dtype=np.float32)
    shove = np.asarray(shove_targets, dtype=np.float32)

    feature_mean = x.mean(axis=0)
    feature_std = x.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    np.savez(
        output_path,
        x=x,
        move=move,
        jump=jump,
        drop=drop,
        shove=shove,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    print(f"Saved dataset to {output_path}")
    print(f"Samples: {len(x)}")
    print(f"Features: {x.shape[1]}")


if __name__ == "__main__":
    main()
