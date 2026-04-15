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
        observation["basePosition"]["x"],
        observation["basePosition"]["y"],
        1.0 if observation["selfHasItem"] else 0.0,
        1.0 if observation["opponentHasItem"] else 0.0,
        1.0 if observation["isGrounded"] else 0.0,
        1.0 if observation["canDropDown"] else 0.0,
        observation["itemDelta"]["x"],
        observation["itemDelta"]["y"],
        observation["baseDelta"]["x"],
        observation["baseDelta"]["y"],
        observation["opponentDelta"]["x"],
        observation["opponentDelta"]["y"],
        observation["targetPosition"]["x"],
        observation["targetPosition"]["y"],
        float(observation["targetType"]),
        1.0 if observation["wallAhead"] else 0.0,
        1.0 if observation["hasGroundBelow"] else 0.0,
        float(observation.get("roundElapsedTime", 0.0)),
    ]


MOVE_TO_INDEX = {
    "Idle": 0,
    "MoveLeft": 1,
    "MoveRight": 2,
}


def normalize_move_action(move_action) -> int:
    if isinstance(move_action, int):
        return move_action

    if isinstance(move_action, str):
        return MOVE_TO_INDEX[move_action]

    raise KeyError(move_action)


def parse_step_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.lstrip("\ufeff").strip()
            if not line:
                continue

            payload = json.loads(line)
            if "observation" not in payload or "action" not in payload:
                continue

            rows.append(payload)

    return rows


def collect_log_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))

    return [input_path]


def downsample_idle_rows(rows: list[dict], keep_ratio: float) -> list[dict]:
    if keep_ratio >= 1.0:
        return rows

    rng = np.random.default_rng(42)
    filtered: list[dict] = []
    for row in rows:
        action = row["action"]
        move_action = normalize_move_action(action["moveAction"])
        is_idle = (
            move_action == MOVE_TO_INDEX["Idle"]
            and not action["jumpPressed"]
            and not action["dropPressed"]
            and not action.get("shovePressed", False)
        )
        if not is_idle or rng.random() <= keep_ratio:
            filtered.append(row)

    return filtered


def build_sequence_features(rows: list[dict], sequence_length: int) -> tuple[list[list[float]], list[dict]]:
    features: list[list[float]] = []
    targets: list[dict] = []
    frame_features = [build_feature_vector(row["observation"]) for row in rows]

    for index, row in enumerate(rows):
        start_index = max(0, index - sequence_length + 1)
        window = frame_features[start_index:index + 1]
        while len(window) < sequence_length:
            window.insert(0, window[0])

        flattened: list[float] = []
        for feature_vector in window:
            flattened.extend(feature_vector)

        features.append(flattened)
        targets.append(row["action"])

    return features, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Deep AI Arena logs into a numpy dataset.")
    parser.add_argument("--input", required=True, help="Path to a round log file or a directory of round log files")
    parser.add_argument("--output", required=True, help="Output NPZ dataset path")
    parser.add_argument("--idle-keep-ratio", type=float, default=0.25, help="Ratio of idle-only rows to keep")
    parser.add_argument("--sequence-length", type=int, default=4, help="Number of consecutive frames to flatten into one training sample")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    move_targets = []
    jump_targets = []
    drop_targets = []
    shove_targets = []
    processed_rounds = 0

    for log_file in collect_log_files(input_path):
        rows = parse_step_rows(log_file)
        if not rows:
            continue

        rows = downsample_idle_rows(rows, args.idle_keep_ratio)
        if not rows:
            continue

        sequence_features, sequence_actions = build_sequence_features(rows, args.sequence_length)
        features.extend(sequence_features)

        for action in sequence_actions:
            move_targets.append(normalize_move_action(action["moveAction"]))
            jump_targets.append(1.0 if action["jumpPressed"] else 0.0)
            drop_targets.append(1.0 if action["dropPressed"] else 0.0)
            shove_targets.append(1.0 if action.get("shovePressed", False) else 0.0)

        processed_rounds += 1

    if not features:
        raise RuntimeError(f"No usable step rows found in {input_path}")

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
    print(f"Rounds: {processed_rounds}")
    print(f"Sequence length: {args.sequence_length}")


if __name__ == "__main__":
    main()
