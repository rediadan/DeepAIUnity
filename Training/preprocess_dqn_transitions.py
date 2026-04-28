import argparse
import json
from pathlib import Path

import numpy as np

from preprocess_logs import build_feature_vector, collect_log_files


def parse_transition_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.lstrip("\ufeff").strip()
            if not line:
                continue

            payload = json.loads(line)
            if payload.get("transitionType") != "DqnTransition":
                continue

            rows.append(payload)

    return rows


def pad_window(window: list[list[float]], sequence_length: int) -> list[list[float]]:
    if not window:
        raise ValueError("Cannot pad an empty sequence window.")

    while len(window) < sequence_length:
        window.insert(0, window[0])

    return window


def flatten_window(window: list[list[float]]) -> list[float]:
    flattened: list[float] = []
    for frame in window:
        flattened.extend(frame)
    return flattened


def build_transition_sequences(rows: list[dict], sequence_length: int):
    state_frames = [build_feature_vector(row["state"]) for row in rows]
    next_state_frames = [build_feature_vector(row["nextState"]) for row in rows]

    states = []
    next_states = []
    actions = []
    rewards = []
    dones = []

    for index, row in enumerate(rows):
        start_index = max(0, index - sequence_length + 1)
        state_window = pad_window(state_frames[start_index:index + 1], sequence_length)
        next_state_window = state_window[1:] + [next_state_frames[index]]

        states.append(flatten_window(state_window))
        next_states.append(flatten_window(next_state_window))
        actions.append(int(row["action"]))
        rewards.append(float(row["reward"]))
        dones.append(1.0 if row["done"] else 0.0)

    return states, actions, rewards, next_states, dones


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Deep AI Arena DQN transition logs.")
    parser.add_argument("--input", required=True, help="Path to a round log file or a directory of round log files")
    parser.add_argument("--output", required=True, help="Output NPZ replay dataset path")
    parser.add_argument("--sequence-length", type=int, default=4, help="Number of consecutive frames per DQN state")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    processed_rounds = 0

    for log_file in collect_log_files(input_path):
        rows = parse_transition_rows(log_file)
        if not rows:
            continue

        round_states, round_actions, round_rewards, round_next_states, round_dones = build_transition_sequences(
            rows,
            args.sequence_length,
        )
        states.extend(round_states)
        actions.extend(round_actions)
        rewards.extend(round_rewards)
        next_states.extend(round_next_states)
        dones.extend(round_dones)
        processed_rounds += 1

    if not states:
        raise RuntimeError(f"No DQN transitions found in {input_path}")

    state = np.asarray(states, dtype=np.float32)
    next_state = np.asarray(next_states, dtype=np.float32)
    action = np.asarray(actions, dtype=np.int64)
    reward = np.asarray(rewards, dtype=np.float32)
    done = np.asarray(dones, dtype=np.float32)

    combined = np.concatenate([state, next_state], axis=0)
    feature_mean = combined.mean(axis=0)
    feature_std = combined.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0

    np.savez(
        output_path,
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
    )

    print(f"Saved DQN dataset to {output_path}")
    print(f"Transitions: {len(state)}")
    print(f"Features: {state.shape[1]}")
    print(f"Rounds: {processed_rounds}")
    print(f"Sequence length: {args.sequence_length}")


if __name__ == "__main__":
    main()
