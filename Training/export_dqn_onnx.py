import argparse
import json
from pathlib import Path

import torch

from train_dqn import DqnModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Deep AI Arena DQN checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained DQN checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_size = int(checkpoint["input_size"])
    hidden_size = int(checkpoint["hidden_size"])
    action_count = int(checkpoint.get("action_count", 10))

    model = DqnModel(input_size=input_size, hidden_size=hidden_size, action_count=action_count)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.zeros(1, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes={
            "observation": {0: "batch"},
            "q_values": {0: "batch"},
        },
        opset_version=17,
    )

    stats_path = output_path.with_suffix(".stats.json")
    stats_payload = {
        "feature_mean": checkpoint["feature_mean"].tolist(),
        "feature_std": checkpoint["feature_std"].tolist(),
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    print(f"Exported DQN ONNX to {output_path}")
    print(f"Exported stats to {stats_path}")
    print(f"Input size: {input_size}")
    print(f"Action count: {action_count}")


if __name__ == "__main__":
    main()
