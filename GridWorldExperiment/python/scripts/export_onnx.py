from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from models.dqn import CNNDQN, DuelingCNNDQN, DuelingNatureDQN, MLPDQN, NatureDQN, ResNetDQN


def build_model_from_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    model_type = checkpoint.get("model_type", "cnn")
    input_channels = int(checkpoint.get("input_channels", 10))
    grid_size = int(checkpoint.get("grid_size", 24))
    grid_height = int(checkpoint.get("grid_height", grid_size))
    grid_width = int(checkpoint.get("grid_width", grid_size))
    action_count = int(checkpoint.get("action_count", 4))
    feature_dim = int(checkpoint.get("feature_dim", 256))
    model_type = str(model_type).lower()
    if model_type == "mlp":
        model = MLPDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("nature", "nature_cnn", "nature_dqn"):
        model = NatureDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("dueling", "dueling_cnn"):
        model = DuelingCNNDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("dueling_nature", "dueling_nature_cnn"):
        model = DuelingNatureDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    elif model_type in ("resnet", "residual_cnn"):
        model = ResNetDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    else:
        model = CNNDQN(input_channels, grid_size, grid_height, grid_width, action_count, feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_channels, grid_height, grid_width, action_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GridWorld DQN checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="exports/dqn_bc_expert_replay.pth")
    parser.add_argument("--output", default="exports/dqn_model.onnx")
    parser.add_argument("--opset", type=int, default=15)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model, input_channels, grid_height, grid_width, action_count = build_model_from_checkpoint(checkpoint_path)
    dummy = torch.zeros((1, input_channels, grid_height, grid_width), dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["q_values"],
        dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    with torch.no_grad():
        pytorch_output = model(dummy)
    if tuple(pytorch_output.shape) != (1, action_count):
        raise RuntimeError(f"Unexpected PyTorch output shape: {tuple(pytorch_output.shape)}")

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"state": dummy.numpy()})[0]
        max_diff = abs(onnx_output - pytorch_output.numpy()).max()
        print(f"ONNX Runtime max abs diff: {max_diff:.6f}")
    except ModuleNotFoundError:
        print("onnxruntime is not installed; skipped runtime output comparison.")

    print(f"Exported ONNX model to {output_path}")
    print(f"Input shape: [1,{input_channels},{grid_height},{grid_width}], output shape: [1,{action_count}]")


if __name__ == "__main__":
    main()
