import argparse
from pathlib import Path

import torch
from torch import nn


class BehaviorCloningExportModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
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
        return (
            self.move_head(hidden),
            torch.sigmoid(self.jump_head(hidden)),
            torch.sigmoid(self.drop_head(hidden)),
            torch.sigmoid(self.shove_head(hidden)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Deep AI Arena BC checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", required=True, help="Path to output .onnx file")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = BehaviorCloningExportModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, checkpoint["input_size"], dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["observation"],
        output_names=["move_logits", "jump_prob", "drop_prob", "shove_prob"],
        dynamic_axes={"observation": {0: "batch"}},
        opset_version=17,
    )

    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
