# Deep AI Arena Training Pipeline

This folder contains a minimal behavior-cloning pipeline for the Unity prototype.

## Flow

1. Play the game in Unity and produce `arena_training_log.jsonl`.
2. Convert the JSONL log into a compact dataset.
3. Train a behavior-cloning model.
4. Export the trained checkpoint to ONNX.

## Expected log location

The runtime logger writes to:

- `Application.persistentDataPath/arena_training_log.jsonl`

Copy that file into `Training/data/` or pass the full path directly.

## 1. Preprocess logs

```bash
python preprocess_logs.py --input "C:/path/to/arena_training_log.jsonl" --output "data/arena_dataset.npz"
```

Notes:

- Reward-only lines are ignored automatically.
- Only player step logs are expected now.
- Idle-only frames are downsampled by default to reduce dataset imbalance.

## 2. Train behavior cloning

```bash
python train_behavior_cloning.py --dataset "data/arena_dataset.npz" --output "models/arena_bc.pt"
```

Optional arguments:

- `--epochs`
- `--batch-size`
- `--lr`
- `--hidden-size`

## 3. Export ONNX

```bash
python export_onnx.py --checkpoint "models/arena_bc.pt" --output "models/arena_bc.onnx"
```

## Python dependencies

Recommended packages:

```bash
pip install numpy torch
```

If ONNX export is needed:

```bash
pip install onnx
```

## Output heads

The model predicts:

- move logits: `Idle / MoveLeft / MoveRight`
- jump probability
- drop probability
- shove probability

## Next Unity-side step

After ONNX export, the next implementation step is:

- load the ONNX model in Unity
- normalize observations using the saved mean/std from the checkpoint
- replace Ghost rule decisions with model inference
