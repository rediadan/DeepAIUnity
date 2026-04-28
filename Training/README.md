# Deep AI Arena Training

This folder trains top-view arena agents from player logs.

## Current Input Features

Each frame uses 21 features.

- self position
- opponent position
- item position
- shared base position
- self has item
- opponent has item
- item delta
- base delta
- opponent delta
- target position
- target type
- wall ahead
- round elapsed time

With `--sequence-length 4`, the model input size is `21 * 4 = 84`.

## BC Training

Behavior Cloning learns the player's top-view movement and shove timing.

```cmd
python preprocess_logs.py --input "dataset" --output "data\arena_dataset.npz" --sequence-length 4

python train_behavior_cloning.py --dataset "data\arena_dataset.npz" --output "models\arena_bc.pt" --epochs 30 --batch-size 128 --lr 0.001 --hidden-size 64

python export_onnx.py --checkpoint "models\arena_bc.pt" --output "models\arena_bc.onnx"
```

BC ONNX outputs:

- `move_logits`: `Idle / MoveUp / MoveDown / MoveLeft / MoveRight`
- `shove_prob`

## DQN Training

DQN uses transition rows from the same round logs.

```cmd
python preprocess_dqn_transitions.py --input "dataset" --output "data\arena_dqn_dataset.npz" --sequence-length 4

python train_dqn.py --dataset "data\arena_dqn_dataset.npz" --output "models\arena_dqn.pt" --bc-checkpoint "models\arena_bc.pt" --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97

python export_dqn_onnx.py --checkpoint "models\arena_dqn.pt" --output "models\arena_dqn.onnx"
```

DQN ONNX outputs:

- `q_values`: six values for `Idle / Up / Down / Left / Right / Shove`

## Unity Setup

Put the exported `.onnx` and `.stats.json` files under `Assets`, then assign them to `ArenaGhostOnnxPolicy`.

- `RuleBased`: collect logs
- `OnnxInference`: run BC
- `DqnInference`: run DQN
