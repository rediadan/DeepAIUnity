# Deep AI Arena DQN Guide

This pipeline uses top-view arena logs. The agent observes positions, target information, wall sensing, and elapsed round time.

## Action Set

- `0`: `Idle`
- `1`: `MoveUp`
- `2`: `MoveDown`
- `3`: `MoveLeft`
- `4`: `MoveRight`
- `5`: `Shove`

## Dataset

Unity writes round logs that include both BC step rows and DQN transition rows.

```cmd
python preprocess_dqn_transitions.py --input "dataset" --output "data\arena_dqn_dataset.npz" --sequence-length 4
```

## Train

Use the BC checkpoint as the initial backbone when you want the DQN model to preserve player-like movement.

```cmd
python train_dqn.py --dataset "data\arena_dqn_dataset.npz" --output "models\arena_dqn.pt" --bc-checkpoint "models\arena_bc.pt" --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97
```

## Export

```cmd
python export_dqn_onnx.py --checkpoint "models\arena_dqn.pt" --output "models\arena_dqn.onnx"
```

The exported model returns `q_values`. Unity selects the highest-value action and converts it to top-view movement.
