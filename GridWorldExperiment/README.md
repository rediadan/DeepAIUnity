# GridWorldExperiment

This experiment supports the presentation claim:

> A non-grid top-down game state can be rasterized into a multi-channel feature map, and CNN-DQN can use that spatial representation more directly than a flattened MLP-DQN. Behavioral cloning initialization is tested as a way to reduce early DQN sample inefficiency.

## Environment

- Game world: continuous 2D top-down square world.
- Observation: rasterized feature map, not the game rule itself.
- State shape: `[C,H,W] = [10,24,24]`.
- Actions: `Up`, `Down`, `Left`, `Right`.

Observation channels:

```text
player
coin
enemy
wall
trap
visited
danger
enemy_previous
enemy_predicted
target_coin
```

## Main Comparison

The presentation comparison is intentionally limited to three agents:

```text
MLP-DQN
CNN-DQN
BC Init + CNN-DQN
```

Rule-Based control is used as the expert source for demonstration data, not as the main claim.

## Quick Verification

```powershell
cd C:\Users\User\Documents\신경망과딥러닝\GridWorldExperiment
run_smoke_pipeline.bat
```

The smoke pipeline runs:

1. Generate a small rule-based demonstration file.
2. Convert JSONL demonstrations to NPZ.
3. Train a short BC model.
4. Train a short CNN-DQN model.
5. Train a short BC-initialized CNN-DQN model.
6. Export a smoke ONNX model.

## Full Experiment

```powershell
run_full_experiments.bat
```

The full pipeline trains:

```text
exports/dqn_mlp.pth
exports/dqn_pure.pth
exports/dqn_bc_init.pth
exports/dqn_model.onnx
```

Important outputs:

```text
runs/dqn_mlp/metrics.csv
runs/dqn_pure/metrics.csv
runs/dqn_bc_init/metrics.csv
runs/evaluation_summary.csv
runs/plots/average_return_curve.png
runs/plots/success_rate_curve.png
runs/plots/time_to_threshold.png
runs/plots/final_evaluation_bar_chart.png
```

## Unity

In Unity, create the demo scene:

```text
Tools > GridWorld > Create Demo Scene
```

The scene displays continuous objects. The 24x24 grid is used as the DQN observation parser, and can be enabled as a debug overlay from the `GridWorldManager` Inspector.

For ONNX inference, attach `SentisDqnAgent` to the `GridWorld` object and assign a model exported from this pipeline. The model must use input shape:

```text
[1,10,24,24]
```

## Live CNN-DQN For DeepAIArena

Reward details are summarized in `docs/arena_reward_system.md`.

Run the Python trainer first:

```bat
cd /d E:\GameDevelopingFile\UniversalRandomPicker\GridWorldExperiment

python python\scripts\train_unity_live_dqn.py ^
  --config configs\train_arena_semantic_screen_stack4_dqn.yaml ^
  --host 127.0.0.1 ^
  --port 5055 ^
  --use-expert-replay ^
  --expert-dataset datasets\arena_semantic_screen_stack4\demo_train.npz ^
  --checkpoint-path exports\arena_live_semantic_screen_stack4_dqn.pth ^
  --output-dir runs\arena_live_stack4_dqn
```

Then in Unity:

```text
ArenaRasterEncoder > Input Mode = SemanticRgbFrameStack4
Left/Right Actor Mode = LiveCnnDqnTraining
ArenaLiveDqnTrainingClient > Accelerate Training = On
ArenaLiveDqnTrainingClient > Training Time Scale = 8
```

Unity sends `[12,32,48]` semantic-screen transitions over TCP JSONL. The 12 channels are four `[3,32,48]` RGB semantic frames ordered from oldest to newest, which lets the CNN-DQN infer motion from moving obstacles and opponent movement. Python updates a shared CNN-DQN policy and saves `.pth` checkpoints. Export the trained checkpoint for Unity inference with:

```bat
python python\scripts\export_onnx.py ^
  --checkpoint exports\arena_live_semantic_screen_stack4_dqn.pth ^
  --output exports\arena_live_semantic_screen_stack4_dqn.onnx
```

For the older single-frame baseline, use `configs\train_arena_semantic_screen_dqn.yaml` and `ArenaRasterEncoder > Input Mode = SemanticRgbScreen`.

To convert Unity DQN transition logs into a frame-stack expert replay dataset:

```bat
python python\data\unity_arena_raster_converter.py ^
  --input "..\Training\dataset" ^
  --config configs\arena_semantic_screen_stack4_config.json ^
  --output-dir datasets\arena_semantic_screen_stack4 ^
  --actor-side Left ^
  --output-format semantic_rgb_frame_stack_4
```
