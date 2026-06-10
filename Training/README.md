# Deep AI Arena Training

This folder trains top-view arena agents from player logs.
The project-level research direction is summarized in the root `README.md`.

The current experiments focus on the neural-network side of DQN: comparing DQN model architectures, testing Behavior Cloning initialization, and using v2 causal tags to explain behavior in a switch-door and moving-obstacle arena.

## Current Input Features

BC/DQN Python scripts use the legacy 21-feature sequence input by default for existing ONNX compatibility.
With `--feature-version v2`, BC/DQN use the same 38-feature arena observation shape as `ArenaMlAgent`.

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

With `--sequence-length 4`, the default v1 model input size is `21 * 4 = 84`.

Legacy ML-Agents observation size was 25:

- the 21 legacy fields above
- distance to item
- distance to base
- distance to opponent
- distance to current target

Arena v2 observation size is 38:

- the 25 fields above
- nearest door delta and open state
- nearest switch delta and active state
- nearest moving obstacle delta, velocity, and ahead flag
- shortcut blocked and detour-needed flags

With `--feature-version v2 --sequence-length 4`, the v2 BC/DQN model input size is `38 * 4 = 152`.

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

## ML-Agents Training

Unity-side ML-Agents training uses `ArenaMlAgent`, not the existing ONNX runner.

1. In Unity, set either `Left Actor Mode` or `Right Actor Mode` to `MlAgents`.
2. Keep the other actor as `RuleBased`, `DqnInference`, or `Human` depending on the experiment.
3. Start training from the project root:

```cmd
mlagents-learn Training\mlagents\arena_rl_only.yaml --run-id arena_rl_only
```

Then press Play in Unity.

For imitation-assisted training, record demonstrations with Unity's `Demonstration Recorder` on the actor that has `ArenaMlAgent`.
Save the `.demo` files under `Training\mlagents\demos`, then run:

```cmd
mlagents-learn Training\mlagents\arena_bc_rl.yaml --run-id arena_bc_rl
```

For manual demonstration recording, set the actor mode to `MlAgents`, set its `Behavior Parameters` to `Heuristic Only`, keep `Use Manual Input In Heuristic` enabled on `ArenaMlAgent`, and set `Demonstration Recorder > Num Steps To Record` to `0` or a large value. WASD/arrow keys drive movement, and `E`, `Ctrl`, or Space records shove.

ML-Agents action branches:

- branch 0: `Idle / Up / Down / Left / Right`
- branch 1: `No Shove / Shove`

The behavior name is fixed to `ArenaMlAgent`.

## Arena v2 Experiments

Arena v2 adds switch-controlled doors and moving obstacles. Keep v1 datasets and models separate from v2 outputs.

1. Collect v2 logs into `Training\dataset_v2`.
2. Run the v2 experiment batch:

```cmd
Training\run_dqn_v2_experiments.bat
```

This creates:

- `Training\data\arena_bc_v2_dataset.npz`
- `Training\data\arena_dqn_v2_dataset.npz`
- `Training\models\v2\...`
- `Training\experiments\dqn\...`
- `Training\experiments\causal\...`

BC/DQN v2 preprocessing writes 152-wide sequence rows. ML-Agents v2 training uses the same behavior name with the 38-feature live observation:

```cmd
mlagents-learn Training\mlagents\arena_rl_only_v2.yaml --run-id arena_rl_only_v2
```
