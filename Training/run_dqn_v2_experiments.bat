@echo off
setlocal
cd /d "%~dp0"

python preprocess_logs.py --input "dataset_v2" --output "data\arena_bc_v2_dataset.npz" --sequence-length 4 --feature-version v2
python preprocess_dqn_transitions.py --input "dataset_v2" --output "data\arena_dqn_v2_dataset.npz" --sequence-length 4 --feature-version v2

python train_behavior_cloning.py --dataset "data\arena_bc_v2_dataset.npz" --output "models\v2\arena_bc_v2.pt" --epochs 30 --batch-size 128 --lr 0.001 --hidden-size 64

python train_dqn.py --dataset "data\arena_dqn_v2_dataset.npz" --output "models\v2\mlp_scratch\arena_dqn.pt" --model-type mlp --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97 --metrics-output "experiments\dqn\mlp_scratch\metrics.csv" --summary-output "experiments\dqn\mlp_scratch\summary.json"
python train_dqn.py --dataset "data\arena_dqn_v2_dataset.npz" --output "models\v2\mlp_bc_initialized\arena_dqn.pt" --model-type mlp --bc-checkpoint "models\v2\arena_bc_v2.pt" --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97 --metrics-output "experiments\dqn\mlp_bc_initialized\metrics.csv" --summary-output "experiments\dqn\mlp_bc_initialized\summary.json"
python train_dqn.py --dataset "data\arena_dqn_v2_dataset.npz" --output "models\v2\dueling\arena_dqn.pt" --model-type dueling --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97 --metrics-output "experiments\dqn\dueling\metrics.csv" --summary-output "experiments\dqn\dueling\summary.json"
python train_dqn.py --dataset "data\arena_dqn_v2_dataset.npz" --output "models\v2\conv1d\arena_dqn.pt" --model-type conv1d --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97 --metrics-output "experiments\dqn\conv1d\metrics.csv" --summary-output "experiments\dqn\conv1d\summary.json"

python analyze_causal_logs.py --input "dataset_v2" --output-dir "experiments\causal"
