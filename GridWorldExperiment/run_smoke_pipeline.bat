@echo off
setlocal
cd /d "%~dp0"

echo [1/6] Generate small rule-based demo
python python\scripts\generate_rule_demo.py --episodes 20 --output datasets\demonstrations\demo_raw.jsonl || exit /b 1

echo [2/6] Convert demo JSONL to NPZ
python python\data\demo_loader.py --input datasets\demonstrations\demo_raw.jsonl --output-dir datasets\demonstrations || exit /b 1

echo [3/6] Train BC smoke model
python python\scripts\train_bc.py --epochs 2 --batch-size 64 --cpu || exit /b 1

echo [4/6] Train CNN-DQN smoke model
python python\scripts\train_dqn.py --mode cnn_pure --episodes 3 --batch-size 8 --min-replay-size 8 --eval-interval 1 --eval-episodes 2 --output-dir runs\smoke_dqn_pure --checkpoint-path exports\smoke_dqn_pure.pth --cpu || exit /b 1

echo [5/6] Train BC Init + CNN-DQN smoke model
python python\scripts\train_dqn.py --mode bc_init --episodes 3 --batch-size 8 --min-replay-size 8 --eval-interval 1 --eval-episodes 2 --output-dir runs\smoke_dqn_bc_init --checkpoint-path exports\smoke_dqn_bc_init.pth --cpu || exit /b 1

echo [6/6] Export smoke ONNX
python python\scripts\export_onnx.py --checkpoint exports\smoke_dqn_bc_init.pth --output exports\smoke_dqn_model.onnx || exit /b 1

echo Smoke pipeline complete.
