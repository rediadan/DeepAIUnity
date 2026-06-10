@echo off
setlocal
cd /d "%~dp0"

echo [1/9] Generate rule-based demonstrations
python python\scripts\generate_rule_demo.py --episodes 500 --output datasets\demonstrations\demo_raw.jsonl || exit /b 1

echo [2/9] Convert demonstrations
python python\data\demo_loader.py --input datasets\demonstrations\demo_raw.jsonl --output-dir datasets\demonstrations || exit /b 1

echo [3/9] Train BC
python python\scripts\train_bc.py --epochs 30 --batch-size 128 || exit /b 1

echo [4/9] Train MLP-DQN baseline
python python\scripts\train_dqn.py --mode mlp_pure --episodes 600 --output-dir runs\dqn_mlp --checkpoint-path exports\dqn_mlp.pth || exit /b 1

echo [5/9] Train Pure CNN-DQN
python python\scripts\train_dqn.py --mode cnn_pure --episodes 600 --output-dir runs\dqn_pure --checkpoint-path exports\dqn_pure.pth || exit /b 1

echo [6/9] Train BC Init + CNN-DQN
python python\scripts\train_dqn.py --mode bc_init --episodes 600 --output-dir runs\dqn_bc_init --checkpoint-path exports\dqn_bc_init.pth || exit /b 1

echo [7/9] Evaluate available agents
python python\scripts\evaluate.py --episodes 50 || exit /b 1

echo [8/9] Analyze causal tags and plot results
python python\scripts\analyze_causal_tags.py --input datasets\demonstrations\demo_train.npz --output-dir runs\causal || exit /b 1
python python\scripts\plot_results.py --threshold 3.0 || exit /b 1

echo [9/9] Export final ONNX
python python\scripts\export_onnx.py --checkpoint exports\dqn_bc_init.pth --output exports\dqn_model.onnx || exit /b 1

echo Full experiment pipeline complete.
