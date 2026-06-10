@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Train Arena MLP-DQN smoke
python python\scripts\train_arena_dqn.py --mode mlp --episodes 3 --batch-size 8 --min-replay-size 8 --eval-interval 1 --eval-episodes 2 --output-dir runs\smoke_arena_mlp --checkpoint-path exports\smoke_arena_mlp.pth --cpu || exit /b 1

echo [2/3] Train Arena CNN-DQN smoke
python python\scripts\train_arena_dqn.py --mode cnn --episodes 3 --batch-size 8 --min-replay-size 8 --eval-interval 1 --eval-episodes 2 --output-dir runs\smoke_arena_cnn --checkpoint-path exports\smoke_arena_cnn.pth --cpu || exit /b 1

echo [3/3] Export Arena smoke CNN ONNX
python python\scripts\export_onnx.py --checkpoint exports\smoke_arena_cnn.pth --output exports\arena_raster_smoke.onnx || exit /b 1

echo Arena raster smoke complete.
