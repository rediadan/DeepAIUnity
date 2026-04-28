cd /d E:\GameDevelopingFile\UniversalRandomPicker\Training

python train_behavior_cloning.py --dataset "data\arena_dataset.npz" --output "models\arena_bc.pt" --epochs 30 --batch-size 128 --lr 0.001 --hidden-size 64
pause