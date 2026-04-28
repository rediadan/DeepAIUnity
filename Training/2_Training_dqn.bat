cd /d E:\GameDevelopingFile\UniversalRandomPicker\Training

python train_dqn.py --dataset "data\arena_dqn_dataset.npz" --output "models\dqn\arena_dqn.pt" --bc-checkpoint "models\arena_bc.pt" --epochs 40 --batch-size 128 --lr 0.0005 --hidden-size 64 --gamma 0.97

pause