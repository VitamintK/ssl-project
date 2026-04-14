#!/bin/bash
#SBATCH --job-name=ld_traj
#SBATCH --output=logs/liars_dice_train_trajectory_%j.out
#SBATCH --error=logs/liars_dice_train_trajectory_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs checkpoints

GAME="liars_dice(numdice=1,dice_sides=4)"
SAFE_GAME="liars_dice_numdice1_dice_sides4"
SEED=${SEED:-42}
POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"

echo "=== Training Trajectory Encoder for Liar's Dice ==="
python trajectory_encoder.py \
    --game "$GAME" \
    --agent-pool "$POOL" \
    --num-agents 500 \
    --batch-size 16 \
    --epochs 200 \
    --lr 1e-4 \
    --val-split 0.2 \
    --lr-scheduler cosine \
    --normalize \
    --seed $SEED \
    --output "checkpoints/trajectory_encoder_${SAFE_GAME}_random_improved_500.pt"

echo "End time: $(date)"
