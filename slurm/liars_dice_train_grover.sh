#!/bin/bash
#SBATCH --job-name=ld_grov
#SBATCH --output=logs/liars_dice_train_grover_%j.out
#SBATCH --error=logs/liars_dice_train_grover_%j.err
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

echo "=== Training Grover Encoder for Liar's Dice ==="
python train_grover.py \
    --game "$GAME" \
    --agent-pool "$POOL" \
    --epochs 200 \
    --batch-size 16 \
    --lr 1e-4 \
    --seed $SEED \
    --output "checkpoints/grover_${SAFE_GAME}.pt"

echo "End time: $(date)"
