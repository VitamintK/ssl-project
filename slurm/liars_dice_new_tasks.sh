#!/bin/bash
#SBATCH --job-name=ld_tasks
#SBATCH --output=logs/liars_dice_new_tasks_%j.out
#SBATCH --error=logs/liars_dice_new_tasks_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs

GAME="liars_dice(numdice=1,dice_sides=4)"
SAFE_GAME="liars_dice_numdice1_dice_sides4"
SEED=${SEED:-42}
TASKS=${TASKS:-DR}
POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"

echo "=== Running Liar's Dice new tasks=$TASKS (seed=$SEED) ==="
python experiments/compare_encoders.py \
    --game "$GAME" \
    --agent-pool "$POOL" \
    --seed $SEED \
    --tasks $TASKS \
    --skip-contrastive \
    --contrastive-checkpoint "checkpoints/trajectory_encoder_${SAFE_GAME}_random_improved_500.pt" \
    --skip-grover \
    --grover-checkpoint "checkpoints/grover_${SAFE_GAME}.pt"

echo "End time: $(date)"
