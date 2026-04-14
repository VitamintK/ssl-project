#!/bin/bash
#SBATCH --job-name=ld_adapt2
#SBATCH --output=logs/liars_dice_opponent_adapt_v2_%j.out
#SBATCH --error=logs/liars_dice_opponent_adapt_v2_%j.err
#SBATCH --time=4:00:00
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
SPLIT=${SPLIT:-random}
POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"
PAYOFFS="payoff_matrices/${SAFE_GAME}__seed${SEED}_payoffs.npz"

echo "=== Opponent Adaptation v2: Trajectory Encoder (seed=$SEED, split=$SPLIT) ==="
python eval_opponent_adaptation.py \
    --game "$GAME" \
    --encoder trajectory \
    --checkpoint "checkpoints/trajectory_encoder_${SAFE_GAME}_random_improved_500.pt" \
    --agent-pool "$POOL" \
    --payoffs "$PAYOFFS" \
    --seed $SEED \
    --split $SPLIT \
    --device cpu

echo ""
echo "=== Opponent Adaptation v2: Grover Encoder (seed=$SEED, split=$SPLIT) ==="
python eval_opponent_adaptation.py \
    --game "$GAME" \
    --encoder grover \
    --checkpoint "checkpoints/grover_${SAFE_GAME}.pt" \
    --agent-pool "$POOL" \
    --payoffs "$PAYOFFS" \
    --seed $SEED \
    --split $SPLIT \
    --device cpu

echo "End time: $(date)"
