#!/bin/bash
#SBATCH --job-name=ld_eval
#SBATCH --output=logs/liars_dice_eval_%j.out
#SBATCH --error=logs/liars_dice_eval_%j.err
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
ENCODER=${ENCODER:-tabular}
POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"
PAYOFFS="payoff_matrices/${SAFE_GAME}__seed${SEED}_payoffs.npz"

# Set checkpoint based on encoder type
if [ "$ENCODER" = "trajectory" ]; then
    CKPT="checkpoints/trajectory_encoder_${SAFE_GAME}_random_improved_500.pt"
    CKPT_FLAG="--checkpoint $CKPT"
elif [ "$ENCODER" = "grover" ]; then
    CKPT="checkpoints/grover_${SAFE_GAME}.pt"
    CKPT_FLAG="--checkpoint $CKPT"
else
    CKPT_FLAG=""
fi

echo "=== Evaluating $ENCODER on Liar's Dice (seed=$SEED) ==="
python eval_with_payoffs.py \
    --game "$GAME" \
    --encoder $ENCODER \
    $CKPT_FLAG \
    --agent-pool "$POOL" \
    --payoffs "$PAYOFFS" \
    --seed $SEED \
    --device cpu

echo "End time: $(date)"
