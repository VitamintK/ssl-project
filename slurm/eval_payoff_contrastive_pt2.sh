#!/bin/bash
#SBATCH --job-name=pc_ev2
#SBATCH --output=logs/eval_payoff_contrastive_pt2_%j.out
#SBATCH --error=logs/eval_payoff_contrastive_pt2_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

eval "$('/oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" || . "/oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh"
conda activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs

GAME="liars_dice(numdice=1,dice_sides=4)"
SAFE_GAME="liars_dice_numdice1_dice_sides4"
POOL="agent_pools/${SAFE_GAME}_seed42_n500.pt"
PAYOFFS="payoff_matrices/${SAFE_GAME}__seed42_payoffs.npz"

# Finish iw=0.25 opponent adaptation (seeds 44-51)
CKPT="checkpoints/payoff_contrastive_${SAFE_GAME}_iw0.25.pt"
echo "=== Opponent Adaptation — iw=0.25 (remaining seeds) ==="
for SEED in 44 45 46 47 48 49 50 51; do
    python eval_opponent_adaptation.py \
        --game "$GAME" \
        --encoder trajectory \
        --checkpoint "$CKPT" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SEED \
        --split random \
        --device cpu
done

# Full iw=0.5 eval
CKPT="checkpoints/payoff_contrastive_${SAFE_GAME}_iw0.5.pt"

echo ""
echo "============================================"
echo "=== Payoff-Contrastive iw=0.5 ==="
echo "============================================"

for SEED in 42 43 44; do
    echo ""
    echo "=== Tasks DFR — iw=0.5 (seed=$SEED) ==="
    python experiments/compare_encoders.py \
        --game "$GAME" \
        --num-pretrain-agents 500 \
        --num-eval-agents 500 \
        --epochs 200 \
        --seed $SEED \
        --tasks DFR \
        --skip-contrastive \
        --contrastive-checkpoint "$CKPT" \
        --skip-grover \
        --grover-checkpoint "checkpoints/grover_${SAFE_GAME}.pt"
done

for SEED in 42 43 44; do
    echo ""
    echo "=== Tasks A,B — iw=0.5 (seed=$SEED) ==="
    python eval_with_payoffs.py \
        --game "$GAME" \
        --encoder trajectory \
        --checkpoint "$CKPT" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SEED \
        --device cpu
done

echo ""
echo "=== Opponent Adaptation — iw=0.5 ==="
for SEED in 42 43 44 45 46 47 48 49 50 51; do
    python eval_opponent_adaptation.py \
        --game "$GAME" \
        --encoder trajectory \
        --checkpoint "$CKPT" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SEED \
        --split random \
        --device cpu
done

echo ""
echo "All remaining evaluations complete."
echo "End time: $(date)"
