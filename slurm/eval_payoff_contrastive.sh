#!/bin/bash
#SBATCH --job-name=pc_eval
#SBATCH --output=logs/eval_payoff_contrastive_%j.out
#SBATCH --error=logs/eval_payoff_contrastive_%j.err
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

for IW in 0.0 0.25 0.5; do
    CKPT="checkpoints/payoff_contrastive_${SAFE_GAME}_iw${IW}.pt"

    echo ""
    echo "============================================"
    echo "=== Payoff-Contrastive iw=${IW} ==="
    echo "============================================"

    # --- Tasks D, F, R (agent ID, few-shot, retrieval) ---
    for SEED in 42 43 44; do
        echo ""
        echo "=== Tasks DFR — iw=${IW} (seed=$SEED) ==="
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

    # --- Tasks A, B (payoff prediction) ---
    for SEED in 42 43 44; do
        POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"
        PAYOFFS="payoff_matrices/${SAFE_GAME}__seed${SEED}_payoffs.npz"

        echo ""
        echo "=== Tasks A,B — iw=${IW} (seed=$SEED) ==="
        python eval_with_payoffs.py \
            --game "$GAME" \
            --encoder trajectory \
            --checkpoint "$CKPT" \
            --agent-pool "$POOL" \
            --payoffs "$PAYOFFS" \
            --seed $SEED \
            --device cpu
    done

    # --- Opponent Adaptation (10 seeds) ---
    POOL="agent_pools/${SAFE_GAME}_seed42_n500.pt"
    PAYOFFS="payoff_matrices/${SAFE_GAME}__seed42_payoffs.npz"

    echo ""
    echo "=== Opponent Adaptation — iw=${IW} ==="
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
done

echo ""
echo "All payoff-contrastive evaluations complete."
echo "End time: $(date)"
