#!/bin/bash
#SBATCH --job-name=pttt_eval
#SBATCH --output=logs/pttt_eval_tasks_%j.out
#SBATCH --error=logs/pttt_eval_tasks_%j.err
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

GAME="phantom_ttt(obstype=reveal-nothing)"
TRAJ_CKPT="checkpoints/trajectory_encoder_phantom_ttt_random_improved_500.pt"
GROV_CKPT="checkpoints/grover_phantom_ttt.pt"

for SEED in 42 43 44; do
    POOL="agent_pools/phantom_ttt_seed${SEED}_n500.pt"
    PAYOFFS="payoff_matrices/phantom_ttt_seed${SEED}_payoffs.npz"

    echo ""
    echo "============================================"
    echo "=== Phantom TTT Eval — Seed $SEED ==="
    echo "============================================"

    # --- Tasks D, F, R (agent ID, few-shot, retrieval) ---
    echo ""
    echo "=== Tasks DFR (seed=$SEED) ==="
    python experiments/compare_encoders.py \
        --game "$GAME" \
        --num-pretrain-agents 500 \
        --num-eval-agents 500 \
        --epochs 200 \
        --seed $SEED \
        --tasks DFR \
        --skip-contrastive \
        --contrastive-checkpoint "$TRAJ_CKPT" \
        --skip-grover \
        --grover-checkpoint "$GROV_CKPT"

    # --- Tasks A, B (payoff prediction) — Trajectory ---
    echo ""
    echo "=== Tasks A,B — Trajectory (seed=$SEED) ==="
    python eval_with_payoffs.py \
        --game "$GAME" \
        --encoder trajectory \
        --checkpoint "$TRAJ_CKPT" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SEED \
        --device cpu

    # --- Tasks A, B — Grover ---
    echo ""
    echo "=== Tasks A,B — Grover (seed=$SEED) ==="
    python eval_with_payoffs.py \
        --game "$GAME" \
        --encoder grover \
        --checkpoint "$GROV_CKPT" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SEED \
        --device cpu

done

echo ""
echo "All Phantom TTT evaluations complete."
echo "End time: $(date)"
