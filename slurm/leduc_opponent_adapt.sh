#!/bin/bash
#SBATCH --job-name=led_adapt
#SBATCH --output=logs/leduc_opponent_adapt_%j.out
#SBATCH --error=logs/leduc_opponent_adapt_%j.err
#SBATCH --time=4:00:00
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

GAME="leduc_poker"
SAFE_GAME="leduc_poker"
POOL_SEED=42
POOL="agent_pools/${SAFE_GAME}_seed${POOL_SEED}_n500.pt"
PAYOFFS="payoff_matrices/${SAFE_GAME}_seed${POOL_SEED}_payoffs.npz"

for SPLIT_SEED in 42 43 44 45 46 47 48 49 50 51; do
    echo ""
    echo "=== Opponent Adaptation: Trajectory Encoder (seed=$SPLIT_SEED) ==="
    python eval_opponent_adaptation.py \
        --game "$GAME" \
        --encoder trajectory \
        --checkpoint "checkpoints/trajectory_encoder_leduc_random_improved_500.pt" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SPLIT_SEED \
        --split random \
        --device cpu

    echo ""
    echo "=== Opponent Adaptation: Grover Encoder (seed=$SPLIT_SEED) ==="
    python eval_opponent_adaptation.py \
        --game "$GAME" \
        --encoder grover \
        --checkpoint "checkpoints/grover_${SAFE_GAME}.pt" \
        --agent-pool "$POOL" \
        --payoffs "$PAYOFFS" \
        --seed $SPLIT_SEED \
        --split random \
        --device cpu
done

echo ""
echo "End time: $(date)"
