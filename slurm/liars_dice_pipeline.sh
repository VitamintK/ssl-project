#!/bin/bash
#SBATCH --job-name=ld_pipe
#SBATCH --output=logs/liars_dice_pipeline_%j.out
#SBATCH --error=logs/liars_dice_pipeline_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs agent_pools payoff_matrices

SEED=${SEED:-42}
GAME="liars_dice(numdice=1,dice_sides=4)"
SAFE_GAME="liars_dice_numdice1_dice_sides4"
POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"

echo "=== Step 1: Generate agent pool ==="
if [ ! -f "$POOL" ]; then
    python generate_agents.py --game "$GAME" --seed $SEED --num-agents 500
else
    echo "Agent pool already exists: $POOL"
fi

echo ""
echo "=== Step 2: Precompute payoff matrices ==="
PAYOFFS="payoff_matrices/${SAFE_GAME}_seed${SEED}_payoffs.npz"
if [ ! -f "$PAYOFFS" ]; then
    python precompute_payoffs.py --game "$GAME" --agent-pool "$POOL" --seed $SEED
else
    echo "Payoffs already exist: $PAYOFFS"
fi

echo ""
echo "End time: $(date)"
echo "Next steps: submit training jobs for trajectory and grover encoders"
