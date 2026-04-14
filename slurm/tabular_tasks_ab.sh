#!/bin/bash
#SBATCH --job-name=tab_ab
#SBATCH --output=logs/tabular_tasks_ab_%j.out
#SBATCH --error=logs/tabular_tasks_ab_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs

SEED=${SEED:-42}
GAME=${GAME:-kuhn_poker}

POOL=agent_pools/${GAME}_seed${SEED}_n500.pt

echo "=== Running Tabular Baseline Tasks A+B: $GAME (seed=$SEED) ==="
python test_tabular_baseline.py \
    --game $GAME \
    --agent-pool $POOL \
    --seed $SEED \
    --device cpu

echo "End time: $(date)"
