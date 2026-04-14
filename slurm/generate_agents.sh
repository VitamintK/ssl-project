#!/bin/bash
#SBATCH --job-name=gen_agents
#SBATCH --output=logs/generate_agents_%j.out
#SBATCH --error=logs/generate_agents_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs agent_pools

for GAME in kuhn_poker leduc_poker; do
    for SEED in 42 43 44; do
        echo "=== Generating $GAME seed=$SEED ==="
        python generate_agents.py --game $GAME --seed $SEED --num-agents 500
    done
done

echo "End time: $(date)"
