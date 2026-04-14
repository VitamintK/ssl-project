#!/bin/bash
#SBATCH --job-name=traj_ab
#SBATCH --output=logs/trajectory_tasks_ab_%j.out
#SBATCH --error=logs/trajectory_tasks_ab_%j.err
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

if [ "$GAME" = "kuhn_poker" ]; then
    CHECKPOINT=checkpoints/trajectory_encoder_kuhn_random_improved_500.pt
elif [ "$GAME" = "leduc_poker" ]; then
    CHECKPOINT=checkpoints/trajectory_encoder_leduc_random_improved_500.pt
else
    echo "Unknown game: $GAME"
    exit 1
fi

POOL=agent_pools/${GAME}_seed${SEED}_n500.pt

echo "=== Running Trajectory Tasks A+B: $GAME (seed=$SEED) ==="
python test_trajectory_encoder.py \
    --game $GAME \
    --checkpoint $CHECKPOINT \
    --agent-pool $POOL \
    --seed $SEED \
    --device cpu

echo "End time: $(date)"
