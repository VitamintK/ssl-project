#!/bin/bash
#SBATCH --job-name=new_leduc
#SBATCH --output=logs/new_tasks_leduc_%j.out
#SBATCH --error=logs/new_tasks_leduc_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

module load anaconda/2023.09-0
module load cuda/12.1.1

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs

SEED=${SEED:-42}
TASKS=${TASKS:-FR}
echo "=== Running Leduc Poker new tasks=$TASKS (seed=$SEED) ==="
python experiments/compare_encoders.py \
    --game leduc_poker \
    --num-pretrain-agents 500 \
    --num-eval-agents 500 \
    --epochs 200 \
    --seed $SEED \
    --tasks $TASKS \
    --skip-contrastive \
    --contrastive-checkpoint checkpoints/trajectory_encoder_leduc_random_improved_500.pt \
    --skip-grover \
    --grover-checkpoint checkpoints/grover_leduc_poker.pt

echo "End time: $(date)"
