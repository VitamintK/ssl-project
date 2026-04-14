#!/bin/bash
#SBATCH --job-name=cmp_leduc
#SBATCH --output=logs/compare_encoders_leduc_%j.out
#SBATCH --error=logs/compare_encoders_leduc_%j.err
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
echo "=== Running Leduc Poker comparison (seed=$SEED) ==="
python experiments/compare_encoders.py \
    --game leduc_poker \
    --num-pretrain-agents 500 \
    --num-eval-agents 500 \
    --epochs 200 \
    --seed $SEED \
    --tasks D \
    --skip-contrastive \
    --contrastive-checkpoint checkpoints/trajectory_encoder_leduc_random_improved_500.pt

echo "End time: $(date)"
