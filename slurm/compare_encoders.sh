#!/bin/bash
#SBATCH --job-name=compare_encoders
#SBATCH --output=logs/compare_encoders_%j.out
#SBATCH --error=logs/compare_encoders_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# ============================================
# Compare contrastive vs Grover VAE encoders
# ============================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Load modules (adjust to match your Oscar environment)
module load python/3.11.0
module load cuda/12.1.1

# Activate virtualenv
source ~/scratch/ssl/venv/bin/activate

cd ~/scratch/ssl

mkdir -p logs

# Run comparison on Kuhn poker (uses existing checkpoint for contrastive)
echo "=== Running Kuhn Poker comparison ==="
python experiments/compare_encoders.py \
    --game kuhn_poker \
    --num-pretrain-agents 500 \
    --num-eval-agents 100 \
    --epochs 200 \
    --seed 42 \
    --skip-contrastive \
    --contrastive-checkpoint checkpoints/trajectory_encoder_kuhn_random_improved_500.pt

# Run comparison on Leduc poker
echo ""
echo "=== Running Leduc Poker comparison ==="
python experiments/compare_encoders.py \
    --game leduc_poker \
    --num-pretrain-agents 500 \
    --num-eval-agents 100 \
    --epochs 200 \
    --seed 42 \
    --skip-contrastive \
    --contrastive-checkpoint checkpoints/trajectory_encoder_leduc_random_improved_500.pt

echo ""
echo "End time: $(date)"
