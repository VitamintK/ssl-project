#!/bin/bash
#SBATCH --job-name=ld_ablat
#SBATCH --output=logs/liars_dice_ablation_traj_%j.out
#SBATCH --error=logs/liars_dice_ablation_traj_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

source activate ~/scratch/ssl/venv
cd ~/scratch/ssl
mkdir -p logs checkpoints

GAME="liars_dice(numdice=1,dice_sides=4)"
SAFE_GAME="liars_dice_numdice1_dice_sides4"
SEED=${SEED:-42}
TRAJ_PER_POLICY=${TRAJ_PER_POLICY:-50}
POOL="agent_pools/${SAFE_GAME}_seed${SEED}_n500.pt"
PAYOFFS="payoff_matrices/${SAFE_GAME}__seed${SEED}_payoffs.npz"
CKPT="checkpoints/trajectory_encoder_${SAFE_GAME}_tpp${TRAJ_PER_POLICY}.pt"

echo "=== Ablation: Training Trajectory Encoder (trajectories_per_policy=$TRAJ_PER_POLICY) ==="

# Step 1: Train
python trajectory_encoder.py \
    --game "$GAME" \
    --agent-pool "$POOL" \
    --num-agents 500 \
    --batch-size 16 \
    --epochs 200 \
    --lr 1e-4 \
    --val-split 0.2 \
    --lr-scheduler cosine \
    --normalize \
    --trajectories-per-policy $TRAJ_PER_POLICY \
    --seed $SEED \
    --output "$CKPT"

echo "=== Ablation: Evaluating (trajectories_per_policy=$TRAJ_PER_POLICY) ==="

# Step 2: Eval with precomputed payoffs
python eval_with_payoffs.py \
    --game "$GAME" \
    --encoder trajectory \
    --checkpoint "$CKPT" \
    --agent-pool "$POOL" \
    --payoffs "$PAYOFFS" \
    --seed $SEED \
    --device cpu

echo "End time: $(date)"
