#!/bin/bash
# Submit comparison jobs for all 3 seeds
# Usage: bash slurm/compare_all_seeds.sh [kuhn|leduc|both]

GAMES=${1:-kuhn}

for SEED in 42 43 44; do
    if [ "$GAMES" = "kuhn" ] || [ "$GAMES" = "both" ]; then
        echo "Submitting Kuhn seed=$SEED"
        sbatch --export=ALL,SEED=$SEED slurm/compare_encoders_kuhn.sh
    fi
    if [ "$GAMES" = "leduc" ] || [ "$GAMES" = "both" ]; then
        echo "Submitting Leduc seed=$SEED"
        sbatch --export=ALL,SEED=$SEED slurm/compare_encoders_leduc.sh
    fi
done
