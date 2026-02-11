"""
Analyze downstream task results from all_downstream_tasks.json.

For each key in the file, prints:
- The key (experiment configuration)
- Mean of baseline MSEs (second element in each pair)
- Mean of MSEs (first element in each pair)
"""

import json
import numpy as np
from pathlib import Path


def analyze_downstream_results(json_path: str = "results/all_downstream_tasks.json"):
    """
    Parse and analyze downstream task results.

    Args:
        json_path: Path to the JSON file containing results
    """
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("DOWNSTREAM TASK RESULTS ANALYSIS")
    print("=" * 80)
    print()

    # Analyze each experiment
    for key, pairs in data.items():
        # Extract MSEs and baseline MSEs
        mses = [pair[0] for pair in pairs]
        baseline_mses = [pair[1] for pair in pairs]

        # Calculate means
        mean_mse = np.mean(mses)
        mean_baseline_mse = np.mean(baseline_mses)

        # Calculate standard deviations for additional context
        std_mse = np.std(mses)
        std_baseline_mse = np.std(baseline_mses)

        # Calculate improvement
        improvement = ((mean_baseline_mse - mean_mse) / mean_baseline_mse) * 100

        print(f"Experiment: {key}")
        print(f"  MSE (with encoder):     {mean_mse:.6f} ± {std_mse:.6f}")
        print(f"  Baseline MSE:           {mean_baseline_mse:.6f} ± {std_baseline_mse:.6f}")
        print(f"  Improvement:            {improvement:.2f}%")
        print()

    print("=" * 80)


if __name__ == "__main__":
    analyze_downstream_results()
