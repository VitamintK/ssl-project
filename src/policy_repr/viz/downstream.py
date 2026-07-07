"""
Visualization utilities for downstream task predictions.
Shows predicted vs actual payoffs for test sets.
"""

from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    metrics: Dict[str, float],
    title: str = "Predicted vs Actual Payoffs",
    save_path: Optional[str] = None,
    show_plot: bool = True,
) -> plt.Figure:
    """
    Create scatter and residual plots comparing predicted and actual payoffs.

    Args:
        predictions: Array of predicted payoff values
        ground_truth: Array of actual payoff values
        metrics: Dictionary containing 'mse', 'mae', and 'r2' metrics
        title: Plot title
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot: Predicted vs Actual
    ax1.scatter(
        ground_truth,
        predictions,
        alpha=0.6,
        s=50,
        edgecolors='k',
        linewidth=0.5
    )

    # Perfect prediction line (diagonal)
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    ax1.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label='Perfect Prediction',
        alpha=0.7
    )

    ax1.set_xlabel('Actual Payoff', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Payoff', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = (
        f"MSE: {metrics['mse']:.4f}\n"
        f"MAE: {metrics['mae']:.4f}\n"
        f"R²: {metrics['r2']:.4f}"
    )
    ax1.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
    )

    # Residual plot
    residuals = predictions - ground_truth
    ax2.scatter(
        ground_truth,
        residuals,
        alpha=0.6,
        s=50,
        edgecolors='k',
        linewidth=0.5
    )
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7)

    ax2.set_xlabel('Actual Payoff', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add residual statistics
    residual_text = (
        f"Mean: {residuals.mean():.4f}\n"
        f"Std: {residuals.std():.4f}"
    )
    ax2.text(
        0.05,
        0.95,
        residual_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment='top',
        bbox={'boxstyle': 'round', 'facecolor': 'lightblue', 'alpha': 0.5}
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig
