"""
Visualize trajectory encoder embeddings using t-SNE.

This script loads a pre-trained trajectory encoder and visualizes the learned
embedding space by encoding random agents and plotting them in 2D using t-SNE.
"""

import logging
from pathlib import Path
from typing import Optional
import argparse

import numpy as np
import torch
import pyspiel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string
from downstream import set_seed
from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def visualize_embeddings_tsne(
    checkpoint_path: str,
    game: pyspiel.Game,
    num_agents: int = 500,
    opponent_pool: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    seed: int = 42,
    perplexity: int = 30,
    save_path: Optional[str] = None,
):
    """
    Generate and visualize t-SNE embeddings of random agents.

    Args:
        checkpoint_path: Path to the trajectory encoder checkpoint
        game: OpenSpiel game instance
        num_agents: Number of random agents to generate and visualize
        opponent_pool: Optional opponent pool for trajectory generation
        device: Device to run on
        seed: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter (typically 5-50)
        save_path: Optional path to save the plot
    """
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    logger.info(f"Using device: {device}")

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Load pre-trained encoder
    logger.info(f"\nLoading pre-trained encoder from {checkpoint_path}...")
    model, adapter, config = load_trajectory_encoder_from_checkpoint(
        checkpoint_path,
        game,
        policies=opponent_pool,
        device=device
    )

    # Create random agents
    logger.info(f"\nGenerating {num_agents} random agents...")
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(num_agents)
    ]

    # Encode agents
    logger.info("Encoding agents to embedding space...")
    encoder_fn = adapter.get_encoder(device=device)
    embeddings = []

    for i, agent in enumerate(agents):
        if (i + 1) % 100 == 0:
            logger.info(f"  Encoded {i + 1}/{num_agents} agents")
        embedding = encoder_fn(agent).detach().cpu().numpy()
        embeddings.append(embedding)

    embeddings = np.array(embeddings)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Apply t-SNE
    logger.info(f"\nApplying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        n_iter=1000,
        verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create visualization
    logger.info("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with a color gradient
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=range(num_agents),
        cmap='viridis',
        alpha=0.6,
        s=20
    )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(
        f't-SNE Visualization of Trajectory Encoder Embeddings\n'
        f'{game_short_name.replace("_", " ").title()} - {num_agents} Random Agents',
        fontsize=14
    )
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Agent Index', fontsize=11)

    plt.tight_layout()

    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        # Default save path
        viz_dir = Path("results") / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        default_path = viz_dir / f"tsne_{game_short_name}_{num_agents}_agents.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {default_path}")

    # Also save embeddings for future analysis
    embeddings_path = Path(save_path).parent / f"embeddings_{game_short_name}_{num_agents}_agents.npz" if save_path else viz_dir / f"embeddings_{game_short_name}_{num_agents}_agents.npz"
    np.savez(
        embeddings_path,
        embeddings_high_dim=embeddings,
        embeddings_2d=embeddings_2d,
        checkpoint_path=checkpoint_path,
        num_agents=num_agents,
        seed=seed,
        perplexity=perplexity
    )
    logger.info(f"Saved embeddings to {embeddings_path}")

    return embeddings, embeddings_2d, fig


def visualize_multiple_checkpoints(
    checkpoint_paths: list[str],
    checkpoint_labels: list[str],
    game: pyspiel.Game,
    num_agents: int = 200,
    device: str = "cpu",
    seed: int = 42,
    save_path: Optional[str] = None,
):
    """
    Compare embeddings from multiple checkpoints side-by-side.

    Args:
        checkpoint_paths: List of paths to trajectory encoder checkpoints
        checkpoint_labels: Labels for each checkpoint
        game: OpenSpiel game instance
        num_agents: Number of random agents to generate
        device: Device to run on
        seed: Random seed
        save_path: Optional path to save the plot
    """
    set_seed(seed)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Create shared random agents
    logger.info(f"\nGenerating {num_agents} random agents (shared across all checkpoints)...")
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(num_agents)
    ]

    # Create subplots
    n_checkpoints = len(checkpoint_paths)
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(6 * n_checkpoints, 5))
    if n_checkpoints == 1:
        axes = [axes]

    for idx, (checkpoint_path, label) in enumerate(zip(checkpoint_paths, checkpoint_labels)):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing checkpoint {idx+1}/{n_checkpoints}: {label}")
        logger.info(f"{'='*80}")

        # Load encoder
        model, adapter, config = load_trajectory_encoder_from_checkpoint(
            checkpoint_path,
            game,
            device=device
        )

        # Encode agents
        logger.info("Encoding agents...")
        encoder_fn = adapter.get_encoder(device=device)
        embeddings = np.array([
            encoder_fn(agent).detach().cpu().numpy()
            for agent in agents
        ])

        # Apply t-SNE
        logger.info("Applying t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        ax = axes[idx]
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(num_agents),
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        ax.set_xlabel('t-SNE Dim 1')
        ax.set_ylabel('t-SNE Dim 2')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f't-SNE Comparison: {game_short_name.replace("_", " ").title()}',
        fontsize=16,
        y=1.02
    )
    plt.tight_layout()

    # Save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"\nSaved comparison plot to {save_path}")
    else:
        viz_dir = Path("results") / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        default_path = viz_dir / f"tsne_comparison_{game_short_name}.png"
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        logger.info(f"\nSaved comparison plot to {default_path}")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize trajectory encoder embeddings')
    parser.add_argument('--game', type=str, default='kuhn_poker',
                        choices=['kuhn_poker', 'leduc_poker'],
                        help='Game to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trajectory encoder checkpoint')
    parser.add_argument('--num-agents', type=int, default=500,
                        help='Number of random agents to visualize')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the plot')

    args = parser.parse_args()

    # Setup
    device = args.device if args.device else get_device_string()
    game = pyspiel.load_game(args.game)

    logger.info("="*80)
    logger.info("TRAJECTORY ENCODER EMBEDDING VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Game: {args.game}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Number of agents: {args.num_agents}")
    logger.info(f"Perplexity: {args.perplexity}")
    logger.info(f"Device: {device}")

    # Visualize
    embeddings, embeddings_2d, fig = visualize_embeddings_tsne(
        checkpoint_path=args.checkpoint,
        game=game,
        num_agents=args.num_agents,
        opponent_pool=None,  # Use uniform random opponent
        device=device,
        seed=args.seed,
        perplexity=args.perplexity,
        save_path=args.save_path,
    )

    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*80)
