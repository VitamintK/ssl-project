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
from open_spiel.python import policy as policy_lib
from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string, PPOAgentPolicy, get_expected_payoffs_agent
from downstream import set_seed
from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint
from trajectory_encoder import TrajectoryEncoderConfig  # needed for unpickling checkpoint
from open_spiel.python.algorithms import exploitability as expl_lib
from psro import load_ppo_agents_from_psro

# Inject TrajectoryEncoderConfig into __main__ for unpickling checkpoints saved from __main__
import sys
sys.modules['__main__'].TrajectoryEncoderConfig = TrajectoryEncoderConfig


def compute_aggression(game: pyspiel.Game, agent: PPOAgent) -> float:
    """
    Compute aggression score of a PPOAgent.

    Aggression is the average probability of taking aggressive actions (bet/raise)
    across all information states where the agent can act.

    Args:
        game: OpenSpiel game instance
        agent: PPOAgent to evaluate

    Returns:
        Aggression score between 0 and 1 (higher = more aggressive)
    """
    game_name = game.get_type().short_name

    # Determine which action is aggressive
    if game_name == "kuhn_poker":
        aggressive_action = 1  # Bet
    elif game_name == "leduc_poker":
        aggressive_action = 2  # Raise
    else:
        raise ValueError(f"Unknown game: {game_name}")

    policy = PPOAgentPolicy(game, agent, player_id=0, use_observation=False)

    # Collect aggression across all info states via game tree traversal
    aggression_values = []

    def traverse(state):
        if state.is_terminal():
            return

        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                child = state.child(action)
                traverse(child)
        else:
            current_player = state.current_player()

            # Get action probs for player 0 (our agent)
            if current_player == 0:
                action_probs = policy.action_probabilities(state, current_player)
                if aggressive_action in action_probs:
                    aggression_values.append(action_probs[aggressive_action])

            # Continue traversal
            for action in state.legal_actions():
                child = state.child(action)
                traverse(child)

    traverse(game.new_initial_state())

    if not aggression_values:
        return 0.0
    return np.mean(aggression_values)


def compute_exploitability(game: pyspiel.Game, agent: PPOAgent) -> float:
    """
    Compute exploitability of a PPOAgent in a given game.

    For 2-player games, creates a joint policy where the agent plays both positions
    and computes how much it can be exploited by a best response.

    Args:
        game: OpenSpiel game instance
        agent: PPOAgent to evaluate

    Returns:
        Exploitability value (lower is better, 0 = Nash equilibrium)
    """
    # Create policies for both player positions
    policy_p0 = PPOAgentPolicy(game, agent, player_id=0, use_observation=False)
    policy_p1 = PPOAgentPolicy(game, agent, player_id=1, use_observation=False)

    # Create a joint tabular policy
    class JointPolicy:
        """Wrapper that routes to the appropriate player's policy."""
        def action_probabilities(self, state, player_id=None):
            if player_id is None:
                player_id = state.current_player()
            if player_id == 0:
                return policy_p0.action_probabilities(state, player_id)
            else:
                return policy_p1.action_probabilities(state, player_id)

    joint_policy = JointPolicy()

    # Compute exploitability
    return expl_lib.exploitability(game, joint_policy)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def visualize_embeddings_tsne(
    checkpoint_path: str,
    game: pyspiel.Game,
    num_agents: int = 500,
    agents: Optional[list[PPOAgent]] = None,
    opponent_pool: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    seed: int = 42,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    color_by: str = "index",
    agent_label: str = "Random",
):
    """
    Generate and visualize t-SNE embeddings of agents.

    Args:
        checkpoint_path: Path to the trajectory encoder checkpoint
        game: OpenSpiel game instance
        num_agents: Number of random agents to generate (ignored if agents provided)
        agents: Optional list of agents to visualize (if None, generates random agents)
        opponent_pool: Optional opponent pool for trajectory generation
        device: Device to run on
        seed: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter (typically 5-50)
        save_path: Optional path to save the plot
        color_by: What to color points by - 'index', 'exploitability', 'ev_vs_random', or 'aggression'
        agent_label: Label for the agents (e.g., 'Random', 'PSRO') used in plot title
    """
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    logger.info(f"Using device: {device}")
    logger.info(f"Color by: {color_by}")

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Load pre-trained encoder
    logger.info(f"\nLoading pre-trained encoder from {checkpoint_path}...")
    _, adapter, _ = load_trajectory_encoder_from_checkpoint(
        checkpoint_path,
        game,
        policies=opponent_pool,
        device=device
    )

    # Use provided agents or create random ones
    if agents is not None:
        logger.info(f"\nUsing {len(agents)} provided agents...")
        num_agents = len(agents)
    else:
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

    # Compute color values based on color_by parameter
    if color_by == "exploitability":
        logger.info("\nComputing exploitability for each agent...")
        color_values = []
        for i, agent in enumerate(agents):
            if (i + 1) % 50 == 0:
                logger.info(f"  Computed exploitability for {i + 1}/{num_agents} agents")
            expl = compute_exploitability(game, agent)
            color_values.append(expl)
        color_values = np.array(color_values)
        color_label = "Exploitability"
        cmap = "plasma"  # Use plasma for exploitability (yellow=high/bad, purple=low/good)
        logger.info(f"Exploitability stats: min={color_values.min():.4f}, max={color_values.max():.4f}, mean={color_values.mean():.4f}")
    elif color_by == "ev_vs_random":
        logger.info("\nComputing EV against uniform random for each agent...")
        uniform_random = policy_lib.UniformRandomPolicy(game)
        color_values = []
        for i, agent in enumerate(agents):
            if (i + 1) % 50 == 0:
                logger.info(f"  Computed EV for {i + 1}/{num_agents} agents")
            ev = get_expected_payoffs_agent(game, agent, uniform_random)
            color_values.append(ev)
        color_values = np.array(color_values)
        color_label = "EV vs Uniform Random"
        cmap = "RdYlGn"  # Red=negative, Yellow=neutral, Green=positive
        logger.info(f"EV stats: min={color_values.min():.4f}, max={color_values.max():.4f}, mean={color_values.mean():.4f}")
    elif color_by == "aggression":
        logger.info("\nComputing aggression score for each agent...")
        color_values = []
        for i, agent in enumerate(agents):
            if (i + 1) % 50 == 0:
                logger.info(f"  Computed aggression for {i + 1}/{num_agents} agents")
            agg = compute_aggression(game, agent)
            color_values.append(agg)
        color_values = np.array(color_values)
        color_label = "Aggression (Bet/Raise Freq)"
        cmap = "coolwarm"  # Blue=passive, Red=aggressive
        logger.info(f"Aggression stats: min={color_values.min():.4f}, max={color_values.max():.4f}, mean={color_values.mean():.4f}")
    else:
        color_values = np.arange(num_agents)
        color_label = "Agent Index"
        cmap = "viridis"

    # Apply t-SNE
    logger.info(f"\nApplying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
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
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=20
    )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    color_suffix = f" (colored by {color_by})" if color_by != "index" else ""
    # ax.set_title(
    #     f't-SNE Visualization of Trajectory Encoder Embeddings\n'
    #     f'{game_short_name.replace("_", " ").title()} - {num_agents} {agent_label} Agents{color_suffix}',
    #     fontsize=14
    # )
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    # Save plot
    viz_dir = Path("results") / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    color_suffix = f"_{color_by}" if color_by != "index" else ""

    agent_suffix = f"_{agent_label.lower()}" if agent_label != "Random" else ""
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        # Default save path
        default_path_png = viz_dir / f"tsne_{game_short_name}_{num_agents}_agents{agent_suffix}{color_suffix}.png"
        default_path_pdf = viz_dir / f"tsne_{game_short_name}_{num_agents}_agents{agent_suffix}{color_suffix}.pdf"
        plt.savefig(default_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(default_path_pdf, bbox_inches='tight')
        logger.info(f"Saved png plot to {default_path_png}")
        logger.info(f"Saved pdf plot to {default_path_pdf}")

    # Also save embeddings for future analysis
    embeddings_path = Path(save_path).parent / f"embeddings_{game_short_name}_{num_agents}_agents{agent_suffix}{color_suffix}.npz" if save_path else viz_dir / f"embeddings_{game_short_name}_{num_agents}_agents{agent_suffix}{color_suffix}.npz"
    save_dict = {
        'embeddings_high_dim': embeddings,
        'embeddings_2d': embeddings_2d,
        'checkpoint_path': checkpoint_path,
        'num_agents': num_agents,
        'seed': seed,
        'perplexity': perplexity,
        'color_by': color_by,
        'color_values': color_values,
    }
    np.savez(embeddings_path, **save_dict)
    logger.info(f"Saved embeddings to {embeddings_path}")

    return embeddings, embeddings_2d, fig, color_values


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
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
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
        ax.set_xlabel('t-SNE Dim 1', fontsize=14)
        ax.set_ylabel('t-SNE Dim 2', fontsize=14)
        ax.set_title(label, fontsize=16)
        ax.tick_params(axis='both', labelsize=11)
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
                        help='Number of random agents to visualize (ignored if --agent-source=psro)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save the plot')
    parser.add_argument('--color-by', type=str, default='index',
                        choices=['index', 'exploitability', 'ev_vs_random', 'aggression'],
                        help='What to color points by (default: index)')
    # PSRO agent options
    parser.add_argument('--agent-source', type=str, default='random',
                        choices=['random', 'psro'],
                        help='Source of agents to visualize (default: random)')
    parser.add_argument('--psro-hidden-size', type=int, default=256,
                        help='Hidden size for PSRO agents (default: 256)')
    parser.add_argument('--psro-player-id', type=int, default=None,
                        help='Player ID for PSRO agents (default: None = both players)')

    args = parser.parse_args()

    # Setup
    device = args.device if args.device else get_device_string()
    game = pyspiel.load_game(args.game)

    logger.info("="*80)
    logger.info("TRAJECTORY ENCODER EMBEDDING VISUALIZATION")
    logger.info("="*80)
    logger.info(f"Game: {args.game}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Agent source: {args.agent_source}")
    if args.agent_source == 'random':
        logger.info(f"Number of agents: {args.num_agents}")
    else:
        logger.info(f"PSRO hidden size: {args.psro_hidden_size}")
        logger.info(f"PSRO player ID: {args.psro_player_id}")
    logger.info(f"Perplexity: {args.perplexity}")
    logger.info(f"Device: {device}")
    logger.info(f"Color by: {args.color_by}")

    # Load agents based on source
    if args.agent_source == 'psro':
        logger.info(f"\nLoading PSRO agents...")
        agents = load_ppo_agents_from_psro(
            game_short_name=args.game,
            player_id=args.psro_player_id,
            hidden_size=args.psro_hidden_size,
            shuffle=True,
        )
        agent_label = "PSRO"
        # Also use PSRO agents as opponent pool for trajectory generation
        opponent_pool = agents
    else:
        agents = None  # Will generate random agents
        agent_label = "Random"
        opponent_pool = None  # Use uniform random opponent

    # Visualize
    embeddings, embeddings_2d, fig, color_values = visualize_embeddings_tsne(
        checkpoint_path=args.checkpoint,
        game=game,
        num_agents=args.num_agents,
        agents=agents,
        opponent_pool=opponent_pool,
        device=device,
        seed=args.seed,
        perplexity=args.perplexity,
        save_path=args.save_path,
        color_by=args.color_by,
        agent_label=agent_label,
    )

    logger.info("\n" + "="*80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("="*80)
