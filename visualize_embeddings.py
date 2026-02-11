"""
Visualize embeddings using t-SNE dimensionality reduction.

This script provides flexible visualization of embeddings from any source:
- Pre-computed embeddings (numpy arrays)
- Custom encoder functions with arbitrary items
- Legacy trajectory encoder mode for backward compatibility

The embeddings are visualized in 2D using t-SNE for exploration and analysis.
"""

import logging
from pathlib import Path
from typing import Literal, Optional
import argparse

import numpy as np
import pyspiel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from open_spiel.python import policy as policy_lib
from open_spiel.python.policy import Policy

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from main import get_policies_and_embeddings
from psro import load_ppo_agents_from_neupl, make_neupl_policies
from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string, PPOAgentPolicy
from downstream import set_seed
from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def calculate_aggressiveness(
    policy: Policy,
    game: pyspiel.Game,
    player: int,
    num_episodes: int = 100,
) -> float:
    """
    Calculate how aggressive a policy is by measuring its tendency to take
    aggressive actions against a uniform random opponent.

    For Kuhn Poker: action 0 = passive (pass/call), action 1 = aggressive (bet/raise)
    For Leduc Poker: action 2 = aggressive (raise), others = passive (fold/call)

    Args:
        policy: The OpenSpiel Policy to evaluate
        game: OpenSpiel game instance
        player: Which player position to evaluate (0 or 1)
        num_episodes: Number of episodes to run for evaluation

    Returns:
        Aggressiveness score between 0 and 1 (proportion of aggressive actions)
    """
    game_short_name = game.get_type().short_name

    # Determine aggressive action(s) based on game
    if game_short_name == "kuhn_poker":
        aggressive_actions = {1}  # Bet/raise
    elif game_short_name == "leduc_poker":
        aggressive_actions = {2}  # Raise
    else:
        raise ValueError(f"Aggressiveness not defined for game: {game_short_name}")

    # Create uniform random opponent
    opponent_policy = policy_lib.UniformRandomPolicy(game)

    # Track aggressive actions
    aggressive_count = 0
    total_actions = 0

    for _ in range(num_episodes):
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.is_chance_node():
                # Chance node: sample random outcome
                outcomes, probs = zip(*state.chance_outcomes())
                action = np.random.choice(outcomes, p=probs)
                state.apply_action(action)
            else:
                # Player node
                current_player = state.current_player()

                # Get policy for current player
                if current_player == player:
                    current_policy = policy
                else:
                    current_policy = opponent_policy

                # Get action probabilities from policy
                action_probs = current_policy.action_probabilities(state)
                actions = list(action_probs.keys())
                probs = np.array(list(action_probs.values())) # if we don't convert to array, random choice complains that probs don't sum to 1?
                action = np.random.choice(actions, p=probs)

                # Track aggressive actions for the specified player
                if current_player == player:
                    total_actions += 1
                    if action in aggressive_actions:
                        aggressive_count += 1

                state.apply_action(action)

    # Return proportion of aggressive actions
    return aggressive_count / total_actions if total_actions > 0 else 0.0


def visualize_embeddings_tsne(
    embeddings: Optional[np.ndarray] = None,
    policies: Optional[list[Policy]] = None,
    checkpoint_path: Optional[str] = None,
    game: Optional[pyspiel.Game] = None,
    num_agents: int = 500,
    opponent_pool: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    seed: int = 42,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    title_suffix: str = "",
    filename_suffix: str = "",
    aggressiveness_episodes: int = 100,
):
    """
    Visualize embeddings using t-SNE dimensionality reduction.

    This function supports two modes of operation:
    1. Direct embeddings: Pass pre-computed embeddings via `embeddings` parameter
       (optionally with `policies` for aggressiveness coloring)
    2. Trajectory encoder: Pass `checkpoint_path` and `game` for legacy trajectory encoder mode

    When policies and a game are provided, points are colored by aggressiveness
    (proportion of aggressive actions vs uniform random opponent).

    Args:
        embeddings: Pre-computed embeddings array of shape (n_samples, embedding_dim)
        policies: Optional list of Policy objects corresponding to embeddings (for aggressiveness)
        checkpoint_path: Path to encoder checkpoint (legacy trajectory encoder mode)
        game: OpenSpiel game instance (used with checkpoint_path or for aggressiveness)
        num_agents: Number of random agents to generate (only for trajectory encoder mode)
        opponent_pool: Optional opponent pool (only for trajectory encoder mode)
        device: Device to run on
        seed: Random seed for reproducibility
        perplexity: t-SNE perplexity parameter (typically 5-50)
        save_path: Optional path to save the plot
        title_suffix: Additional text to append to the plot title
        filename_suffix: Additional text to append to the filename
        aggressiveness_episodes: Number of episodes for aggressiveness calculation
    """
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")
    logger.info(f"Using device: {device}")

    # Determine the mode and get embeddings
    policies_list = None  # Track policies for aggressiveness calculation

    if embeddings is not None:
        # Mode 1: Direct embeddings provided
        embeddings = np.array(embeddings)
        logger.info(f"Using provided embeddings of shape {embeddings.shape}")
        num_samples = len(embeddings)
        game_short_name = "custom"
        policies_list = policies
    elif checkpoint_path is not None and game is not None:
        # Mode 3: Legacy trajectory encoder mode
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
        encoder_fn_local = adapter.get_encoder(device=device)
        embeddings = []

        for i, agent in enumerate(agents):
            if (i + 1) % 100 == 0:
                logger.info(f"  Encoded {i + 1}/{num_agents} agents")
            embedding = encoder_fn_local(agent).detach().cpu().numpy()
            embeddings.append(embedding)

        embeddings = np.array(embeddings)
        num_samples = num_agents
        policies_list = agents
        logger.info(f"Embeddings shape: {embeddings.shape}")
    else:
        raise ValueError(
            "Must provide either:\n"
            "  1. 'embeddings' (pre-computed), or\n"
            "  2. 'checkpoint_path' and 'game' (legacy trajectory encoder mode)"
        )

    # Calculate aggressiveness for coloring if we have policies and a game
    color_values = None
    color_label = "Sample Index"

    if policies_list is not None and game is not None:
        # Check if policies_list contains PPOAgent instances
        if policies_list and isinstance(policies_list[0], PPOAgent):
            logger.info(f"\nCalculating aggressiveness for {len(policies_list)} agents...")
            aggressiveness_scores = []
            for i, agent in enumerate(policies_list):
                if (i + 1) % 100 == 0:
                    logger.info(f"  Calculated aggressiveness for {i + 1}/{len(policies_list)} agents")
                # Wrap PPOAgent as OpenSpiel Policy
                agent_policy = PPOAgentPolicy(game, agent, player_id=0, use_observation=False)
                score = calculate_aggressiveness(agent_policy, game, player=0, num_episodes=aggressiveness_episodes)
                aggressiveness_scores.append(score)

            color_values = np.array(aggressiveness_scores)
            color_label = "Aggressiveness"
            logger.info(f"Aggressiveness range: [{color_values.min():.3f}, {color_values.max():.3f}]")
        elif policies_list:
            # Policies are already Policy objects
            logger.info(f"\nCalculating aggressiveness for {len(policies_list)} policies...")
            aggressiveness_scores = []
            for i, policy in enumerate(policies_list):
                if (i + 1) % 100 == 0:
                    logger.info(f"  Calculated aggressiveness for {i + 1}/{len(policies_list)} policies")
                score = calculate_aggressiveness(policy, game, player=0, num_episodes=aggressiveness_episodes)
                aggressiveness_scores.append(score)

            color_values = np.array(aggressiveness_scores)
            color_label = "Aggressiveness"
            logger.info(f"Aggressiveness range: [{color_values.min():.3f}, {color_values.max():.3f}]")

    # Default to sample index if no aggressiveness calculated
    if color_values is None:
        color_values = np.arange(num_samples)
        color_label = "Sample Index"

    # Apply t-SNE
    logger.info(f"\nApplying t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=2000,
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
        # cmap='viridis',
        cmap='coolwarm',
        alpha=0.6,
        s=20
    )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)

    # Build title based on mode
    if title_suffix:
        title = f't-SNE Visualization of Embeddings\n{title_suffix}'
    else:
        title = (
            f't-SNE Visualization of Embeddings\n'
            f'{game_short_name.replace("_", " ").title()} - {num_samples} Samples'
        )
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=11)

    plt.tight_layout()

    # Save plot
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        # Default save path
        viz_dir = Path("results") / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tsne_{game_short_name}_{num_samples}_samples{filename_suffix}.png"
        default_path = viz_dir / filename
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {default_path}")

    # Also save embeddings for future analysis
    embeddings_filename = f"embeddings_{game_short_name}_{num_samples}_samples{filename_suffix}.npz"
    embeddings_path = Path(save_path).parent / embeddings_filename if save_path else viz_dir / embeddings_filename

    save_dict = {
        'embeddings_high_dim': embeddings,
        'embeddings_2d': embeddings_2d,
        'checkpoint_path': checkpoint_path if checkpoint_path else "N/A",
        'num_samples': num_samples,
        'seed': seed,
        'perplexity': perplexity,
        'color_values': color_values,
        'color_label': color_label,
    }

    # Add aggressiveness scores if calculated
    if color_label == "Aggressiveness":
        save_dict['aggressiveness_scores'] = color_values

    np.savez(embeddings_path, **save_dict)
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
        _, adapter, _ = load_trajectory_encoder_from_checkpoint(
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

def visualize_wrapper(
        encoder_type: Literal["neupl", "identity", "weight_autoencoder", "functional_autoencoder"],
        game_name: str = "kuhn_poker",
):
    PERPLEXITY = 40
    game = pyspiel.load_game(game_name)
    if encoder_type == "neupl":
        SAMPLING_MODE = "gaussian"
        neupl_config = {
            'hidden_size': 256,
            'policy_embedding_size': 64,
            'use_randall_loss': False,
        }
        policies_and_embeddings = make_neupl_policies(
            game_short_name=game_name,
            neupl_config=neupl_config,
            original_num_policies=23,
            num_policies_to_make=1000,
            interpolate_prenorm=True,
            sampling_mode=SAMPLING_MODE,
        )
        p0_policies_and_embeddings = policies_and_embeddings[0]
        policies = [pe[1] for pe in p0_policies_and_embeddings]
        embeddings = [pe[0].detach().cpu().numpy() for pe in p0_policies_and_embeddings]
        # perplexity = 5
    elif encoder_type == "identity":
        p0_policies_and_embeddings = [
            (np.zeros(64), PPOAgentPolicy(game, PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE), player_id=0, use_observation=False))
            for _ in range(1000)
        ]
    elif encoder_type == "weight_autoencoder":
        PLAYER_ID = 0
        N = 2000 # half for training the encoder, half to be used for visualization
        device = get_device_string()
        game_short_name = game.get_type().short_name
        info_state_size = game.information_state_tensor_shape()
        num_actions = game.num_distinct_actions()
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)
        ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, 256) for _ in range(N)]
        policies, embeddings = get_policies_and_embeddings(game, PLAYER_ID, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
        # perplexity = 30
        # _, identity_embeddings = get_policies_and_embeddings2(player_id, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
    elif encoder_type == "functional_autoencoder":
        N = 2000 # half for training the encoder, half to be used for visualization
        device = get_device_string()
        game_short_name = game.get_type().short_name
        info_state_size = game.information_state_tensor_shape()
        num_actions = game.num_distinct_actions()
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)
        ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, 256) for _ in range(N)]
        policies, embeddings = get_policies_and_embeddings(game, PLAYER_ID, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")
    visualize_embeddings_tsne(
        embeddings=embeddings,
        policies=policies,
        game=game,
        num_agents=1,
        opponent_pool=None,
        device="cpu",
        seed=42,
        perplexity=PERPLEXITY,
        save_path=f"results/visualizations/tsne_{game_name}_{encoder_type}_P{PERPLEXITY}.png",
        title_suffix=encoder_type,
        filename_suffix=f"_{encoder_type}",
        aggressiveness_episodes=100,
    )

if __name__ == "__main__":
    # visualize_wrapper('weight_autoencoder', game_name='kuhn_poker')
    visualize_wrapper('neupl', game_name='leduc_poker')
    exit()
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
