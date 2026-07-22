"""
Visualize policy encoder embeddings using t-SNE or PCA dimensionality reduction.

Supports all 5 encoder types described in CLAUDE.md: neupl, identity,
weight_autoencoder, functional_autoencoder, trajectory_encoder. Run as a
CLI; see `parse_args` for all options, or `python visualize_embeddings.py --help`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pyspiel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from open_spiel.python import policy as policy_lib
from open_spiel.python.policy import Policy
from open_spiel.python.algorithms import policy_utils, best_response
from open_spiel.python.algorithms.psro_v2 import utils as psro_utils
from pyspiel import TabularBestResponse

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from downstream import set_seed
from psro import make_neupl_policies, load_ppo_agents_from_psro
from functional_autoencoder import FunctionalEncoderAdapter
from test_weight_encoder import load_weight_encoder_from_checkpoint
from test_functional_encoder import load_functional_encoder_from_checkpoint
from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint
from trajectory_encoder import TrajectoryEncoderConfig  # needed for unpickling checkpoint
from utils import (
    make_diverse_random_kuhn_poker_layer_init,
    get_device_string,
    PPOAgentPolicy,
    get_expected_payoffs,
)
from weight_autoencoder import ppo_agent_to_vector

# Checkpoints saved from a __main__ script need this to unpickle.
sys.modules['__main__'].TrajectoryEncoderConfig = TrajectoryEncoderConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


PPO_AGENT_HIDDEN_SIZE = 256

ENCODER_TYPES = (
    "neupl",
    "identity",
    "weight_autoencoder",
    "functional_autoencoder",
    "trajectory_encoder",
)

# Which action index is "aggressive" (bet/raise) per game, for compute_aggression.
AGGRESSIVE_ACTION = {
    "kuhn_poker": 1,   # Bet
    "leduc_poker": 2,  # Raise
}


def _make_random_agents(
    game: pyspiel.Game, num_agents: int, hidden_size: int = PPO_AGENT_HIDDEN_SIZE
) -> list[PPOAgent]:
    """Generate `num_agents` randomly initialized PPOAgents for `game`."""
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    return [
        PPOAgent(num_actions, info_state_size, "cpu", layer_init, hidden_size)
        for _ in range(num_agents)
    ]


def compute_aggression(game: pyspiel.Game, policy: Policy) -> float:
    """
    Compute the exact aggression score of a policy via full game-tree traversal.

    Aggression is the average probability of taking the game's aggressive
    action (bet/raise) across all information states where `policy`'s
    player (assumed to be player 0) can act.
    """
    game_name = game.get_type().short_name
    if game_name not in AGGRESSIVE_ACTION:
        raise ValueError(f"Aggression not defined for game: {game_name}")
    aggressive_action = AGGRESSIVE_ACTION[game_name]

    aggression_values: list[float] = []

    def traverse(state: pyspiel.State) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                traverse(state.child(action))
            return

        current_player = state.current_player()
        if current_player == 0:
            action_probs = policy.action_probabilities(state, current_player)
            if aggressive_action in action_probs:
                aggression_values.append(action_probs[aggressive_action])

        for action in state.legal_actions():
            traverse(state.child(action))

    traverse(game.new_initial_state())
    return float(np.mean(aggression_values)) if aggression_values else 0.0


def compute_exploitabilities(
    game: pyspiel.Game, policies: list[Policy], player_id: int = 0
) -> np.ndarray:
    """
    Best-response exploitability of each policy playing seat `player_id`:
    the payoff a best-responding opponent achieves in the other seat (higher
    = more exploitable).

    This is the same definition as Task D's ExploitabilityPredictor
    (downstream.py) and operates on any OpenSpiel Policy -- including neupl
    PPONeuplAgentPolicy objects -- so no raw PPOAgent is required. The tabular
    best-response processor is built once and reused across all policies.
    """
    best_responder_id = 1 - player_id
    initial_state = game.new_initial_state()
    all_states, state_to_info = psro_utils.compute_states_and_info_states_if_none(
        game, None, None
    )
    processor = TabularBestResponse(
        game,
        best_responder_id,
        policy_utils.policy_to_dict(
            policy_lib.UniformRandomPolicy(game), game, all_states, state_to_info
        ),
    )

    values = []
    for policy in tqdm(policies, desc="Computing exploitability"):
        processor.set_policy(
            policy_utils.policy_to_dict(policy, game, all_states, state_to_info)
        )
        best_responder = best_response.CPPBestResponsePolicy(
            game, best_responder_id, policy, all_states, state_to_info, processor
        )
        values.append(best_responder.value(initial_state))
    return np.array(values)


def compute_color_values(
    color_by: Literal["index", "aggression", "exploitability", "ev_vs_random"],
    game: Optional[pyspiel.Game],
    policies: Optional[list[Policy]],
    num_samples: int,
) -> tuple[np.ndarray, str, str]:
    """
    Compute the per-sample scalar used to color the embedding scatter plot.

    Returns (color_values, color_label, colormap_name). This is the single
    place coloring is computed, and every metric operates on `policies`, so
    it works uniformly across all encoder types (including neupl).
    """
    if color_by == "aggression":
        if policies is None or game is None:
            raise ValueError("color_by='aggression' requires `policies` and `game`.")
        logger.info(f"\nComputing aggression for {len(policies)} policies...")
        values = np.array([
            compute_aggression(game, p)
            for p in tqdm(policies, desc="Computing aggression")
        ])
        logger.info(
            f"Aggression stats: min={values.min():.4f}, max={values.max():.4f}, "
            f"mean={values.mean():.4f}"
        )
        return values, "Aggression (Bet/Raise Freq)", "coolwarm"

    if color_by == "exploitability":
        if policies is None or game is None:
            raise ValueError("color_by='exploitability' requires `policies` and `game`.")
        logger.info(f"\nComputing best-response exploitability for {len(policies)} policies...")
        values = compute_exploitabilities(game, policies)
        logger.info(
            f"Exploitability stats: min={values.min():.4f}, max={values.max():.4f}, "
            f"mean={values.mean():.4f}"
        )
        return values, "Exploitability (BR value)", "plasma"

    if color_by == "ev_vs_random":
        if policies is None or game is None:
            raise ValueError("color_by='ev_vs_random' requires `policies` and `game`.")
        logger.info(f"\nComputing EV vs uniform random for {len(policies)} policies...")
        uniform_random = policy_lib.UniformRandomPolicy(game)
        values = np.array([get_expected_payoffs(game, p, uniform_random) for p in policies])
        logger.info(
            f"EV stats: min={values.min():.4f}, max={values.max():.4f}, "
            f"mean={values.mean():.4f}"
        )
        return values, "EV vs Uniform Random", "RdYlGn"

    return np.arange(num_samples), "Sample Index", "viridis"


def visualize_embeddings(
    embeddings: np.ndarray,
    policies: list[Policy],
    game: pyspiel.Game,
    game_short_name: Optional[str] = None,
    seed: int = 42,
    perplexity: int = 30,
    save_path: Optional[str] = None,
    filename_suffix: str = "",
    title_suffix: str = "",
    color_by: Literal["index", "aggression", "exploitability", "ev_vs_random"] = "index",
    agent_label: str = "Random",
    reduction_method: Literal["tsne", "pca"] = "tsne",
):
    """
    Reduce `embeddings` to 2D with t-SNE or PCA, color by `color_by`, plot, and save.

    Args:
        embeddings: Embeddings array of shape (n_samples, embedding_dim).
        policies: Policy objects parallel to `embeddings`, used for coloring.
        game: OpenSpiel game instance the policies act in.
        game_short_name: Overrides the filename/log label (defaults to `game`'s name).
        seed: Random seed for reproducibility.
        perplexity: t-SNE perplexity (ignored when reduction_method='pca').
        save_path: Optional explicit path to save the plot; otherwise a default
            path under results/visualizations/ is derived from the other args.
        filename_suffix: Extra text appended to auto-derived filenames.
        title_suffix: Extra text appended to the plot title in parentheses.
        color_by: What to color points by -- 'index', 'exploitability',
            'ev_vs_random', or 'aggression'.
        agent_label: Label for the agents (e.g. 'Random', 'PSRO') in the plot title.
        reduction_method: 'tsne' or 'pca'.

    Returns:
        (embeddings, embeddings_2d, fig, color_values)
    """
    set_seed(seed)
    embeddings = np.asarray(embeddings)
    num_samples = len(embeddings)
    game_short_name = game_short_name or game.get_type().short_name

    logger.info(f"Set random seed to {seed}")
    logger.info(f"Reduction method: {reduction_method}, color by: {color_by}")
    logger.info(f"Using {num_samples} embeddings of shape {embeddings.shape}")

    color_values, color_label, cmap = compute_color_values(
        color_by, game, policies, num_samples
    )

    if reduction_method == "pca":
        logger.info("\nApplying PCA...")
        reducer = PCA(n_components=2, random_state=seed)
        embeddings_2d = reducer.fit_transform(embeddings)
        explained = reducer.explained_variance_ratio_
        logger.info(
            f"PCA explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f} "
            f"(total={sum(explained):.3f})"
        )
        dim_labels = (f"PC1 ({explained[0]*100:.1f}%)", f"PC2 ({explained[1]*100:.1f}%)")
    else:
        logger.info(f"\nApplying t-SNE with perplexity={perplexity}...")
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=seed,
            max_iter=2000,
            learning_rate=200,
            verbose=1,
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        dim_labels = ("", "")

    logger.info("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=color_values,
        cmap=cmap,
        alpha=0.6,
        s=20,
    )
    ax.set_xlabel(dim_labels[0])
    ax.set_ylabel(dim_labels[1])
    if not dim_labels[0]:
        ax.tick_params(axis="both", labelbottom=False, labelleft=False)
    ax.grid(True, alpha=0.3)

    title = f"{agent_label} agents -- {game_short_name}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()

    viz_dir = Path("results") / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    color_suffix = f"_{color_by}" if color_by != "index" else ""
    agent_suffix = f"_{agent_label.lower()}" if agent_label != "Random" else ""

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    else:
        filename = (
            f"{reduction_method}_{game_short_name}_{num_samples}_samples"
            f"{agent_suffix}{color_suffix}{filename_suffix}"
        )
        plt.savefig(viz_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
        plt.savefig(viz_dir / f"{filename}.pdf", bbox_inches="tight")
        logger.info(f"Saved plots to {viz_dir / filename}.[png|pdf]")

    embeddings_filename = f"embeddings_{game_short_name}_{num_samples}_samples{filename_suffix}.npz"
    embeddings_path = (
        Path(save_path).parent / embeddings_filename if save_path else viz_dir / embeddings_filename
    )
    np.savez(
        embeddings_path,
        embeddings_high_dim=embeddings,
        embeddings_2d=embeddings_2d,
        num_samples=num_samples,
        seed=seed,
        perplexity=perplexity,
        reduction_method=reduction_method,
        color_by=color_by,
        color_values=color_values,
        color_label=color_label,
    )
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
    Compare embeddings from multiple trajectory-encoder checkpoints side-by-side.

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

    logger.info(f"\nGenerating {num_agents} random agents (shared across all checkpoints)...")
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    agents = [
        PPOAgent(num_actions, info_state_size, "cpu", layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(num_agents)
    ]

    n_checkpoints = len(checkpoint_paths)
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(6 * n_checkpoints, 5))
    if n_checkpoints == 1:
        axes = [axes]

    for idx, (checkpoint_path, label) in enumerate(zip(checkpoint_paths, checkpoint_labels)):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing checkpoint {idx+1}/{n_checkpoints}: {label}")
        logger.info(f"{'='*80}")

        _, adapter, _ = load_trajectory_encoder_from_checkpoint(checkpoint_path, game, device=device)

        logger.info("Encoding agents...")
        encoder_fn = adapter.get_encoder(device=device)
        embeddings = np.array([
            encoder_fn(agent).detach().cpu().numpy()
            for agent in agents
        ])

        logger.info("Applying t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)

        ax = axes[idx]
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=range(num_agents),
            cmap="viridis",
            alpha=0.6,
            s=20,
        )
        ax.set_xlabel("t-SNE Dim 1", fontsize=14)
        ax.set_ylabel("t-SNE Dim 2", fontsize=14)
        ax.set_title(label, fontsize=16)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f't-SNE Comparison: {game_short_name.replace("_", " ").title()}',
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"\nSaved comparison plot to {save_path}")
    else:
        viz_dir = Path("results") / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        default_path = viz_dir / f"tsne_comparison_{game_short_name}.png"
        plt.savefig(default_path, dpi=300, bbox_inches="tight")
        logger.info(f"\nSaved comparison plot to {default_path}")

    return fig


def _build_neupl_inputs(
    game_name: str, num_policies: int, device: str, neupl_directory: Optional[str] = None
) -> tuple[list[Policy], Optional[list[PPOAgent]], np.ndarray]:
    neupl_config = {
        "hidden_size": 256,
        "policy_embedding_size": 64,
        "use_randall_loss": True,
    }
    policies_and_embeddings = make_neupl_policies(
        game_short_name=game_name,
        neupl_config=neupl_config,
        num_policies_to_make=num_policies,
        directory=neupl_directory,
        interpolate_prenorm=True,
        sampling_mode="gaussian",
        device=device,
    )
    p0_policies_and_embeddings = policies_and_embeddings[0]
    policies = [pe[1] for pe in p0_policies_and_embeddings]
    embeddings = np.array([pe[0].detach().cpu().numpy() for pe in p0_policies_and_embeddings])
    return policies, None, embeddings


def _build_identity_inputs(
    game: pyspiel.Game, num_agents: int, agents: Optional[list[PPOAgent]]
) -> tuple[list[Policy], list[PPOAgent], np.ndarray]:
    agents = agents if agents is not None else _make_random_agents(game, num_agents)
    embeddings = np.array([ppo_agent_to_vector(a).detach().cpu().numpy() for a in agents])
    policies = [PPOAgentPolicy(game, a, player_id=0, use_observation=False) for a in agents]
    return policies, agents, embeddings


def _build_autoencoder_inputs(
    game: pyspiel.Game,
    checkpoint_path: str,
    encoder_type: Literal["weight_autoencoder", "functional_autoencoder"],
    device: str,
    num_agents: int,
    agents: Optional[list[PPOAgent]],
) -> tuple[list[Policy], list[PPOAgent], np.ndarray]:
    if encoder_type == "weight_autoencoder":
        model, _ = load_weight_encoder_from_checkpoint(checkpoint_path, device=device)
    else:
        model, _, _ = load_functional_encoder_from_checkpoint(checkpoint_path, device=device)
    encoder_fn = FunctionalEncoderAdapter(model, ppo_agent_to_vector).get_encoder(device=device)

    agents = agents if agents is not None else _make_random_agents(game, num_agents)
    embeddings = np.array([encoder_fn(a).detach().cpu().numpy() for a in agents])
    policies = [PPOAgentPolicy(game, a, player_id=0, use_observation=False) for a in agents]
    return policies, agents, embeddings


def _build_trajectory_encoder_inputs(
    game: pyspiel.Game,
    checkpoint_path: str,
    device: str,
    num_agents: int,
    agents: Optional[list[PPOAgent]],
    opponent_pool: Optional[list[PPOAgent]],
) -> tuple[list[Policy], list[PPOAgent], np.ndarray]:
    _, adapter, _ = load_trajectory_encoder_from_checkpoint(
        checkpoint_path, game, policies=opponent_pool, device=device
    )
    encoder_fn = adapter.get_encoder(device=device)

    agents = agents if agents is not None else _make_random_agents(game, num_agents)
    embeddings = np.array([encoder_fn(a).detach().cpu().numpy() for a in agents])
    policies = [PPOAgentPolicy(game, a, player_id=0, use_observation=False) for a in agents]
    return policies, agents, embeddings


def build_policies_and_embeddings(
    encoder_type: Literal[
        "neupl", "identity", "weight_autoencoder", "functional_autoencoder", "trajectory_encoder"
    ],
    game: pyspiel.Game,
    num_agents: int,
    device: str,
    checkpoint_path: Optional[str] = None,
    agents: Optional[list[PPOAgent]] = None,
    opponent_pool: Optional[list[PPOAgent]] = None,
    neupl_directory: Optional[str] = None,
) -> tuple[list[Policy], Optional[list[PPOAgent]], np.ndarray]:
    """
    Build (policies, agents, embeddings) for `encoder_type`.

    `agents` in the return value is None for encoder types (currently just
    neupl) whose policies aren't backed by a standalone PPOAgent -- this
    disables color_by="exploitability" for those types.

    `neupl_directory` selects which NEUPL training run to load from
    (`results/test/neupl/ppo/hs{hidden_size}/{game}/<neupl_directory>`). If
    omitted, NEUPL falls back to an interactive prompt (see
    `psro.select_neupl_directory`) -- pass this explicitly for any
    non-interactive/scripted run.
    """
    if encoder_type == "neupl":
        return _build_neupl_inputs(game.get_type().short_name, num_agents, device, neupl_directory)
    if encoder_type == "identity":
        return _build_identity_inputs(game, num_agents, agents)
    if encoder_type in ("weight_autoencoder", "functional_autoencoder"):
        if checkpoint_path is None:
            raise ValueError(f"encoder_type='{encoder_type}' requires --checkpoint.")
        return _build_autoencoder_inputs(game, checkpoint_path, encoder_type, device, num_agents, agents)
    if encoder_type == "trajectory_encoder":
        if checkpoint_path is None:
            raise ValueError("encoder_type='trajectory_encoder' requires --checkpoint.")
        return _build_trajectory_encoder_inputs(
            game, checkpoint_path, device, num_agents, agents, opponent_pool
        )
    raise ValueError(f"Unknown encoder type: {encoder_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize policy encoder embeddings")
    parser.add_argument(
        "--encoder-type", type=str, default="trajectory_encoder", choices=list(ENCODER_TYPES)
    )
    parser.add_argument(
        "--game", type=str, default="kuhn_poker", choices=["kuhn_poker", "leduc_poker"]
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to encoder checkpoint (required for weight_autoencoder, "
        "functional_autoencoder, trajectory_encoder)",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=500,
        help="Number of agents/policies to visualize (ignored if --agent-source=psro)",
    )
    parser.add_argument(
        "--perplexity", type=int, default=30, help="t-SNE perplexity (ignored for pca)"
    )
    parser.add_argument(
        "--reduction-method", type=str, default="tsne", choices=["tsne", "pca"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--color-by",
        type=str,
        default="index",
        choices=["index", "exploitability", "ev_vs_random", "aggression"],
    )
    parser.add_argument(
        "--agent-source",
        type=str,
        default="random",
        choices=["random", "psro"],
        help="Only applies to identity/weight_autoencoder/functional_autoencoder/trajectory_encoder",
    )
    parser.add_argument("--psro-hidden-size", type=int, default=256)
    parser.add_argument("--psro-player-id", type=int, default=None)
    parser.add_argument(
        "--neupl-directory",
        type=str,
        default=None,
        help="NEUPL run directory name under results/test/neupl/ppo/hs{size}/{game}/. "
        "Only applies to --encoder-type=neupl. If omitted, falls back to an "
        "interactive prompt -- pass this explicitly for non-interactive runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device if args.device else get_device_string()
    game = pyspiel.load_game(args.game)

    logger.info("=" * 80)
    logger.info("POLICY EMBEDDING VISUALIZATION")
    logger.info("=" * 80)
    logger.info(f"Encoder type: {args.encoder_type}")
    logger.info(f"Game: {args.game}")
    logger.info(f"Agent source: {args.agent_source}")
    logger.info(f"Reduction method: {args.reduction_method}")
    logger.info(f"Device: {device}")
    logger.info(f"Color by: {args.color_by}")

    agents = None
    opponent_pool = None
    agent_label = "Random"
    if args.encoder_type != "neupl" and args.agent_source == "psro":
        agents = load_ppo_agents_from_psro(
            game_short_name=args.game,
            player_id=args.psro_player_id,
            hidden_size=args.psro_hidden_size,
            shuffle=True,
        )
        agent_label = "PSRO"
        opponent_pool = agents

    policies, _, embeddings = build_policies_and_embeddings(
        encoder_type=args.encoder_type,
        game=game,
        num_agents=args.num_agents,
        device=device,
        checkpoint_path=args.checkpoint,
        agents=agents,
        neupl_directory=args.neupl_directory,
        opponent_pool=opponent_pool,
    )

    visualize_embeddings(
        embeddings=embeddings,
        policies=policies,
        game=game,
        seed=args.seed,
        perplexity=args.perplexity,
        save_path=args.save_path,
        filename_suffix=f"_{args.encoder_type}",
        title_suffix=args.encoder_type,
        color_by=args.color_by,
        agent_label=agent_label,
        reduction_method=args.reduction_method,
    )

    logger.info("\n" + "=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
