"""
Test script for Tabular Action-Distribution Baseline on downstream tasks A and B.

For each agent, computes the full action distribution at every decision-point
information state in the game. This flat vector is used directly as the
"embedding" — no learned encoder is needed.

This baseline tests whether a simple, hand-crafted representation (the
complete tabular policy) is sufficient for payoff prediction, providing
context for whether learned embeddings capture anything beyond the raw
action distribution.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import get_all_states

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/tabular_baseline_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def collect_decision_states(game: pyspiel.Game) -> list[pyspiel.State]:
    """Get all non-terminal, non-chance states in the game."""
    all_states_dict = get_all_states.get_all_states(game)
    return [
        state
        for state in all_states_dict.values()
        if not state.is_terminal() and not state.is_chance_node()
    ]


def compute_tabular_embedding(agent: PPOAgent, game: pyspiel.Game,
                               decision_states: list[pyspiel.State]) -> np.ndarray:
    """Compute the tabular action-distribution vector for an agent.

    For each decision state, queries the agent's policy to get the action
    probabilities over all actions. Concatenates these into a single flat vector.
    """
    num_actions = game.num_distinct_actions()
    embedding_parts = []

    for state in decision_states:
        player_id = state.current_player()
        legal_actions = state.legal_actions(player_id)
        legal_action_mask = torch.zeros(num_actions)
        legal_action_mask[legal_actions] = 1
        info_state = torch.Tensor(state.information_state_tensor(player_id))

        with torch.no_grad():
            _, _, _, _, probs = agent.get_action_and_value(info_state, legal_action_mask)

        embedding_parts.append(probs.detach().cpu().numpy())

    return np.concatenate(embedding_parts)


def test_task_a(game, embeddings, downstream_agents, seed=42):
    """Task A: Predict expected payoff vs uniform random opponent."""
    set_seed(seed)  # Ensure identical train/val split across all encoders
    logger.info("\n" + "="*80)
    logger.info("Task A: Fixed opponent payoff prediction (Tabular Baseline)")
    logger.info("="*80)

    opponent_policy = policy_lib.UniformRandomPolicy(game)

    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, agent, 0, False) for agent in downstream_agents],
        p2_policies=[opponent_policy],
        p1_embeddings=embeddings,
        p2_embeddings=[np.array([0])],
        hidden_dims=[],  # linear probe
        dropout=0.2,
        device="cpu",
    )

    logger.info("Training linear probe...")
    history = predictor.train(
        num_epochs=100, batch_size=16, learning_rate=1e-4,
        validation_split=0.2, verbose=True,
    )

    val_metrics = predictor.evaluate(eval_set="val")
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(train_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)

    logger.info(f"\nTask A Results:")
    logger.info(f"  MSE:      {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline: {baseline_mse:.6f}")
    imp = (1 - val_metrics['mse'] / baseline_mse) * 100
    logger.info(f"  Improvement: {imp:.2f}%")

    return val_metrics['mse'], baseline_mse, imp


def test_task_b(game, embeddings, downstream_agents, seed=42):
    """Task B: Predict expected payoff for agent vs agent matchups."""
    set_seed(seed)  # Ensure identical train/val split across all encoders
    logger.info("\n" + "="*80)
    logger.info("Task B: Agent matchup payoff prediction (Tabular Baseline)")
    logger.info("="*80)

    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, agent, 0, False) for agent in downstream_agents],
        p2_policies=[PPOAgentPolicy(game, agent, 1, False) for agent in downstream_agents],
        p1_embeddings=embeddings,
        p2_embeddings=embeddings,
        hidden_dims=[],  # linear probe
        dropout=0.2,
        device="cpu",
    )

    logger.info("Training linear probe...")
    history = predictor.train(
        num_epochs=100, batch_size=16, learning_rate=1e-4,
        validation_split=0.2, verbose=True,
    )

    val_metrics = predictor.evaluate(eval_set="val")
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(train_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)

    logger.info(f"\nTask B Results:")
    logger.info(f"  MSE:      {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline: {baseline_mse:.6f}")
    imp = (1 - val_metrics['mse'] / baseline_mse) * 100
    logger.info(f"  Improvement: {imp:.2f}%")

    return val_metrics['mse'], baseline_mse, imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tabular action-distribution baseline on Tasks A and B")
    parser.add_argument("--game", type=str, default="kuhn_poker",
                        choices=["kuhn_poker", "leduc_poker"])
    parser.add_argument("--agent-pool", type=str, default=None,
                        help="Path to saved agent pool (from generate_agents.py)")
    parser.add_argument("--num-agents", type=int, default=500,
                        help="Number of eval agents (ignored if --agent-pool)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or get_device_string()
    logger.info(f"Device: {device}")
    logger.info(f"Game: {args.game}")
    logger.info(f"Seed: {args.seed}")

    game = pyspiel.load_game(args.game)

    if args.agent_pool:
        from generate_agents import load_agents
        logger.info(f"\nLoading agents from {args.agent_pool}...")
        pretrain_agents, eval_agents = load_agents(args.agent_pool, game)
    else:
        info_state_size = game.information_state_tensor_shape()
        num_actions = game.num_distinct_actions()
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        logger.info(f"\nCreating {args.num_agents} eval agents...")
        eval_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(args.num_agents)
        ]

    # Compute tabular embeddings
    decision_states = collect_decision_states(game)
    num_actions = game.num_distinct_actions()
    embed_dim = len(decision_states) * num_actions
    logger.info(f"\nDecision states: {len(decision_states)}")
    logger.info(f"Tabular embedding dimension: {embed_dim}")

    logger.info(f"Computing tabular embeddings for {len(eval_agents)} agents...")
    embeddings = [compute_tabular_embedding(agent, game, decision_states) for agent in eval_agents]

    # Run tasks
    results = {}
    a_mse, a_base, a_imp = test_task_a(game, embeddings, eval_agents, seed=args.seed)
    results["task_a"] = {"mse": a_mse, "baseline": a_base, "improvement": a_imp}

    b_mse, b_base, b_imp = test_task_b(game, embeddings, eval_agents, seed=args.seed)
    results["task_b"] = {"mse": b_mse, "baseline": b_base, "improvement": b_imp}

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Task A: MSE={a_mse:.6f}  Baseline={a_base:.6f}  Imp={a_imp:.2f}%")
    logger.info(f"Task B: MSE={b_mse:.6f}  Baseline={b_base:.6f}  Imp={b_imp:.2f}%")
    logger.info("="*80)
