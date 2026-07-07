"""
Test script for Grover Encoder on downstream tasks A and B.

Loads a pre-trained Grover encoder from checkpoint and evaluates on:
- Task A: Predict policy payoff vs uniform random opponent
- Task B: Predict payoff for (P1_agent, P2_agent) matchups
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import pyspiel
from open_spiel.python import policy as policy_lib

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from policy_repr.utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from policy_repr.downstream.heads import PayoffPredictor, set_seed

from policy_repr.encoders.grover import (
    GroverEncoder,
    GroverConfig,
    GroverAdapter,
    load_grover_from_checkpoint,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/grover_encoder_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def test_task_a(game, encoder_fn, downstream_agents, device="cpu"):
    """Task A: Predict expected payoff vs uniform random opponent."""
    logger.info("\n" + "="*80)
    logger.info("Task A: Fixed opponent payoff prediction (Grover)")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    opponent_policy = policy_lib.UniformRandomPolicy(game)

    logger.info(f"Embedding {len(downstream_agents)} agents...")
    embeddings = [encoder_fn(agent).detach().cpu().numpy() for agent in downstream_agents]

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


def test_task_b(game, encoder_fn, downstream_agents, device="cpu"):
    """Task B: Predict expected payoff for agent vs agent matchups."""
    logger.info("\n" + "="*80)
    logger.info("Task B: Agent matchup payoff prediction (Grover)")
    logger.info("="*80)

    game_short_name = game.get_type().short_name

    logger.info(f"Embedding {len(downstream_agents)} agents...")
    embeddings = [encoder_fn(agent).detach().cpu().numpy() for agent in downstream_agents]

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
    parser = argparse.ArgumentParser(description="Test Grover encoder on Tasks A and B")
    parser.add_argument("--game", type=str, default="kuhn_poker",
                        choices=["kuhn_poker", "leduc_poker"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Grover checkpoint")
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
        from policy_repr.datasets.generate_agents import load_agents
        logger.info(f"\nLoading agents from {args.agent_pool}...")
        pretrain_agents, eval_agents = load_agents(args.agent_pool, game)
    else:
        info_state_size = game.information_state_tensor_shape()
        num_actions = game.num_distinct_actions()
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        logger.info(f"\nCreating {args.num_agents} pretrain agents (opponent pool)...")
        pretrain_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(args.num_agents)
        ]

        logger.info(f"Creating {args.num_agents} eval agents...")
        eval_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(args.num_agents)
        ]

    # Load Grover encoder
    encoder, adapter, config = load_grover_from_checkpoint(
        args.checkpoint, game, policies=pretrain_agents, device=device,
    )
    encoder_fn = adapter.get_encoder(device=device)

    # Run tasks
    results = {}
    a_mse, a_base, a_imp = test_task_a(game, encoder_fn, eval_agents, device)
    results["task_a"] = {"mse": a_mse, "baseline": a_base, "improvement": a_imp}

    b_mse, b_base, b_imp = test_task_b(game, encoder_fn, eval_agents, device)
    results["task_b"] = {"mse": b_mse, "baseline": b_base, "improvement": b_imp}

    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Task A: MSE={a_mse:.6f}  Baseline={a_base:.6f}  Imp={a_imp:.2f}%")
    logger.info(f"Task B: MSE={b_mse:.6f}  Baseline={b_base:.6f}  Imp={b_imp:.2f}%")
    logger.info("="*80)
