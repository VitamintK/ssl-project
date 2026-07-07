"""
Test script for Identity Encoder (baseline) on downstream tasks A, B, and C.

This script uses a trivial identity encoder that returns a constant embedding
for all agents, serving as a baseline to compare against trajectory and weight encoders.
The three downstream prediction tasks are:
- Task A: Predict policy payoff vs uniform random
- Task B: Predict payoff for (P1_agent, P2_agent) matchups
- Task C: Predict payoff for (P1_agent, P2_agent, state) triples
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import pyspiel
from open_spiel.python import policy as policy_lib

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from policy_repr.utils import get_device_string, make_diverse_random_kuhn_poker_layer_init
from policy_repr.downstream.heads import PayoffPredictor, StatePayoffPredictor, set_seed, sample_random_states
from policy_repr.encoders.weight_autoencoder import ppo_agent_to_vector


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/identity_encoder_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def test_downstream_task_a_with_identity_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
):
    """
    Test Task A with identity encoder.

    Task A: Predict expected payoff for P1_agent against uniform random opponent.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task A: Payoff prediction vs uniform random with identity encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)

    # Use identity encoder (raw weight vector)
    logger.info("\nUsing identity encoder (raw weight vector)...")
    encoder_fn = ppo_agent_to_vector

    # Set up downstream task
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    NUM_AGENTS_DOWNSTREAM = len(downstream_task_ppo_agents) if downstream_task_ppo_agents is not None else 100
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_DOWNSTREAM)
        ]

    # Dummy encoder for fixed P2 (uniform random)
    p2_encoder_fn = lambda x: np.array([0])

    predictor = PayoffPredictor(
        game=game,
        p1_agents=downstream_task_ppo_agents,
        p2_agents=[opponent_policy],
        p1_encoder_fn=encoder_fn,
        p2_encoder_fn=p2_encoder_fn,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )

    logger.info("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    logger.info("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    logger.info(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    logger.info(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    logger.info(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    logger.info(f"MSE: {baseline_mse:.6f}")
    logger.info(f"MAE: {baseline_mae:.6f}")
    logger.info(f"\nModel improvement over baseline:")
    logger.info(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    logger.info(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    # logger.info("\nEvaluating model on training set...")
    # train_metrics = predictor.evaluate(eval_set="train")
    # logger.info(f"\nTraining Set Results:")
    # logger.info(f"MSE: {train_metrics['mse']:.6f}")
    # logger.info(f"MAE: {train_metrics['mae']:.6f}")

    # Save results
    results_dir = Path("results") / "task_a_identity"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_identity_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


def test_downstream_task_b_with_identity_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
):
    """
    Test Task B with identity encoder.

    Task B: Predict expected payoff for (P1_agent, P2_agent) matchups.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task B: Agent matchup payoff prediction with identity encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Use identity encoder (same for both P1 and P2)
    logger.info("\nUsing identity encoder (raw weight vector)...")
    encoder_fn = ppo_agent_to_vector

    # Set up downstream task
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    NUM_AGENTS_DOWNSTREAM = len(downstream_task_ppo_agents) if downstream_task_ppo_agents is not None else 50
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_DOWNSTREAM)
        ]

    predictor = PayoffPredictor(
        game=game,
        p1_agents=downstream_task_ppo_agents,
        p2_agents=downstream_task_ppo_agents,
        p1_encoder_fn=encoder_fn,
        p2_encoder_fn=encoder_fn,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )

    logger.info("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    logger.info("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    logger.info(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    logger.info(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    logger.info(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    logger.info(f"MSE: {baseline_mse:.6f}")
    logger.info(f"MAE: {baseline_mae:.6f}")
    logger.info(f"\nModel improvement over baseline:")
    logger.info(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    logger.info(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    # logger.info("\nEvaluating model on training set...")
    # train_metrics = predictor.evaluate(eval_set="train")
    # logger.info(f"\nTraining Set Results:")
    # logger.info(f"MSE: {train_metrics['mse']:.6f}")
    # logger.info(f"MAE: {train_metrics['mae']:.6f}")

    # Save results
    results_dir = Path("results") / "task_b_identity"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_identity_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


def test_downstream_task_c_with_identity_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
):
    """
    Test Task C with identity encoder.

    Task C: Predict expected payoff for (P1_agent, P2_agent, state) triples.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task C: Full state-conditioned matchup prediction with identity encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Use identity encoder (same for both P1 and P2)
    logger.info("\nUsing identity encoder (raw weight vector)...")
    encoder_fn = ppo_agent_to_vector

    # Set up downstream task
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    NUM_AGENTS_DOWNSTREAM = len(downstream_task_ppo_agents) if downstream_task_ppo_agents is not None else 50
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_DOWNSTREAM)
        ]

    # Create state sampler function
    NUM_STATES = 100
    def state_sampler(game, num_states):
        return sample_random_states(game, num_states, max_depth=10)

    predictor = StatePayoffPredictor(
        game=game,
        p1_agents=downstream_task_ppo_agents,
        p2_agents=downstream_task_ppo_agents,
        p1_encoder_fn=encoder_fn,
        p2_encoder_fn=encoder_fn,
        state_sampler=state_sampler,
        num_states_per_pair=NUM_STATES,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )

    logger.info("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    logger.info("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    logger.info(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    logger.info(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    logger.info(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    logger.info(f"MSE: {baseline_mse:.6f}")
    logger.info(f"MAE: {baseline_mae:.6f}")
    logger.info(f"\nModel improvement over baseline:")
    logger.info(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    logger.info(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    # logger.info("\nEvaluating model on training set...")
    # train_metrics = predictor.evaluate(eval_set="train")
    # logger.info(f"\nTraining Set Results:")
    # logger.info(f"MSE: {train_metrics['mse']:.6f}")
    # logger.info(f"MAE: {train_metrics['mae']:.6f}")

    # Save results
    results_dir = Path("results") / "task_c_identity"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_identity_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    device = get_device_string()
    logger.info(f"Using device: {device}")

    game_name = "leduc_poker"
    game = pyspiel.load_game(game_name)

    # Create random agents for downstream tasks
    logger.info("\nCreating random agents for downstream tasks...")
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    # Create 50 random agents for downstream tasks
    downstream_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(50)
    ]

    logger.info(f"Created {len(downstream_agents)} agents for downstream tasks")

    # Test Task A
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK A")
    logger.info("="*80)
    test_downstream_task_a_with_identity_encoder(
        game,
        predictor_type="linear",
        downstream_task_ppo_agents=downstream_agents,
        device=device,
    )

    # Test Task B
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK B")
    logger.info("="*80)
    test_downstream_task_b_with_identity_encoder(
        game,
        predictor_type="linear",
        downstream_task_ppo_agents=downstream_agents,
        device=device,
    )

    # Test Task C
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK C")
    logger.info("="*80)
    test_downstream_task_c_with_identity_encoder(
        game,
        predictor_type="linear",
        downstream_task_ppo_agents=downstream_agents,
        device=device,
    )

    logger.info("\n" + "="*80)
    logger.info("ALL TESTS COMPLETE")
    logger.info("="*80)
