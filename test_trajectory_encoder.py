"""
Test script for Trajectory Encoder on downstream tasks A, B, and C.

This script trains a trajectory encoder using contrastive learning on a set
of PPO agents, then evaluates it on the three downstream prediction tasks:
- Task A: Predict policy payoff vs uniform random, conditioned on state
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
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, StatePayoffPredictor, set_seed, sample_random_states
from trajectory_encoder import (
    TrajectoryEncoderConfig,
    TrajectoryEncoderTrainer,
    TrajectoryEncoderAdapter,
)
from psro import load_ppo_agents_from_psro, load_ppo_agents_from_single_psro_folder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/trajectory_encoder_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def load_trajectory_encoder_from_checkpoint(checkpoint_path: str, game: pyspiel.Game, policies: Optional[list[PPOAgent]] = None, device: str = "cpu"):
    """Load a pre-trained trajectory encoder from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        game: OpenSpiel game instance
        policies: Optional list of policies to use as opponent pool (should match training distribution)
        device: Device to load model on

    Returns:
        Tuple of (model, adapter, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    config.device = device  # Update device in case it's different

    # Create model and load state dict
    from trajectory_encoder import TrajectoryEncoder
    state_dim = game.information_state_tensor_shape()[0]
    action_dim = game.num_distinct_actions()

    # Use default max_seq_len=512 to match checkpoint (not num_transitions)
    model = TrajectoryEncoder(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_seq_len=512,  # Must match checkpoint's positional encoding size
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create adapter and set normalization stats
    # NOTE: Keep normalization stats on CPU because normalize_transitions operates on CPU
    # The trajectory is moved to device only after normalization
    adapter = TrajectoryEncoderAdapter(model, game, config, policies=policies)
    state_min = checkpoint['state_min'].cpu() if isinstance(checkpoint['state_min'], torch.Tensor) else checkpoint['state_min']
    state_max = checkpoint['state_max'].cpu() if isinstance(checkpoint['state_max'], torch.Tensor) else checkpoint['state_max']
    adapter.set_normalization_stats(state_min, state_max)

    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Training history: Best epoch {checkpoint['history']['best_epoch']}, "
                f"Best val loss: {checkpoint['history'].get('best_val_loss', 'N/A')}")
    if policies is not None:
        logger.info(f"Using {len(policies)} policies as opponent pool for consistent trajectory generation")
    else:
        logger.warning("No policies provided - will use uniform random opponent (may cause distribution shift!)")

    return model, adapter, config


def test_downstream_task_a_with_trajectory_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    pretrained_encoder: Optional[tuple] = None,  # (model, adapter, config) tuple
):
    """
    Test Task A with trajectory encoder.

    Task A: Predict expected payoff for (P1_agent, state) against uniform random opponent.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task A: State-conditioned payoff prediction with trajectory encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)

    # Use pre-trained encoder if provided, otherwise train from scratch
    if pretrained_encoder is not None:
        logger.info("\nUsing pre-trained trajectory encoder...")
        model, adapter, traj_config = pretrained_encoder
        encoder_fn = adapter.get_encoder(device=device)
    else:
        # Train trajectory encoder from scratch
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        NUM_AGENTS_AUTOENCODE = len(autoencoder_ppo_agents) if autoencoder_ppo_agents is not None else 500
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
                for _ in range(NUM_AGENTS_AUTOENCODE)
            ]

        logger.info(f"\nTraining trajectory encoder on {len(autoencoder_ppo_agents)} policies...")
        traj_config = TrajectoryEncoderConfig(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            num_transitions=200,
            batch_size=8,
            epochs=50,
            lr=1e-4,
            weight_decay=1e-5,
            temperature=0.07,
            normalize=True,
            device=device,
            seed=0,
        )

        trainer = TrajectoryEncoderTrainer(traj_config, game, autoencoder_ppo_agents)
        model, history = trainer.train()

        logger.info(f"Trajectory encoder trained. Final loss: {history[-1]:.4f}")

        # Create adapter and get encoder function
        adapter = TrajectoryEncoderAdapter(model, game, traj_config, policies=autoencoder_ppo_agents)
        adapter.set_normalization_stats(trainer.train_dataset.state_min, trainer.train_dataset.state_max)
        encoder_fn = adapter.get_encoder(device=device)

    # Set up downstream task
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    NUM_AGENTS_DOWNSTREAM = len(downstream_task_ppo_agents) if downstream_task_ppo_agents is not None else 500
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_DOWNSTREAM)
        ]

    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, agent, 0, False) for agent in downstream_task_ppo_agents],
        p2_policies=[opponent_policy],
        p1_embeddings=[encoder_fn(agent).detach().cpu().numpy() for agent in downstream_task_ppo_agents],
        p2_embeddings=[np.array([0])],
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

    # Baseline: predict mean of training set for everything
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(train_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    logger.info(f"\nBaseline (constant mean prediction from training set: {mean_payoff:.6f}):")
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
    results_dir = Path("results") / "task_a_trajectory"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_trajectory_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


def test_downstream_task_b_with_trajectory_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    pretrained_encoder: Optional[tuple] = None,  # (model, adapter, config) tuple
):
    """
    Test Task B with trajectory encoder.

    Task B: Predict expected payoff for (P1_agent, P2_agent) matchups.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task B: Agent matchup payoff prediction with trajectory encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Use pre-trained encoder if provided, otherwise train from scratch
    if pretrained_encoder is not None:
        logger.info("\nUsing pre-trained trajectory encoder...")
        model, adapter, traj_config = pretrained_encoder
        encoder_fn = adapter.get_encoder(device=device)
    else:
        # Train trajectory encoder from scratch (shared for both P1 and P2)
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        NUM_AGENTS_AUTOENCODE = len(autoencoder_ppo_agents) if autoencoder_ppo_agents is not None else 500
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
                for _ in range(NUM_AGENTS_AUTOENCODE)
            ]

        logger.info(f"\nTraining trajectory encoder on {len(autoencoder_ppo_agents)} policies...")
        traj_config = TrajectoryEncoderConfig(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            num_transitions=200,
            batch_size=8,
            epochs=50,
            lr=1e-4,
            weight_decay=1e-5,
            temperature=0.07,
            normalize=True,
            device=device,
            seed=0,
        )

        trainer = TrajectoryEncoderTrainer(traj_config, game, autoencoder_ppo_agents)
        model, history = trainer.train()

        logger.info(f"Trajectory encoder trained. Final loss: {history[-1]:.4f}")

        # Create adapter and get encoder function (same for both players)
        adapter = TrajectoryEncoderAdapter(model, game, traj_config, policies=autoencoder_ppo_agents)
        adapter.set_normalization_stats(trainer.train_dataset.state_min, trainer.train_dataset.state_max)
        encoder_fn = adapter.get_encoder(device=device)

    # Set up downstream task
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    NUM_AGENTS_DOWNSTREAM = len(downstream_task_ppo_agents) if downstream_task_ppo_agents is not None else 500
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_DOWNSTREAM)
        ]

    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, agent, 0, False) for agent in downstream_task_ppo_agents],
        p2_policies=[PPOAgentPolicy(game, agent, 1, False) for agent in downstream_task_ppo_agents],
        p1_embeddings=[encoder_fn(agent).detach().cpu().numpy() for agent in downstream_task_ppo_agents],
        p2_embeddings=[encoder_fn(agent).detach().cpu().numpy() for agent in downstream_task_ppo_agents],
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

    # Baseline: predict mean of training set for everything
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(train_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    logger.info(f"\nBaseline (constant mean prediction from training set: {mean_payoff:.6f}):")
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
    results_dir = Path("results") / "task_b_trajectory"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_trajectory_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


def test_downstream_task_c_with_trajectory_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    pretrained_encoder: Optional[tuple] = None,  # (model, adapter, config) tuple
):
    """
    Test Task C with trajectory encoder.

    Task C: Predict expected payoff for (P1_agent, P2_agent, state) triples.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task C: Full state-conditioned matchup prediction with trajectory encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Use pre-trained encoder if provided, otherwise train from scratch
    if pretrained_encoder is not None:
        logger.info("\nUsing pre-trained trajectory encoder...")
        model, adapter, traj_config = pretrained_encoder
        encoder_fn = adapter.get_encoder(device=device)
    else:
        # Train trajectory encoder from scratch (shared for both P1 and P2)
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        NUM_AGENTS_AUTOENCODE = len(autoencoder_ppo_agents) if autoencoder_ppo_agents is not None else 500
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
                for _ in range(NUM_AGENTS_AUTOENCODE)
            ]

        logger.info(f"\nTraining trajectory encoder on {len(autoencoder_ppo_agents)} policies...")
        traj_config = TrajectoryEncoderConfig(
            hidden_dim=128,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            num_transitions=200,
            batch_size=8,
            epochs=50,
            lr=1e-4,
            weight_decay=1e-5,
            temperature=0.07,
            normalize=True,
            device=device,
            seed=0,
        )

        trainer = TrajectoryEncoderTrainer(traj_config, game, autoencoder_ppo_agents)
        model, history = trainer.train()

        logger.info(f"Trajectory encoder trained. Final loss: {history[-1]:.4f}")

        # Create adapter and get encoder function (same for both players)
        adapter = TrajectoryEncoderAdapter(model, game, traj_config, policies=autoencoder_ppo_agents)
        adapter.set_normalization_stats(trainer.train_dataset.state_min, trainer.train_dataset.state_max)
        encoder_fn = adapter.get_encoder(device=device)

    # Set up downstream task
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    NUM_AGENTS_DOWNSTREAM = len(downstream_task_ppo_agents) if downstream_task_ppo_agents is not None else 500
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

    # Baseline: predict mean of training set for everything
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(train_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    logger.info(f"\nBaseline (constant mean prediction from training set: {mean_payoff:.6f}):")
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
    results_dir = Path("results") / "task_c_trajectory"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_trajectory_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


if __name__ == "__main__":
    # Configuration
    USE_PSRO_AGENTS = False  # Set to True to use PSRO agents instead of random agents
    PSRO_HIDDEN_SIZE = 256   # Hidden size for PSRO agents (if USE_PSRO_AGENTS=True)

    # Set seed for reproducibility
    seed = 42
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    device = get_device_string()
    logger.info(f"Using device: {device}")

    game_name = "kuhn_poker"
    game = pyspiel.load_game(game_name)

    # Load opponent pool for encoder (should match what encoder was trained on)
    if USE_PSRO_AGENTS:
        logger.info(f"\nLoading PSRO policies for encoder opponent pool...")
        encoder_opponent_pool = load_ppo_agents_from_single_psro_folder(
            game_short_name=game_name,
            hidden_size=PSRO_HIDDEN_SIZE,
            shuffle=True,
            player_id=0,
            folder_selection='newest'
        )
        logger.info(f"Loaded {len(encoder_opponent_pool)} PSRO policies for opponent pool")
    else:
        logger.info(f"\nUsing uniform random opponent pool (encoder trained on random agents)")
        encoder_opponent_pool = None  # Will use uniform random opponent

    # Load pre-trained trajectory encoder
    checkpoint_path = "checkpoints/trajectory_encoder_kuhn_random_improved_500.pt"
    logger.info(f"\nLoading pre-trained encoder from {checkpoint_path}...")
    pretrained_encoder = load_trajectory_encoder_from_checkpoint(
        checkpoint_path,
        game,
        policies=encoder_opponent_pool,  # Pass policies for consistent opponent distribution
        device=device
    )

    # Create or load agents for downstream tasks
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    if USE_PSRO_AGENTS:
        # Load PSRO agents using the built-in function
        logger.info(f"\nLoading PSRO agents with hidden_size={PSRO_HIDDEN_SIZE}...")
        all_psro_agents = load_ppo_agents_from_single_psro_folder(game_short_name=game_name, hidden_size=PSRO_HIDDEN_SIZE, shuffle=True, player_id=0, folder_selection=0)
        logger.info(f"Loaded {len(all_psro_agents)} PSRO agents")
        downstream_agents = all_psro_agents[:120] if len(all_psro_agents) >= 50 else all_psro_agents
        logger.info(f"Using {len(downstream_agents)} PSRO agents for downstream tasks")
    else:
        # Create random agents for downstream tasks
        logger.info("\nCreating random agents for downstream tasks...")
        downstream_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(500)
        ]
        logger.info(f"Created {len(downstream_agents)} random agents for downstream tasks")

    # Test Task A
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK A")
    logger.info("="*80)
    test_downstream_task_a_with_trajectory_encoder(
        game,
        predictor_type="linear",
        downstream_task_ppo_agents=downstream_agents,
        device=device,
        pretrained_encoder=pretrained_encoder
    )

    # Test Task B
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK B")
    logger.info("="*80)
    test_downstream_task_b_with_trajectory_encoder(
        game,
        predictor_type="linear",
        downstream_task_ppo_agents=downstream_agents,
        device=device,
        pretrained_encoder=pretrained_encoder
    )

    # Test Task C
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK C")
    logger.info("="*80)
    test_downstream_task_c_with_trajectory_encoder(
        game,
        predictor_type="linear",
        downstream_task_ppo_agents=downstream_agents,
        device=device,
        pretrained_encoder=pretrained_encoder
    )

    logger.info("\n" + "="*80)
    logger.info("ALL TESTS COMPLETE")
    logger.info("="*80)
