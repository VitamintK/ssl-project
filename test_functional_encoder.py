"""
Test script for Functional Encoder on downstream tasks A, B, and C.

This script trains a functional autoencoder that matches PPO action probabilities,
then evaluates it on the three downstream prediction tasks:
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
from utils import get_device_string, make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, StatePayoffPredictor, set_seed, sample_random_states
from functional_autoencoder import (
    TrainingConfig,
    train_functional_autoencoder,
    FunctionalEncoderAdapter,
)
from weight_autoencoder import ppo_agent_to_vector, AutoencoderConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/functional_encoder_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def load_functional_encoder_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a pre-trained functional autoencoder from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, adapter, config)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct the model
    from weight_autoencoder import Autoencoder
    config = checkpoint['config']
    weight_dim = checkpoint['weight_dim']

    model = Autoencoder(
        weight_dim,
        config.autoencoder.hidden_dims,
        config.autoencoder.bottleneck_dim
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create adapter
    adapter = FunctionalEncoderAdapter(model, ppo_agent_to_vector)

    logger.info(f"Loaded functional autoencoder checkpoint from {checkpoint_path}")
    logger.info(f"Config: hidden_dims={config.autoencoder.hidden_dims}, "
                f"bottleneck_dim={config.autoencoder.bottleneck_dim}")

    return model, adapter, config


def test_downstream_task_a_with_functional_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    pretrained_encoder: Optional[tuple] = None,  # (model, adapter, config) tuple
):
    """
    Test Task A with functional encoder.

    Task A: Predict expected payoff for P1_agent against uniform random opponent.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task A: Payoff prediction vs uniform random with functional encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)

    # Use pre-trained encoder if provided, otherwise train from scratch
    if pretrained_encoder is not None:
        logger.info("\nUsing pre-trained functional autoencoder...")
        model, adapter, func_config = pretrained_encoder
        encoder_fn = adapter.get_encoder(device=device)
    else:
        # Train functional autoencoder from scratch
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        NUM_AGENTS_AUTOENCODE = len(autoencoder_ppo_agents) if autoencoder_ppo_agents is not None else 100
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
                for _ in range(NUM_AGENTS_AUTOENCODE)
            ]

        logger.info(f"\nTraining functional autoencoder on {len(autoencoder_ppo_agents)} policies...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=20,
            batch_size=16,
            lr=3e-4,
            device=device,
            dataset_fraction=0.3,
        )

        func_config = TrainingConfig(
            num_agents=NUM_AGENTS_AUTOENCODE,
            ppo_hidden_size=PPO_AGENT_HIDDEN_SIZE,
            autoencoder=ae_config,
        )

        model, epoch_losses = train_functional_autoencoder(
            func_config,
            game=game,
            agents=autoencoder_ppo_agents,
        )

        logger.info(f"Functional autoencoder trained. Final KL loss: {epoch_losses[-1]:.4f}")

        adapter = FunctionalEncoderAdapter(model, ppo_agent_to_vector)
        encoder_fn = adapter.get_encoder(device=device)

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

    # Compute embeddings for P1 agents
    p1_embeddings = [encoder_fn(agent) for agent in downstream_task_ppo_agents]

    # Dummy encoder and embedding for fixed P2 (uniform random)
    p2_embeddings = [np.array([0.0])]

    # Convert PPO agents to Policy objects
    from utils import PPOAgentPolicy
    p1_policies = [PPOAgentPolicy(game, agent, 0, use_observation=False) for agent in downstream_task_ppo_agents]
    p2_policies = [opponent_policy]

    predictor = PayoffPredictor(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=p1_embeddings,
        p2_embeddings=p2_embeddings,
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
    results_dir = Path("results") / "task_a_functional"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_functional_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


def test_downstream_task_b_with_functional_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    pretrained_encoder: Optional[tuple] = None,  # (model, adapter, config) tuple
):
    """
    Test Task B with functional encoder.

    Task B: Predict expected payoff for (P1_agent, P2_agent) matchups.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task B: Agent matchup payoff prediction with functional encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Use pre-trained encoder if provided, otherwise train from scratch
    if pretrained_encoder is not None:
        logger.info("\nUsing pre-trained functional autoencoder...")
        model, adapter, func_config = pretrained_encoder
        encoder_fn = adapter.get_encoder(device=device)
    else:
        # Train functional autoencoder from scratch (shared for both P1 and P2)
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        NUM_AGENTS_AUTOENCODE = len(autoencoder_ppo_agents) if autoencoder_ppo_agents is not None else 100
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
                for _ in range(NUM_AGENTS_AUTOENCODE)
            ]

        logger.info(f"\nTraining functional autoencoder on {len(autoencoder_ppo_agents)} policies...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=20,
            batch_size=16,
            lr=3e-4,
            device=device,
            dataset_fraction=0.3,
        )

        func_config = TrainingConfig(
            num_agents=NUM_AGENTS_AUTOENCODE,
            ppo_hidden_size=PPO_AGENT_HIDDEN_SIZE,
            autoencoder=ae_config,
        )

        model, epoch_losses = train_functional_autoencoder(
            func_config,
            game=game,
            agents=autoencoder_ppo_agents,
        )

        logger.info(f"Functional autoencoder trained. Final KL loss: {epoch_losses[-1]:.4f}")

        adapter = FunctionalEncoderAdapter(model, ppo_agent_to_vector)
        encoder_fn = adapter.get_encoder(device=device)

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

    # Compute embeddings for agents
    embeddings = [encoder_fn(agent) for agent in downstream_task_ppo_agents]

    # Convert PPO agents to Policy objects
    from utils import PPOAgentPolicy
    p1_policies = [PPOAgentPolicy(game, agent, 0, use_observation=False) for agent in downstream_task_ppo_agents]
    p2_policies = [PPOAgentPolicy(game, agent, 1, use_observation=False) for agent in downstream_task_ppo_agents]

    predictor = PayoffPredictor(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=embeddings,
        p2_embeddings=embeddings,
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
    results_dir = Path("results") / "task_b_functional"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_functional_encoder.pth"
    predictor.save(str(save_path))
    logger.info(f"Saved predictor to {save_path}")

    return predictor, history, val_metrics


def test_downstream_task_c_with_functional_encoder(
    game: pyspiel.Game,
    predictor_type: Literal["mlp", "linear"],
    autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
    downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
    device: str = "cpu",
    pretrained_encoder: Optional[tuple] = None,  # (model, adapter, config) tuple
):
    """
    Test Task C with functional encoder.

    Task C: Predict expected payoff for (P1_agent, P2_agent, state) triples.
    """
    logger.info("\n" + "="*80)
    logger.info(f"Task C: Full state-conditioned matchup prediction with functional encoder")
    logger.info(f"Predictor: {predictor_type}")
    logger.info("="*80)

    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Use pre-trained encoder if provided, otherwise train from scratch
    if pretrained_encoder is not None:
        logger.info("\nUsing pre-trained functional autoencoder...")
        model, adapter, func_config = pretrained_encoder
        encoder_fn = adapter.get_encoder(device=device)
    else:
        # Train functional autoencoder from scratch (shared for both P1 and P2)
        PPO_AGENT_HIDDEN_SIZE = 256
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)

        NUM_AGENTS_AUTOENCODE = len(autoencoder_ppo_agents) if autoencoder_ppo_agents is not None else 100
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
                for _ in range(NUM_AGENTS_AUTOENCODE)
            ]

        logger.info(f"\nTraining functional autoencoder on {len(autoencoder_ppo_agents)} policies...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=50,
            batch_size=16,
            lr=3e-4,
            device=device,
            dataset_fraction=0.3,
        )

        func_config = TrainingConfig(
            num_agents=NUM_AGENTS_AUTOENCODE,
            ppo_hidden_size=PPO_AGENT_HIDDEN_SIZE,
            autoencoder=ae_config,
        )

        model, epoch_losses = train_functional_autoencoder(
            func_config,
            game=game,
            agents=autoencoder_ppo_agents,
        )

        logger.info(f"Functional autoencoder trained. Final KL loss: {epoch_losses[-1]:.4f}")

        adapter = FunctionalEncoderAdapter(model, ppo_agent_to_vector)
        encoder_fn = adapter.get_encoder(device=device)

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
    results_dir = Path("results") / "task_c_functional"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_functional_encoder.pth"
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

    # Option 1: Load pre-trained functional encoder (if available)
    # checkpoint_path = "checkpoints/functional_autoencoder_kuhn.pt"
    # logger.info(f"\nLoading pre-trained encoder from {checkpoint_path}...")
    # pretrained_encoder = load_functional_encoder_from_checkpoint(checkpoint_path, device)

    # Option 2: Train from scratch (no pretrained encoder)
    pretrained_encoder = None
    logger.info("\nNo pre-trained encoder specified - will train from scratch for each task")

    # Create random agents for downstream tasks
    logger.info("\nCreating random agents for downstream tasks...")
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    # Create 100 random agents for autoencoding (if training from scratch)
    autoencoder_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(100)
    ]

    # Create 50 random agents for downstream tasks
    downstream_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(50)
    ]

    logger.info(f"Created {len(autoencoder_agents)} agents for encoder training")
    logger.info(f"Created {len(downstream_agents)} agents for downstream tasks")

    # Test Task A
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK A")
    logger.info("="*80)
    test_downstream_task_a_with_functional_encoder(
        game,
        predictor_type="linear",
        autoencoder_ppo_agents=autoencoder_agents,
        downstream_task_ppo_agents=downstream_agents,
        device=device,
        pretrained_encoder=pretrained_encoder
    )

    # Test Task B
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK B")
    logger.info("="*80)
    test_downstream_task_b_with_functional_encoder(
        game,
        predictor_type="linear",
        autoencoder_ppo_agents=autoencoder_agents,
        downstream_task_ppo_agents=downstream_agents,
        device=device,
        pretrained_encoder=pretrained_encoder
    )

    # Test Task C
    logger.info("\n" + "="*80)
    logger.info("TESTING TASK C")
    logger.info("="*80)
    test_downstream_task_c_with_functional_encoder(
        game,
        predictor_type="linear",
        autoencoder_ppo_agents=autoencoder_agents,
        downstream_task_ppo_agents=downstream_agents,
        device=device,
        pretrained_encoder=pretrained_encoder
    )

    logger.info("\n" + "="*80)
    logger.info("ALL TESTS COMPLETE")
    logger.info("="*80)
