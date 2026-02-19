from collections import defaultdict
import json
import math
import random
from typing import Literal, Optional
import pyspiel
from pathlib import Path
import logging
import numpy as np
from datetime import datetime

from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from psro import load_ppo_agents_from_psro, make_neupl_policies
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from functional_autoencoder import (
    TrainingConfig,
    train_functional_autoencoder,
    FunctionalEncoderAdapter,
)
from downstream import ExploitabilityPredictorTrainer, PayoffPredictor, StatePayoffPredictor, set_seed, sample_random_states
from weight_autoencoder import (
    AutoencoderConfig,
    WeightAutoencoder,
    ppo_agent_to_vector,
    save_autoencoder,
    load_autoencoder,
)

results = {'run': []}
run_start_time = None


def register_result(experiment_label: str, config_dict: dict, mse, baseline_mse):
    """
    Register a result for an experiment.

    Args:
        experiment_label: String label for the experiment
        config_dict: Dictionary with config parameters (prefixed for method-specific params)
        mse: Mean squared error
        baseline_mse: Baseline mean squared error
    """
    # Convert tensors to floats
    if hasattr(mse, "item"):
        mse = mse.item()
    if hasattr(baseline_mse, "item"):
        baseline_mse = baseline_mse.item()

    # Find existing run with same experiment_label
    existing_run = None
    for run in results['run']:
        if run['experiment_label'] == experiment_label:
            existing_run = run
            break

    if existing_run:
        # Append to existing results
        existing_run['results'].append({
            'mse': mse,
            'baseline_mse': baseline_mse
        })
    else:
        # Create new run entry
        results['run'].append({
            'experiment_label': experiment_label,
            'config': config_dict,
            'results': [{
                'mse': mse,
                'baseline_mse': baseline_mse
            }]
        })


def save_results():
    """Save results to both main file and timestamped archive."""
    # Print summary
    for run in results['run']:
        print(run['experiment_label'])
        for result in run['results']:
            print(f"{result['mse']:.6f},{result['baseline_mse']:.6f}")

    # Save to main file
    with open('results/all_downstream_tasks.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save to archive with timestamp
    archive_dir = Path("results/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = run_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = archive_dir / f"downstream_results_{timestamp}.json"

    with open(archive_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  - results/all_downstream_tasks.json")
    print(f"  - {archive_path}")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/all_downstream_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


def get_feel_for_values(values: np.ndarray, label: str = "Values"):
    """
    Print statistics and a sample of values to get a feel for the data distribution.

    Args:
        values: Array of values to analyze
        label: Label for the values (e.g., "Ground Truth Payoffs")
    """
    print(f"\n{'='*80}")
    print(f"Getting a feel for {label}")
    print(f"{'='*80}")

    # Flatten if needed
    if values.ndim > 1:
        values_flat = values.flatten()
    else:
        values_flat = values

    # Basic statistics
    print(f"Shape: {values.shape}")
    print(f"Count: {len(values_flat)}")
    print(f"Min: {values_flat.min():.6f}")
    print(f"Max: {values_flat.max():.6f}")
    print(f"Mean: {values_flat.mean():.6f}")
    print(f"Median: {np.median(values_flat):.6f}")
    print(f"Std: {values_flat.std():.6f}")

    # Percentiles
    percentiles = [0, 25, 50, 75, 100]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p}%: {np.percentile(values_flat, p):.6f}")

    # Sample of 20 values
    print(f"\nSample of 20 values:")
    sample_indices = np.linspace(0, len(values_flat) - 1, min(20, len(values_flat)), dtype=int)
    for i, idx in enumerate(sample_indices):
        print(f"  [{idx}]: {values_flat[idx]:.6f}")

    # Simple text histogram (10 bins)
    print(f"\nHistogram (10 bins):")
    hist, bin_edges = np.histogram(values_flat, bins=10)
    max_bar_width = 50
    max_count = hist.max()

    for i in range(len(hist)):
        bar_width = int((hist[i] / max_count) * max_bar_width) if max_count > 0 else 0
        bar = '█' * bar_width
        print(f"  [{bin_edges[i]:7.4f}, {bin_edges[i+1]:7.4f}): {hist[i]:5d} {bar}")

    print(f"{'='*80}\n")

def test_downstream_task_a(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear", "random_forest"],
        encoder_type: Literal["identity", "weight_autoencoder", "functional_autoencoder"],
        autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
        downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
        experiment_label: str = "downstream_a",
        device: str = "cpu",
        functional_dataset_fraction: float = 1.0,
):
    """
    Test the PayoffPredictor on Kuhn Poker with the specified configuration.

    Args:
        game: The OpenSpiel game to use
        predictor_type: Type of predictor ("mlp", "linear", or "random_forest")
        encoder_type: Type of encoder ("identity", "weight_autoencoder", or "functional_autoencoder")
        experiment_label: Label for the experiment subdirectory (default: "downstream_a")
        functional_dataset_fraction: Fraction of functional AE training data to keep (only used for the functional encoder)
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    encoder_defaults = {
        "identity": {
            "ppo_agent_hidden_size": 256,
            "predictor_agent_count": 100,
            # "encoder_hidden_size": None,
            "autoencoder_agent_count": 0,
        },
        "weight_autoencoder": {
            "ppo_agent_hidden_size": 256,
            "predictor_agent_count": 100,
            # "encoder_hidden_size": 256,
            "autoencoder_agent_count": 100,
        },
        "functional_autoencoder": {
            "ppo_agent_hidden_size": 64,
            "predictor_agent_count": 100,
            # "encoder_hidden_size": 64,
            "autoencoder_agent_count": 100,
        },
    }

    if encoder_type not in encoder_defaults:
        raise ValueError(f"Invalid encoder type: {encoder_type}")

    defaults = encoder_defaults[encoder_type]
    ppo_agent_hidden_size = defaults["ppo_agent_hidden_size"]
    predictor_agent_count = defaults["predictor_agent_count"]
    # encoder_hidden_size = defaults["encoder_hidden_size"]
    autoencoder_agent_count = defaults["autoencoder_agent_count"]

    if encoder_type == 'weight_autoencoder':
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, ppo_agent_hidden_size)
                for _ in range(autoencoder_agent_count)
            ]
        print("\nTraining autoencoder on agent weights...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            device=device,
        )
        weight_autoencoder = WeightAutoencoder(ae_config, autoencoder_ppo_agents, ppo_agent_to_vector)
        autoencoder_model, ae_history = weight_autoencoder.train()
        save_autoencoder(
            autoencoder_model,
            ae_config,
            Path("results") / experiment_label / f"{game_short_name}_autoencoder.pth",
        )
        print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
            f"val loss: {ae_history['val_loss'][-1]:.6f}")
        encoder_fn = weight_autoencoder.get_encoder(device=device)
    elif encoder_type == 'functional_autoencoder':
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, ppo_agent_hidden_size)
                for _ in range(autoencoder_agent_count)
            ]
        if not (0 < functional_dataset_fraction <= 1):
            raise ValueError("functional_dataset_fraction must be in the interval (0, 1].")
        functional_cfg = TrainingConfig(
            num_agents=len(autoencoder_ppo_agents),
            ppo_hidden_size=ppo_agent_hidden_size,
            autoencoder=AutoencoderConfig(
                hidden_dims=(512, 256),
                bottleneck_dim=128,
                epochs=10,
                batch_size=64,
                lr=3e-4,
                device=device,
                dataset_fraction=functional_dataset_fraction,
            ),
        )
        print("\nTraining functional functional encoder...")
        functional_model, functional_history = train_functional_autoencoder(
            functional_cfg,
            game=game,
            agents=autoencoder_ppo_agents,
        )
        print(f"functional encoder trained. Final KL: {functional_history[-1]:.6f}")
        encoder_adapter = FunctionalEncoderAdapter(functional_model)
        encoder_fn = encoder_adapter.get_encoder(device=device)
    elif encoder_type == 'identity':
        encoder_fn = ppo_agent_to_vector
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")

    # Create and train payoff predictor
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(
                num_actions,
                info_state_size,
                device,
                layer_init,
                ppo_agent_hidden_size,
            )
            for _ in range(predictor_agent_count)
        ]

    # Common data preparation
    p1_embeddings = np.array([encoder_fn(agent).detach().cpu().numpy() for agent in downstream_task_ppo_agents])
    p1_policies = [PPOAgentPolicy(game, agent, 0, False) for agent in downstream_task_ppo_agents]
    p2_policies = [opponent_policy]

    # Create predictor (unified interface for all types)
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    elif predictor_type == "random_forest":
        hidden_dims = None  # Not used for random forest
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    predictor = PayoffPredictor(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=p1_embeddings,
        p2_embeddings=[np.array([0])],
        model_type=predictor_type,
        hidden_dims=hidden_dims,
        dropout=0.0,
        device="cpu"
    )

    # Train the model
    print(f"\nTraining {predictor_type} predictor...")
    history = predictor.train(
        num_epochs=5000,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Get a feel for the ground truth payoffs
    get_feel_for_values(predictor.ground_truth_payoffs, label="Ground Truth Payoffs (Task A)")

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Compute baseline (predict mean of training set)
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    # Note: ground_truth_payoffs is already 1D for single opponent case
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    mean_payoff2 = np.mean(train_payoffs)
    baseline_mse2 = np.mean((val_payoffs - mean_payoff2) ** 2)
    
    # print(f"\nBaseline (constant prediction w/val mean: {mean_payoff:.6f}):")
    # print(f"MSE: {baseline_mse:.6f}")
    # print(f"MAE: {baseline_mae:.6f}")
    print(f"\nBaseline 2 (constant prediction w/train mean: {mean_payoff2:.6f}):")
    # print(f"MSE: {baseline_mse2:.6f}")
    logger.info(f"Baseline (constant prediction w/train mean: {baseline_mse2:.6f})")
    print(f"\nModel improvement over baseline:")
    logger.info(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse2)*100:.2f}%")
    # print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate payoff predictor on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics

def test_downstream_task_a_(
        game: pyspiel.Game,
        policies: list[policy_lib.Policy],
        embeddings: list[np.ndarray],
        predictor_type: Literal["mlp", "linear", "random_forest"],
        exp_label: str = "downstream_a_neupl",
        predictor_dropout: float = 0.0,
        device: str = "cpu",
        config: Optional[dict] = None,
):
    game_short_name = game.get_type().short_name
    opponent_policy = policy_lib.UniformRandomPolicy(game)
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    elif predictor_type == "random_forest":
        hidden_dims = None  # Not used for random forest
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    predictor = PayoffPredictor(
        game=game,
        p1_policies=policies,
        p2_policies=[opponent_policy],
        p1_embeddings=embeddings,
        p2_embeddings=[np.array([0])],
        model_type=predictor_type,
        hidden_dims=hidden_dims,
        dropout=predictor_dropout,
        device=device
    )
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=5000,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    get_feel_for_values(predictor.ground_truth_payoffs, label="Ground Truth Payoffs (Task A)")

    # Evaluate payoff predictor on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything (validation set)
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    # Note: ground_truth_payoffs is already 1D for single opponent case
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    # baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))
    # Baseline 2: predict mean for everything (training set)
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    mean_payoff2 = np.mean(train_payoffs)
    baseline_mse2 = np.mean((val_payoffs - mean_payoff2) ** 2)

    # print(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    print(f"baseline cheating MSE (mean of val): {baseline_mse:.6f}")
    # print(f"MAE: {baseline_mae:.6f}")
    print(f"\nBaseline (constant mean prediction of {mean_payoff2:.6f}):")
    logger.info(f"Baseline (constant prediction w/train mean): {baseline_mse2:.6f} MSE")
    print(f"\nModel improvement over baseline:")
    logger.info(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse2)*100:.2f}%")
    # print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate payoff predictor on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Use provided config or create minimal default
    if config is None:
        config = {
            'game': game_short_name,
            'task': 'task_a',
            'task_a_predictor_type': predictor_type,
        }

    register_result(exp_label, config, val_metrics['mse'], baseline_mse2)

    # Save the payoff predictor model
    results_dir = Path("results") / exp_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics

def test_downstream_task_a_with_neupl(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"],
        experiment_label: str = "downstream_a_neupl",
        device: str = "cpu",
):
    game_short_name = game.get_type().short_name
    policies_and_embeddings = make_neupl_policies(game_short_name, hidden_size=256, policy_embedding_size=64, original_num_policies=22, num_policies_to_make=3000)
    opponent_policy = policy_lib.UniformRandomPolicy(game)
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    predictor = PayoffPredictor(
        game=game,
        p1_policies=[p_e[1] for p_e in policies_and_embeddings[0]],
        p2_policies=[opponent_policy],
        p1_embeddings=[p_e[0].detach().cpu().numpy() for p_e in policies_and_embeddings[0]],
        p2_embeddings=[np.array([0])],
        hidden_dims=hidden_dims,
        dropout=0.0,
        device=device
    )
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate payoff predictor on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything (validation set)
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))
    # Baseline 2: predict mean for everything (training set)
    train_payoffs = predictor.ground_truth_payoffs[predictor.train_indices]
    mean_payoff2 = np.mean(train_payoffs)
    baseline_mse2 = np.mean((val_payoffs - mean_payoff2) ** 2)

    print(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"MAE: {baseline_mae:.6f}")
    print(f"\nBaseline 2 (constant mean prediction: {mean_payoff2:.6f}):")
    logger.info(f"Baseline2 (constant prediction w/train mean: {baseline_mse2:.6f})")
    print(f"\nModel improvement over baseline:")
    print(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse2)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate payoff predictor on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics
    

def test_downstream_task_b(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"],
        encoder_type: Literal["identity", "weight_autoencoder"],
        experiment_label: str = "downstream_b",
        device: str = "cpu",
):
    """
    Test the PayoffPredictor on agent vs agent matchups.

    Args:
        game: The OpenSpiel game to use
        predictor_type: Type of predictor ("mlp" or "linear")
        encoder_type: Type of encoder ("identity" or "weight_autoencoder")
        experiment_label: Label for the experiment subdirectory (default: "downstream_b")
        device: Device to use for training
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    if encoder_type == 'weight_autoencoder':
        # Train autoencoder on a separate set of agents
        NUM_AGENTS_AUTOENCODE = 1000
        print("\nCreating agents for autoencoder training...")
        agents_for_ae = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_AUTOENCODE)
        ]
        print("\nTraining autoencoder on agent weights...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=50,
            batch_size=16,
            lr=1e-3,
            device=device,
        )
        weight_autoencoder = WeightAutoencoder(ae_config, agents_for_ae, ppo_agent_to_vector)
        _, ae_history = weight_autoencoder.train()
        print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
              f"val loss: {ae_history['val_loss'][-1]:.6f}")
        encoder_fn = weight_autoencoder.get_encoder(device=device)
    elif encoder_type == 'identity':
        encoder_fn = ppo_agent_to_vector
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")

    # Create separate agents for downstream task
    NUM_P1_AGENTS = 500
    NUM_P2_AGENTS = 500
    print("\nCreating P1 and P2 agents for downstream task...")
    p1_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(NUM_P1_AGENTS)
    ]
    p2_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(NUM_P2_AGENTS)
    ]

    # Create and train payoff predictor
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor_type: {predictor_type}")

    predictor = PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, agent, 0, False) for agent in p1_agents],
        p2_policies=[PPOAgentPolicy(game, agent, 1, False) for agent in p2_agents],
        p1_embeddings=[encoder_fn(agent).detach().cpu().numpy() for agent in p1_agents],
        p2_embeddings=[encoder_fn(agent).detach().cpu().numpy() for agent in p2_agents],
        hidden_dims=hidden_dims,
        dropout=0.0,
        device=device
    )

    # Train the model
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    print(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"MAE: {baseline_mae:.6f}")
    print(f"\nModel improvement over baseline:")
    print(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics

def test_downstream_task_c(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"],
        encoder_type: Literal["identity", "weight_autoencoder"],
        experiment_label: str = "downstream_c",
        device: str = "cpu",
):
    """
    Test the StatePayoffPredictor with state-conditioned predictions.

    Args:
        game: The OpenSpiel game to use
        predictor_type: Type of predictor ("mlp" or "linear")
        encoder_type: Type of encoder ("identity" or "weight_autoencoder")
        experiment_label: Label for the experiment subdirectory (default: "downstream_c")
        device: Device to use for training
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    NUM_STATES = 20
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    if encoder_type == 'weight_autoencoder':
        # Train autoencoder on a separate set of agents
        NUM_AGENTS_AUTOENCODE = 1000
        print("\nCreating agents for autoencoder training...")
        agents_for_ae = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
            for _ in range(NUM_AGENTS_AUTOENCODE)
        ]
        print("\nTraining autoencoder on agent weights...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            device=device,
        )
        weight_autoencoder = WeightAutoencoder(ae_config, agents_for_ae, ppo_agent_to_vector)
        _, ae_history = weight_autoencoder.train()
        print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
              f"val loss: {ae_history['val_loss'][-1]:.6f}")
        encoder_fn = weight_autoencoder.get_encoder(device=device)
    elif encoder_type == 'identity':
        encoder_fn = ppo_agent_to_vector
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")

    # Create separate agents for downstream task
    NUM_P1_AGENTS = 100
    NUM_P2_AGENTS = 100
    print("\nCreating P1 and P2 agents for downstream task...")
    p1_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(NUM_P1_AGENTS)
    ]
    p2_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(NUM_P2_AGENTS)
    ]

    # Create state sampler
    def state_sampler(game, num_states):
        return sample_random_states(game, num_states, max_depth=5)

    # Create and train payoff predictor
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor_type: {predictor_type}")

    # Create and train state-conditioned predictor
    predictor = StatePayoffPredictor(
        game=game,
        p1_agents=p1_agents,
        p2_agents=p2_agents,
        p1_encoder_fn=encoder_fn,
        p2_encoder_fn=encoder_fn,
        state_sampler=state_sampler,
        num_states_per_pair=NUM_STATES,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device=device
    )

    # Train the model
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    mean_payoff = np.mean(val_payoffs)
    baseline_mse = np.mean((val_payoffs - mean_payoff) ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs - mean_payoff))

    print(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"MAE: {baseline_mae:.6f}")
    print(f"\nModel improvement over baseline:")
    print(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics

def test_downstream_task_d(
        game_short_name: str,
        player_id: int,
        policies: list[policy_lib.Policy],
        embeddings: list[np.ndarray],
        predictor_type: Literal["mlp", "linear"],
        exp_label: str = "downstream_d",
        device: str = "cpu",
        config: Optional[dict] = None,
):
    """
    Test the ExploitabilityPredictorTrainer with exploitability predictions.
    """
    # game_short_name = game.get_type().short_name
    # info_state_size = game.information_state_tensor_shape()
    # num_actions = game.num_distinct_actions()
    trainer = ExploitabilityPredictorTrainer(
        game=game,
        policies=policies,
        embeddings=embeddings,
        player_id=player_id,
        hidden_dims=[],
        dropout=0.0,
        device=device
    )
    trainer.train()

    # Get a feel for the ground truth payoffs
    get_feel_for_values(trainer.ground_truth_payoffs.cpu().numpy(), label="Ground Truth Payoffs (Task D)")

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = trainer.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict mean for everything
    train_payoffs = trainer.ground_truth_payoffs[trainer.train_indices]
    val_payoffs = trainer.ground_truth_payoffs[trainer.val_indices]
    mean_payoff = train_payoffs.mean()
    baseline_mse = ((val_payoffs - mean_payoff) ** 2).mean()
    baseline_mae = (val_payoffs - mean_payoff).abs().mean()

    print(f"\nBaseline (constant mean prediction: {mean_payoff:.6f}):")
    logger.info(f"Baseline (constant mean prediction): {baseline_mse:.6f}")
    logger.info(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Use provided config or create minimal default
    if config is None:
        config = {
            'game': game_short_name,
            'player_id': player_id,
            'task': 'task_d',
            'task_d_predictor_type': predictor_type,
        }

    register_result(exp_label, config, val_metrics['mse'], baseline_mse)

def get_policies_and_embeddings(game, player_id: int, ppo_agents: list[PPOAgent], experiment_label: str, game_short_name: str, device: str):
    print("\nTraining autoencoder on agent weights...")
    autoencoder_ppo_agents = ppo_agents[:len(ppo_agents)//2]
    downstream_ppo_agents = ppo_agents[len(ppo_agents)//2:]
    ae_config = AutoencoderConfig(
        hidden_dims=(512, 256),
        bottleneck_dim=64,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=device,
    )
    weight_autoencoder = WeightAutoencoder(ae_config, autoencoder_ppo_agents, ppo_agent_to_vector)
    autoencoder_model, ae_history = weight_autoencoder.train()
    save_autoencoder(
        autoencoder_model,
        ae_config,
        Path("results") / experiment_label / f"{game_short_name}_autoencoder.pth",
    )
    print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
        f"val loss: {ae_history['val_loss'][-1]:.6f}")
    encoder_fn = weight_autoencoder.get_encoder(device=device)
    embeddings = [encoder_fn(agent).detach().cpu().numpy() for agent in downstream_ppo_agents]
    policies = [PPOAgentPolicy(game, agent, player_id, False) for agent in downstream_ppo_agents]
    return policies, embeddings

def get_policies_and_embeddings2(game, player_id: int, ppo_agents: list[PPOAgent], experiment_label: str, game_short_name: str, device: str):
    embeddings = [ppo_agent_to_vector(agent).detach().cpu().numpy() for agent in ppo_agents]
    policies = [PPOAgentPolicy(game, agent, player_id, False) for agent in ppo_agents]
    return policies, embeddings


def run_all():
    device = get_device_string()
    print("Using device:", device)
    # set_seed(42)

    # GAME_NAMES = ["kuhn_poker", "leduc_poker"]
    GAME_NAMES = ["kuhn_poker"]
    for game_name in GAME_NAMES:
        game = pyspiel.load_game(game_name)
        if game_name == "kuhn_poker":
            psro_ppo_agents_256 = load_ppo_agents_from_psro(hidden_size=256, shuffle=True)
            first_half, second_half = psro_ppo_agents_256[:len(psro_ppo_agents_256)//2], psro_ppo_agents_256[len(psro_ppo_agents_256)//2:]
            exp_label = f"Task A: psro {game_name} linear weight_autoencoder"
            logger.info(f"Number of PSRO agents: {len(psro_ppo_agents_256)}")
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", autoencoder_ppo_agents=first_half, downstream_task_ppo_agents=second_half,
device=device)
            exp_label = f"Task A: psro {game_name} linear identity"
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", autoencoder_ppo_agents=first_half, downstream_task_ppo_agents=second_half, device=device)
            exp_label = f"Task A: psro {game_name} linear functional_encoder "
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(
                game,
                predictor_type="linear",
                encoder_type="functional_autoencoder",
                autoencoder_ppo_agents=first_half,
                downstream_task_ppo_agents=second_half,
                experiment_label=exp_label,
                device=device,
            )
            exp_label = f"Task A: neupl {game_name} linear"
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a_with_neupl(game, predictor_type="linear", device=device)
        exp_label = f"Task A: random {game_name} linear weight_autoencoder"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", device=device)

        exp_label = f"Task A: random {game_name} linear identity"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", device=device)

        exp_label = f"Task A: random {game_name} linear functional_encoder"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(
            game,
            predictor_type="linear",
            encoder_type="functional_autoencoder",
            experiment_label=exp_label,
            device=device,
            functional_dataset_fraction=0.01,
        )

        exp_label = f"Task B: random {game_name} linear weight_autoencoder"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_b(game, predictor_type="linear", encoder_type="weight_autoencoder", device=device)
        exp_label = f"Task B: random {game_name} linear identity"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_b(game, predictor_type="linear", encoder_type="identity", device=device)



runs_to_load = {
    True: [
        '2025-12-12_11-19-54-744419_40b',
        '2025-12-11_23-59-52-950280_b14',
        '2025-12-11_18-00-46-759499_234',
    ],
    False: [
        '2025-12-12_11-19-52-549639_bb8',
        '2025-12-11_23-59-48-262932_dfa',
        '2025-12-11_18-00-37-642050_b0d',
    ],
}
if __name__ == "__main__":
    # run_all()
    # exit()

    RUN_TASK_A = True
    RUN_TASK_B = True
    RUN_TASK_C = True
    RUN_TASK_D = True

    RUN_NEUPL = True
    RUN_PSRO = False
    RUN_RANDOM = False

    # Initialize run start time
    run_start_time = datetime.now()

    device = get_device_string()
    print("Using device:", device)
    # game = pyspiel.load_game("kuhn_poker")
    game = pyspiel.load_game("leduc_poker")
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    for seed_num in range(3):
        # for player_id in range(2):
        if RUN_NEUPL:
            for use_randall_loss in [True, False]:
                # logger.info(f"--Running task: NEUPL {game_short_name} player_id={player_id} use_randall_loss={use_randall_loss}")
                N = 1000 # 1500 or 3000 works fine for kuhn
                NEUPL_SAMPLING_MODE = "gaussian"
                INTERPOLATE_PRENORM = True
                PREDICTOR_TYPE = 'random_forest'
                neupl_config = {
                    'use_randall_loss': use_randall_loss,
                    'hidden_size': 256,
                    'policy_embedding_size': 64,
                }
                for player_id in range(2):
                    policies_and_embeddings = make_neupl_policies(
                        game_short_name,
                        neupl_config=neupl_config,
                        original_num_policies=23,
                        num_policies_to_make=N,
                        directory=runs_to_load[use_randall_loss][seed_num],
                        interpolate_prenorm=INTERPOLATE_PRENORM,
                        sampling_mode=NEUPL_SAMPLING_MODE,
                    )
                    policies, embeddings = [p_e[1] for p_e in policies_and_embeddings[player_id]], [p_e[0].detach().cpu().numpy() for p_e in policies_and_embeddings[player_id]]

                    if RUN_TASK_A:
                        exp_label = f'{game_short_name} neupl p{player_id} randloss={use_randall_loss} N={N} Task A'
                        logger.info(f"--Running task: NEUPL {game_short_name} player_id={player_id} use_randall_loss={use_randall_loss} Task A")
                        test_downstream_task_a_(
                            game,
                            exp_label=exp_label,
                            policies=policies,
                            embeddings=embeddings,
                            predictor_type=PREDICTOR_TYPE,
                            device=device,
                            config={
                                'game': game_short_name,
                                'agent_source': 'neupl',
                                'player_id': player_id,
                                'task': 'task_a',
                                'task_a_predictor_type': PREDICTOR_TYPE,
                                'neupl_use_randall_loss': use_randall_loss,
                                'neupl_interpolate_prenorm': INTERPOLATE_PRENORM,
                                'neupl_sampling_mode': NEUPL_SAMPLING_MODE,
                                'neupl_num_policies': N
                            }
                        )
                    if RUN_TASK_D:
                        exp_label = f'{game_short_name} neupl p{player_id} randloss={use_randall_loss} N={N} Task D'
                        logger.info(f"--Running task: NEUPL {game_short_name} player_id={player_id} use_randall_loss={use_randall_loss} Task D")
                        test_downstream_task_d(
                            game_short_name,
                            exp_label=exp_label,
                            player_id=player_id,
                            policies=policies,
                            embeddings=embeddings,
                            predictor_type=PREDICTOR_TYPE,
                            device=device,
                            config={
                                'game': game_short_name,
                                'agent_source': 'neupl',
                                'player_id': player_id,
                                'task': 'task_d',
                                'task_d_predictor_type': PREDICTOR_TYPE,
                                'neupl_use_randall_loss': use_randall_loss,
                                'neupl_interpolate_prenorm': INTERPOLATE_PRENORM,
                                'neupl_sampling_mode': NEUPL_SAMPLING_MODE,
                                'neupl_num_policies': N
                            }
                        )
        if RUN_PSRO:
            for player_id in range(2):
            # logger.info(f"--Running tasks: PSRO {game_short_name} player_id={player_id}")
                psro_ppo_agents_256 = load_ppo_agents_from_psro(game_short_name=game_short_name, hidden_size=256, player_id=player_id, shuffle=True)
                policies, embeddings = get_policies_and_embeddings(game, player_id, psro_ppo_agents_256, "psro_" + game_short_name, game_short_name, device)
                _, identity_embeddings = get_policies_and_embeddings2(game, player_id, psro_ppo_agents_256, "psro_" + game_short_name, game_short_name, device)
                if RUN_TASK_A:
                    exp_label = f'{game_short_name} psro {player_id} reconstruction-autoencoder Task A'
                    logger.info(f"--Running task: PSRO {game_short_name} player_id={player_id} reconstruction-autoencoder Task A")
                    test_downstream_task_a_(
                        game,
                        exp_label=exp_label,
                        policies=policies,
                        embeddings=embeddings,
                        predictor_type="linear",
                        device=device,
                        config={
                            'game': game_short_name,
                            'agent_source': 'psro',
                            'encoder_type': 'reconstruction-autoencoder',
                            'player_id': player_id,
                            'task': 'task_a',
                            'task_a_predictor_type': 'linear'
                        }
                    )
                    exp_label = f'{game_short_name} psro {player_id} identity Task A'
                    logger.info(f"--Running task: PSRO {game_short_name} player_id={player_id} identity Task A")
                    test_downstream_task_a_(
                        game,
                        exp_label=exp_label,
                        policies=policies,
                        embeddings=identity_embeddings,
                        predictor_type="linear",
                        device=device,
                        config={
                            'game': game_short_name,
                            'agent_source': 'psro',
                            'encoder_type': 'identity',
                            'player_id': player_id,
                            'task': 'task_a',
                            'task_a_predictor_type': 'linear'
                        }
                    )
                if RUN_TASK_D:
                    exp_label = f'{game_short_name} psro {player_id} reconstruction-autoencoder Task D'
                    logger.info(f"--Running task: PSRO {game_short_name} player_id={player_id} reconstruction-autoencoder Task D")
                    test_downstream_task_d(
                        game_short_name,
                        exp_label=exp_label,
                        player_id=player_id,
                        policies=policies,
                        embeddings=embeddings,
                        predictor_type="linear",
                        device=device,
                        config={
                            'game': game_short_name,
                            'agent_source': 'psro',
                            'encoder_type': 'reconstruction-autoencoder',
                            'player_id': player_id,
                            'task': 'task_d',
                            'task_d_predictor_type': 'linear'
                        }
                    )
                    exp_label = f'{game_short_name} psro {player_id} identity Task D'
                    logger.info(f"--Running task: PSRO {game_short_name} player_id={player_id} identity Task D")
                    test_downstream_task_d(
                        game_short_name,
                        exp_label=exp_label,
                        player_id=player_id,
                        policies=policies,
                        embeddings=identity_embeddings,
                        predictor_type="linear",
                        device=device,
                        config={
                            'game': game_short_name,
                            'agent_source': 'psro',
                            'encoder_type': 'identity',
                            'player_id': player_id,
                            'task': 'task_d',
                            'task_d_predictor_type': 'linear'
                        }
                    )
        if RUN_RANDOM:
            N = 1000
            ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, 256) for _ in range(N)]
            player_id = 0
            policies, embeddings = get_policies_and_embeddings(game, player_id, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
            _, identity_embeddings = get_policies_and_embeddings2(game, player_id, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
            if RUN_TASK_A:
                exp_label = f'{game_short_name} ppo random {player_id} reconstruction-autoencoder Task A'
                logger.info(f"--Running task: PPO random {game_short_name} reconstruction-autoencoder Task A")
                test_downstream_task_a_(
                    game,
                    exp_label=exp_label,
                    policies=policies,
                    embeddings=embeddings,
                    predictor_type="linear",
                    device=device,
                    config={
                        'game': game_short_name,
                        'agent_source': 'ppo_random',
                        'encoder_type': 'reconstruction-autoencoder',
                        'player_id': player_id,
                        'task': 'task_a',
                        'task_a_predictor_type': 'linear'
                    }
                )
                exp_label = f'{game_short_name} ppo random {player_id} identity Task A'
                logger.info(f"--Running task: PPO random {game_short_name} identity Task A")
                test_downstream_task_a_(
                    game,
                    exp_label=exp_label,
                    policies=policies,
                    embeddings=identity_embeddings,
                    predictor_type="linear",
                    device=device,
                    config={
                        'game': game_short_name,
                        'agent_source': 'ppo_random',
                        'encoder_type': 'identity',
                        'player_id': player_id,
                        'task': 'task_a',
                        'task_a_predictor_type': 'linear'
                    }
                )
            if RUN_TASK_D:
                exp_label = f'{game_short_name} ppo random {player_id} reconstruction-autoencoder Task D'
                logger.info(f"--Running task: PPO random {game_short_name} reconstruction-autoencoder Task D")
                test_downstream_task_d(
                    game_short_name,
                    exp_label=exp_label,
                    player_id=player_id,
                    policies=policies,
                    embeddings=embeddings,
                    predictor_type="linear",
                    device=device,
                    config={
                        'game': game_short_name,
                        'agent_source': 'ppo_random',
                        'encoder_type': 'reconstruction-autoencoder',
                        'player_id': player_id,
                        'task': 'task_d',
                        'task_d_predictor_type': 'linear'
                    }
                )
                exp_label = f'{game_short_name} ppo random {player_id} identity Task D'
                logger.info(f"--Running task: PPO random {game_short_name} identity Task D")
                test_downstream_task_d(
                    game_short_name,
                    exp_label=exp_label,
                    player_id=player_id,
                    policies=policies,
                    embeddings=identity_embeddings,
                    predictor_type="linear",
                    device=device,
                    config={
                        'game': game_short_name,
                        'agent_source': 'ppo_random',
                        'encoder_type': 'identity',
                        'player_id': player_id,
                        'task': 'task_d',
                        'task_d_predictor_type': 'linear'
                    }
                )
    save_results()