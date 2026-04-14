from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import random
from typing import Literal, Optional
import pyspiel
from pathlib import Path
import logging
import numpy as np
from datetime import datetime

from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from psro import load_ppo_agents_from_psro, make_neupl_policies, select_neupl_directory
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from functional_autoencoder import (
    TrainingConfig,
    train_functional_autoencoder,
    FunctionalEncoderAdapter,
)
# from downstream_deprecated import ExploitabilityPredictorTrainer, PayoffPredictor, StatePayoffPredictor, set_seed, sample_random_states
from downstream import PayoffPredictor, StatePayoffPredictor, ExploitabilityPredictor, sample_random_states
from weight_autoencoder import (
    AutoencoderConfig,
    WeightAutoencoder,
    ppo_agent_to_vector,
    save_autoencoder,
    load_autoencoder,
)
from tasks import run_task_a, run_task_d, run_task_e
from config import TaskAConfig, TaskDConfig, ModelConfig, TaskEConfig, ExperimentInfo
from tqdm import tqdm

results = {'run': []}
run_start_time = None


def _strip_arrays(result: dict) -> dict:
    """Remove list-valued entries from metrics dicts to keep the JSON small."""
    out = {}
    for k, v in result.items():
        if isinstance(v, dict):
            out[k] = {mk: mv for mk, mv in v.items() if not isinstance(mv, list)}
        else:
            out[k] = v
    return out


def register_result(experiment_label: str, result: dict, save: bool = True):
    existing = next((r for r in results['run'] if r['experiment_label'] == experiment_label), None)
    stripped = _strip_arrays(result)
    if existing:
        existing['results'].append(stripped)
    else:
        results['run'].append({'experiment_label': experiment_label, 'results': [stripped]})

    if save:
        save_results()


def save_results():
    """Save results to both main file and timestamped archive."""
    # Print summary
    # for run in results['run']:
    #     print(run['experiment_label'])
    #     for result in run['results']:
    #         print(f"{result['mse']:.6f},{result['baseline_mse']:.6f}")

    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Save to main file
    with open('results/all_downstream_tasks.json', 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)

    # Save to archive with timestamp
    archive_dir = Path("results/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = run_start_time.strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = archive_dir / f"downstream_results_{timestamp}.json"

    with open(archive_path, 'w') as f:
        json.dump(results, f, indent=2, cls=_NumpyEncoder)

    print(f"\nResults saved to:")
    print(f"  - results/all_downstream_tasks.json")
    print(f"  - {archive_path}")

logger = logging.getLogger("ssl_project")
logger.setLevel(logging.INFO)


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

    register_result(exp_label, {'mse': val_metrics['mse'], 'baseline_mse': baseline_mse2, 'config': config})

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

    register_result(exp_label, {'mse': val_metrics['mse'], 'baseline_mse': baseline_mse, 'config': config})

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



def _worker_init():
    """Disable tqdm in worker processes.

    os.environ["TQDM_DISABLE"] doesn't work here because unpickling this
    function triggers the import of this module (and tqdm) before the
    initializer body runs. Patching the class works because it fires at
    instance-creation time, after all imports are done.
    """
    from functools import partialmethod
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def _run_experiment(spec: dict) -> tuple:
    """Generic worker: loads data for one source, runs one task, returns (label, result)."""
    source = spec['source']
    task = spec['task']
    game_name = spec['game_name']
    player_id = spec['player_id']
    label = spec['label']
    embedding_type = spec['embedding_type']

    game = pyspiel.load_game(game_name)
    game_short_name = game.get_type().short_name
    device = get_device_string()

    # --- load policies and embeddings ---
    if source == 'neupl':
        policies_and_embeddings = make_neupl_policies(
            game_short_name,
            neupl_config=spec['neupl_config'],
            original_num_policies=23,
            num_policies_to_make=spec['N'],
            directory=spec['run_dir'],
            interpolate_prenorm=spec['INTERPOLATE_PRENORM'],
            sampling_mode=spec['NEUPL_SAMPLING_MODE'],
        )
        policies = [p_e[1] for p_e in policies_and_embeddings[player_id]]
        embeddings = [p_e[0].detach().cpu().numpy() for p_e in policies_and_embeddings[player_id]]
    elif source == 'psro':
        ppo_agents = load_ppo_agents_from_psro(
            game_short_name=game_short_name, hidden_size=256, player_id=player_id, shuffle=True)
        if embedding_type == 'identity':
            policies, embeddings = get_policies_and_embeddings2(
                game, player_id, ppo_agents, "psro_" + game_short_name, game_short_name, device)
        else:
            policies, embeddings = get_policies_and_embeddings(
                game, player_id, ppo_agents, "psro_" + game_short_name, game_short_name, device)
    elif source == 'random':
        info_state_size = game.information_state_tensor_shape()
        num_actions = game.num_distinct_actions()
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)
        ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, 256)
                      for _ in range(spec['N_random'])]
        if embedding_type == 'identity':
            policies, embeddings = get_policies_and_embeddings2(
                game, player_id, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
        else:
            policies, embeddings = get_policies_and_embeddings(
                game, player_id, ppo_agents, "ppo random " + game_short_name, game_short_name, device)
    else:
        raise ValueError(f"Unknown source: {source}")

    # --- run task ---
    experiment_info = ExperimentInfo(_label_string=label, embedding_type=embedding_type)
    if task == 'a':
        config = TaskAConfig(
            model_config=ModelConfig(model_type=spec['predictor_type']),
            validation_split=0.2,
        )
        result = run_task_a(game=game, policies=policies, embeddings=embeddings,
                            config=config, experiment_info=experiment_info, device=device)
    elif task == 'd':
        config = TaskDConfig(
            model_config=ModelConfig(model_type=spec['predictor_type']),
            player_id=player_id,
            validation_split=0.2,
        )
        result = run_task_d(game=game, policies=policies, embeddings=embeddings,
                            config=config, experiment_info=experiment_info, device=device)
    elif task == 'e':
        config = TaskEConfig(
            model_config=ModelConfig(optimizer_type=spec.get('optimizer_type', 'adamw')),
            player_id=player_id,
            validation_split=0.2,
            epochs=spec.get('epochs', 10),
            num_steps_per_policy_per_epoch=spec.get('num_steps_per_policy_per_epoch', 40),
            num_trajectories_per_policy_per_epoch=spec.get('num_trajectories_per_policy_per_epoch', 7),
            compare_to_control=True,
        )
        result = run_task_e(game=game, policies=policies, embeddings=embeddings,
                            config=config, experiment_info=experiment_info, device=device)
    else:
        raise ValueError(f"Unknown task: {task}")

    return label, result


LEDUC_RUNS_TO_LOAD = {
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

    RUN_TASK_A = False
    RUN_TASK_B = False
    RUN_TASK_C = False
    RUN_TASK_D = False
    RUN_TASK_E = True

    RUN_NEUPL = True
    RUN_PSRO = False
    RUN_RANDOM = False

    NUM_SEEDS = 1

    MAX_WORKERS = 3  # tune to available CPUs

    # Set up file + stream logging only in the main process
    run_start_time = datetime.now()
    Path("logs").mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler('logs/all_downstream_tasks.log', mode='w')
    _fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_sh)

    game_name = "kuhn_poker"
    # game_name = "leduc_poker"
    device = get_device_string()
    print("Using device:", device)

    N = 1000
    NEUPL_SAMPLING_MODE = "gaussian"
    INTERPOLATE_PRENORM = True
    PREDICTOR_TYPE = 'random_forest'

    # Pre-resolve NEUPL directories in the main process so workers never call input().
    # For leduc_poker, directories are hardcoded per (use_randall_loss, seed_num).
    # For other games, prompt once per use_randall_loss and reuse across all seeds.
    neupl_run_dirs = {}  # (use_randall_loss, seed_num) -> dir_name str
    if RUN_NEUPL:
        if game_name == "leduc_poker":
            for rl in [True, False]:
                for s in range(3):
                    neupl_run_dirs[(rl, s)] = LEDUC_RUNS_TO_LOAD[rl][s]
        else:
            for use_randall_loss in [True, False]:
                chosen = select_neupl_directory(game_name, use_randall_loss, hidden_size=256)
                for s in range(3):
                    neupl_run_dirs[(use_randall_loss, s)] = chosen

    # Build one spec dict per individual experiment (one run_task_* call)
    game_short_name = pyspiel.load_game(game_name).get_type().short_name
    specs = []
    for seed_num in range(3):
        if RUN_NEUPL:
            for use_randall_loss in [True, False]:
                run_dir = neupl_run_dirs[(use_randall_loss, seed_num)]
                neupl_config = {'use_randall_loss': use_randall_loss,
                                'hidden_size': 256, 'policy_embedding_size': 64}
                for player_id in range(2):
                    base = dict(source='neupl', game_name=game_name, player_id=player_id,
                                embedding_type='neupl', run_dir=run_dir, neupl_config=neupl_config,
                                N=N, NEUPL_SAMPLING_MODE=NEUPL_SAMPLING_MODE,
                                INTERPOLATE_PRENORM=INTERPOLATE_PRENORM)
                    if RUN_TASK_A:
                        specs.append({**base, 'task': 'a', 'predictor_type': PREDICTOR_TYPE,
                                      'label': f'{game_short_name} neupl({INTERPOLATE_PRENORM})({NEUPL_SAMPLING_MODE[:1]}) p{player_id} randloss={use_randall_loss} N={N} Task A'})
                    if RUN_TASK_D:
                        specs.append({**base, 'task': 'd', 'predictor_type': PREDICTOR_TYPE,
                                      'label': f'{game_short_name} neupl p{player_id} randloss={use_randall_loss} N={N} Task D'})
                    if RUN_TASK_E:
                        specs.append({**base, 'task': 'e',
                                      'label': f'{game_short_name} neupl p{player_id} randloss={use_randall_loss} N={N} Task E'})
        if RUN_PSRO:
            for player_id in range(2):
                for emb_type in ['reconstruction-autoencoder', 'identity']:
                    base = dict(source='psro', game_name=game_name, player_id=player_id,
                                embedding_type=emb_type)
                    if RUN_TASK_A:
                        specs.append({**base, 'task': 'a', 'predictor_type': 'linear',
                                      'label': f'{game_short_name} psro {player_id} {emb_type} Task A'})
                    if RUN_TASK_D:
                        specs.append({**base, 'task': 'd', 'predictor_type': 'linear',
                                      'label': f'{game_short_name} psro {player_id} {emb_type} Task D'})
                    if RUN_TASK_E:
                        specs.append({**base, 'task': 'e',
                                      'label': f'{game_short_name} psro {player_id} {emb_type} Task E'})
        if RUN_RANDOM:
            for emb_type in ['reconstruction-autoencoder', 'identity']:
                base = dict(source='random', game_name=game_name, player_id=0,
                            embedding_type=emb_type, N_random=1000)
                if RUN_TASK_A:
                    specs.append({**base, 'task': 'a', 'predictor_type': 'linear',
                                  'label': f'{game_short_name} ppo random 0 {emb_type} Task A'})
                if RUN_TASK_D:
                    specs.append({**base, 'task': 'd', 'predictor_type': 'linear',
                                  'label': f'{game_short_name} ppo random 0 {emb_type} Task D'})

    logger.info(f"Submitting {len(specs)} jobs with max_workers={MAX_WORKERS}")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init) as executor:
        future_to_label = {executor.submit(_run_experiment, spec): spec['label'] for spec in specs}
        for future in tqdm(as_completed(future_to_label), total=len(future_to_label), desc="Jobs"):
            label = future_to_label[future]
            try:
                label, result = future.result()
                register_result(label, result, save=True)
                logger.info(f"Completed: {label}")
            except Exception as exc:
                logger.error(f"Worker failed [{label}]: {exc}")

    save_results()