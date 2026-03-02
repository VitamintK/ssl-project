"""
Unified task functions with standardized interfaces.

All tasks follow the same pattern:
1. Accept pre-generated policies and embeddings
2. Accept TaskConfig object for settings
3. Always register results
4. Return consistent dict with metrics, history, predictor

Benefits:
- Consistent interface across all tasks
- Separation of concerns (agent generation in main.py, prediction here)
- Type-safe configuration
- All tasks register results
"""

import numpy as np
import logging
from typing import List
from open_spiel.python.policy import Policy, UniformRandomPolicy

from config import TaskAConfig, TaskBConfig, TaskCConfig, TaskDConfig, config_to_dict
from downstream import (
    PayoffPredictorRefactored,
    StatePayoffPredictorRefactored,
    ExploitabilityPredictorRefactored
)


# Use the existing logger from main.py
logger = logging.getLogger(__name__)


def run_task_a(
    game,
    policies: List[Policy],
    embeddings: List[np.ndarray],
    config: TaskAConfig,
    exp_label: str,
    device: str = "cpu"
) -> dict:
    """
    Task A: Predict payoff of agents vs fixed opponent (uniform random).

    Args:
        game: OpenSpiel game instance
        policies: Pre-generated P1 policies
        embeddings: Pre-computed P1 embeddings
        config: Task A configuration (model type, validation split, etc.)
        exp_label: Experiment label for result registration
        device: Device for computation

    Returns:
        dict with keys:
            - predictor: Trained predictor instance
            - history: Training history
            - val_metrics: Validation metrics (mse, mae, baseline_mse)
            - train_metrics: Training metrics
            - config: Configuration used

    Example:
        >>> from config import TaskAConfig, ModelConfig
        >>> config = TaskAConfig(
        ...     model_config=ModelConfig(model_type="random_forest"),
        ...     validation_split=0.2
        ... )
        >>> results = run_task_a(game, policies, embeddings, config, "exp1", "cpu")
        >>> print(f"Val MSE: {results['val_metrics']['mse']:.6f}")
    """
    game_short_name = game.get_type().short_name

    logger.info(f"Running Task A: {exp_label}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of policies: {len(policies)}")
    logger.info(f"Embedding dimension: {embeddings[0].shape[0] if len(embeddings) > 0 else 'N/A'}")

    # Create opponent (uniform random)
    opponent_policy = UniformRandomPolicy(game)

    # Create predictor
    predictor = PayoffPredictorRefactored(
        game=game,
        p1_policies=policies,
        p2_policies=[opponent_policy],
        p1_embeddings=embeddings,
        p2_embeddings=[np.array([0])],  # Dummy embedding for fixed opponent
        model_config=config.model_config,
        device=device
    )

    # Compute ground truth
    logger.info("Computing ground truth payoffs...")
    predictor.compute_ground_truth_payoffs()

    # Train with agent-level splitting
    logger.info(f"Training {config.model_config.model_type} predictor...")
    history = predictor.train_with_agent_level_split(config.validation_split)

    # Evaluate
    logger.info("Evaluating on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    train_metrics = predictor.evaluate(eval_set="train")

    # Log results
    logger.info(f"Results for {exp_label}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    # ALWAYS register results (fixes inconsistent result tracking)
    from main import register_result  # Import here to avoid circular dependency
    register_result(
        experiment_label=exp_label,
        config_dict=config_to_dict(config),
        mse=val_metrics['mse'],
        baseline_mse=val_metrics['baseline_mse']
    )

    return {
        'predictor': predictor,
        'history': history,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config
    }


def run_task_b(
    game,
    p1_policies: List[Policy],
    p1_embeddings: List[np.ndarray],
    p2_policies: List[Policy],
    p2_embeddings: List[np.ndarray],
    config: TaskBConfig,
    exp_label: str,
    device: str = "cpu"
) -> dict:
    """
    Task B: Predict payoff for agent vs agent matchups.

    This task evaluates how well we can predict expected payoffs for matchups
    between two variable agents (not against a fixed opponent).

    Args:
        game: OpenSpiel game instance
        p1_policies: Pre-generated P1 policies
        p1_embeddings: Pre-computed P1 embeddings
        p2_policies: Pre-generated P2 policies
        p2_embeddings: Pre-computed P2 embeddings
        config: Task B configuration (model type, validation split, etc.)
        exp_label: Experiment label for result registration
        device: Device for computation

    Returns:
        dict with keys:
            - predictor: Trained predictor instance
            - history: Training history
            - val_metrics: Validation metrics (mse, mae, baseline_mse)
            - train_metrics: Training metrics
            - config: Configuration used

    Example:
        >>> from config import TaskBConfig, ModelConfig
        >>> config = TaskBConfig(
        ...     model_config=ModelConfig(model_type="random_forest"),
        ...     validation_split=0.2
        ... )
        >>> results = run_task_b(game, p1_policies, p1_emb, p2_policies, p2_emb,
        ...                      config, "exp1", "cpu")
        >>> print(f"Val MSE: {results['val_metrics']['mse']:.6f}")
    """
    logger.info(f"Running Task B: {exp_label}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of P1 policies: {len(p1_policies)}")
    logger.info(f"Number of P2 policies: {len(p2_policies)}")
    logger.info(f"P1 embedding dimension: {p1_embeddings[0].shape[0] if len(p1_embeddings) > 0 else 'N/A'}")
    logger.info(f"P2 embedding dimension: {p2_embeddings[0].shape[0] if len(p2_embeddings) > 0 else 'N/A'}")

    # Create predictor
    predictor = PayoffPredictorRefactored(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=p1_embeddings,
        p2_embeddings=p2_embeddings,
        model_config=config.model_config,
        device=device
    )

    # Compute ground truth
    logger.info("Computing ground truth payoffs...")
    predictor.compute_ground_truth_payoffs()

    # Train with agent-level splitting
    logger.info(f"Training {config.model_config.model_type} predictor...")
    history = predictor.train_with_agent_level_split(config.validation_split)

    # Evaluate
    logger.info("Evaluating on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    train_metrics = predictor.evaluate(eval_set="train")

    # Log results
    logger.info(f"Results for {exp_label}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    # ALWAYS register results
    from main import register_result
    register_result(
        experiment_label=exp_label,
        config_dict=config_to_dict(config),
        mse=val_metrics['mse'],
        baseline_mse=val_metrics['baseline_mse']
    )

    return {
        'predictor': predictor,
        'history': history,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config
    }


def run_task_c(
    game,
    p1_policies: List[Policy],
    p1_embeddings: List[np.ndarray],
    p2_policies: List[Policy],
    p2_embeddings: List[np.ndarray],
    config: TaskCConfig,
    exp_label: str,
    device: str = "cpu"
) -> dict:
    """
    Task C: State-conditioned payoff prediction.

    This task evaluates how well we can predict expected payoffs conditioned on
    the current game state (in addition to the agents playing).

    Args:
        game: OpenSpiel game instance
        p1_policies: Pre-generated P1 policies
        p1_embeddings: Pre-computed P1 embeddings
        p2_policies: Pre-generated P2 policies
        p2_embeddings: Pre-computed P2 embeddings
        config: Task C configuration (model type, validation split, num_states, etc.)
        exp_label: Experiment label for result registration
        device: Device for computation

    Returns:
        dict with keys:
            - predictor: Trained predictor instance
            - history: Training history
            - val_metrics: Validation metrics (mse, mae, baseline_mse)
            - train_metrics: Training metrics
            - config: Configuration used

    Example:
        >>> from config import TaskCConfig, ModelConfig
        >>> config = TaskCConfig(
        ...     model_config=ModelConfig(model_type="mlp"),
        ...     num_states=20,
        ...     max_state_depth=5,
        ...     validation_split=0.2
        ... )
        >>> results = run_task_c(game, p1_policies, p1_emb, p2_policies, p2_emb,
        ...                      config, "exp1", "cpu")
        >>> print(f"Val MSE: {results['val_metrics']['mse']:.6f}")
    """
    logger.info(f"Running Task C: {exp_label}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of P1 policies: {len(p1_policies)}")
    logger.info(f"Number of P2 policies: {len(p2_policies)}")
    logger.info(f"Number of states to sample: {config.num_states}")
    logger.info(f"Max state depth: {config.max_state_depth}")

    # Create predictor
    predictor = StatePayoffPredictorRefactored(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=p1_embeddings,
        p2_embeddings=p2_embeddings,
        model_config=config.model_config,
        num_states=config.num_states,
        max_depth=config.max_state_depth,
        device=device
    )

    # Compute ground truth
    logger.info("Computing ground truth payoffs for sampled states...")
    predictor.compute_ground_truth_payoffs()

    # Train with agent-level splitting
    logger.info(f"Training {config.model_config.model_type} predictor...")
    history = predictor.train_with_agent_level_split(config.validation_split)

    # Evaluate
    logger.info("Evaluating on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    train_metrics = predictor.evaluate(eval_set="train")

    # Log results
    logger.info(f"Results for {exp_label}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    # ALWAYS register results
    from main import register_result
    register_result(
        experiment_label=exp_label,
        config_dict=config_to_dict(config),
        mse=val_metrics['mse'],
        baseline_mse=val_metrics['baseline_mse']
    )

    return {
        'predictor': predictor,
        'history': history,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config
    }


def run_task_d(
    game,
    policies: List[Policy],
    embeddings: List[np.ndarray],
    config: TaskDConfig,
    exp_label: str,
    device: str = "cpu"
) -> dict:
    """
    Task D: Exploitability prediction.

    This task evaluates how well we can predict a policy's exploitability
    (the best response value against it).

    Args:
        game: OpenSpiel game instance
        policies: Pre-generated policies
        embeddings: Pre-computed embeddings
        config: Task D configuration (model type, validation split, player_id, etc.)
        exp_label: Experiment label for result registration
        device: Device for computation

    Returns:
        dict with keys:
            - predictor: Trained predictor instance
            - history: Training history
            - val_metrics: Validation metrics (mse, mae, baseline_mse)
            - train_metrics: Training metrics
            - config: Configuration used

    Example:
        >>> from config import TaskDConfig, ModelConfig
        >>> config = TaskDConfig(
        ...     model_config=ModelConfig(model_type="random_forest"),
        ...     player_id=0,
        ...     validation_split=0.2
        ... )
        >>> results = run_task_d(game, policies, embeddings, config, "exp1", "cpu")
        >>> print(f"Val MSE: {results['val_metrics']['mse']:.6f}")
    """
    logger.info(f"Running Task D: {exp_label}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of policies: {len(policies)}")
    logger.info(f"Player ID: {config.player_id}")
    logger.info(f"Embedding dimension: {embeddings[0].shape[0] if len(embeddings) > 0 else 'N/A'}")

    # Create predictor
    predictor = ExploitabilityPredictorRefactored(
        game=game,
        policies=policies,
        embeddings=embeddings,
        model_config=config.model_config,
        player_id=config.player_id,
        device=device
    )

    # Compute ground truth
    logger.info("Computing ground truth exploitability values...")
    predictor.compute_ground_truth_payoffs()

    # Train with agent-level splitting
    logger.info(f"Training {config.model_config.model_type} predictor...")
    history = predictor.train_with_agent_level_split(config.validation_split)

    # Evaluate
    logger.info("Evaluating on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    train_metrics = predictor.evaluate(eval_set="train")

    # Log results
    logger.info(f"Results for {exp_label}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    # ALWAYS register results
    from main import register_result
    register_result(
        experiment_label=exp_label,
        config_dict=config_to_dict(config),
        mse=val_metrics['mse'],
        baseline_mse=val_metrics['baseline_mse']
    )

    return {
        'predictor': predictor,
        'history': history,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config
    }
