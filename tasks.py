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
from omegaconf import OmegaConf
from open_spiel.python.policy import Policy, UniformRandomPolicy

from config import TaskAConfig, TaskBConfig, TaskCConfig, TaskDConfig, TaskEConfig, ExperimentInfo, config_to_dict
from downstream import (
    BestResponseLearner,
    PayoffPredictor,
    StatePayoffPredictor,
    ExploitabilityPredictor
)


logger = logging.getLogger("ssl_project")


def run_task_a(
    game,
    policies: List[Policy],
    embeddings: List[np.ndarray],
    config: TaskAConfig,
    experiment_info: ExperimentInfo,
    device: str = "cpu"
) -> dict:
    """
    Task A: Predict payoff of agents vs fixed opponent (uniform random).

    Args:
        game: OpenSpiel game instance
        policies: Pre-generated P1 policies
        embeddings: Pre-computed P1 embeddings
        config: Task A configuration (model type, validation split, etc.)
        experiment_info: Experiment info for labeling and result registration
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

    logger.info(f"Running Task A: {experiment_info.label_string}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of policies: {len(policies)}")
    logger.info(f"Embedding dimension: {embeddings[0].shape[0] if len(embeddings) > 0 else 'N/A'}")

    # Create opponent (uniform random)
    opponent_policy = UniformRandomPolicy(game)

    # Create predictor
    predictor = PayoffPredictor(
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
    logger.info(f"Results for {experiment_info.label_string}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    return {
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config_to_dict(config),
    }


def run_task_b(
    game,
    p1_policies: List[Policy],
    p1_embeddings: List[np.ndarray],
    p2_policies: List[Policy],
    p2_embeddings: List[np.ndarray],
    config: TaskBConfig,
    experiment_info: ExperimentInfo,
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
        experiment_info: Experiment info for labeling and result registration
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
    logger.info(f"Running Task B: {experiment_info.label_string}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of P1 policies: {len(p1_policies)}")
    logger.info(f"Number of P2 policies: {len(p2_policies)}")
    logger.info(f"P1 embedding dimension: {p1_embeddings[0].shape[0] if len(p1_embeddings) > 0 else 'N/A'}")
    logger.info(f"P2 embedding dimension: {p2_embeddings[0].shape[0] if len(p2_embeddings) > 0 else 'N/A'}")

    # Create predictor
    predictor = PayoffPredictor(
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
    logger.info(f"Results for {experiment_info.label_string}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    return {
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config_to_dict(config),
    }


def run_task_c(
    game,
    p1_policies: List[Policy],
    p1_embeddings: List[np.ndarray],
    p2_policies: List[Policy],
    p2_embeddings: List[np.ndarray],
    config: TaskCConfig,
    experiment_info: ExperimentInfo,
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
        experiment_info: Experiment info for labeling and result registration
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
    logger.info(f"Running Task C: {experiment_info.label_string}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of P1 policies: {len(p1_policies)}")
    logger.info(f"Number of P2 policies: {len(p2_policies)}")
    logger.info(f"Number of states to sample: {config.num_states}")
    logger.info(f"Max state depth: {config.max_state_depth}")

    # Create predictor
    predictor = StatePayoffPredictor(
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
    logger.info(f"Results for {experiment_info.label_string}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    return {
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config_to_dict(config),
    }


def run_task_d(
    game,
    policies: List[Policy],
    embeddings: List[np.ndarray],
    config: TaskDConfig,
    experiment_info: ExperimentInfo,
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
        experiment_info: Experiment info for labeling and result registration
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
    logger.info(f"Running Task D: {experiment_info.label_string}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of policies: {len(policies)}")
    logger.info(f"Player ID: {config.player_id}")
    logger.info(f"Embedding dimension: {embeddings[0].shape[0] if len(embeddings) > 0 else 'N/A'}")

    # Create predictor
    predictor = ExploitabilityPredictor(
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
    logger.info(f"Results for {experiment_info.label_string}:")
    logger.info(f"  Validation MSE: {val_metrics['mse']:.6f}")
    logger.info(f"  Baseline MSE: {val_metrics['baseline_mse']:.6f}")
    improvement = (1 - val_metrics['mse'] / val_metrics['baseline_mse']) * 100
    logger.info(f"  Improvement over baseline: {improvement:.2f}%")
    logger.info(f"  Training MSE: {train_metrics['mse']:.6f}")

    return {
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'config': config_to_dict(config),
    }

def run_task_e(
    game,
    policies: List[Policy],
    embeddings: List[np.ndarray],
    config: TaskEConfig,
    experiment_info: ExperimentInfo,
    device: str = "cpu"
) -> dict:
    logger.info(f"Running Task E: {experiment_info.label_string}")
    logger.info(f"Model type: {config.model_config.model_type}")
    logger.info(f"Number of policies: {len(policies)}")
    logger.info(f"Player ID: {config.player_id}")
    logger.info(f"Embedding dimension: {embeddings[0].shape[0] if len(embeddings) > 0 else 'N/A'}")
    EVAL_SET = 'val'

    # Create predictor
    # TODO: actually use hydra or don't
    if game.get_type().short_name == 'kuhn_poker':
        config_path = 'configs/ppo_kuhn_poker.yaml'
    elif game.get_type().short_name == 'leduc_poker':
        config_path = 'configs/ppo_leduc_poker.yaml'
    else:
        raise ValueError(f"Unknown game: {game.get_type().short_name}")
    algorithm_config = OmegaConf.load(config_path)
    args = OmegaConf.load('configs/experiment.yaml')
    args.algorithm = algorithm_config
    # policies, embeddings = policies[:3], embeddings[:3] # TODO: remove this
    predictor = BestResponseLearner(
        game=game,
        policies=policies,
        embeddings=embeddings,
        policy_player_id=config.player_id,
        config=args,
    )

    predictor.train_best_responder(
        optimizer_type=config.model_config.optimizer_type,
        epochs=config.epochs,
        max_batch_size=config.max_batch_size,
        num_trajectories_per_policy_per_epoch=config.num_trajectories_per_policy_per_epoch,
        experiment_info=experiment_info,
    )

    # Evaluate
    logger.info("Evaluating on validation set...")
    predictor_metrics = predictor.evaluate(eval_set=EVAL_SET, num_episodes_per_policy=400)

    # result = {
    #     'predictor': predictor,
    #     'predictor_metrics': predictor_metrics,
    #     'config': config,
    # }

    # Control comparison: train with shuffled embeddings
    if config.compare_to_control:
        logger.info("Running control comparison with shuffled embeddings...")

        # Shuffle embeddings (break the correspondence between policies and embeddings)
        # shuffled_indices = np.random.permutation(len(embeddings))
        # shuffled_embeddings = [embeddings[i] for i in shuffled_indices]

        # control_predictor = BestResponseLearner(
        #     game=game,
        #     policies=policies,
        #     embeddings=shuffled_embeddings,
        #     policy_player_id=config.player_id,
        #     config=args,
        # )
        # a bit ad hoc but it's ok!
        predictor.val_embeddings = predictor.val_embeddings[np.random.permutation(len(predictor.val_embeddings))]
        predictor.train_embeddings = predictor.train_embeddings[np.random.permutation(len(predictor.train_embeddings))]

        predictor.train_best_responder(
            epochs=config.epochs,
            max_batch_size=config.max_batch_size,
            num_trajectories_per_policy_per_epoch=config.num_trajectories_per_policy_per_epoch,
            experiment_info=experiment_info,
        )

        logger.info("Evaluating control (shuffled embeddings) on validation set...")
        control_metrics = predictor.evaluate(eval_set=EVAL_SET, num_episodes_per_policy=500)

        logger.info(f"Control comparison results:")
        logger.info(f"  Original avg empirical payoff: {predictor_metrics['avg_empirical_payoff']:.4f}")
        logger.info(f"  Control avg empirical payoff:  {control_metrics['avg_empirical_payoff']:.4f}")
        logger.info(f"  Original avg exact exploitability: {predictor_metrics['avg_exact_exploitability']:.4f}")
        logger.info(f"  Control avg exact exploitability:  {control_metrics['avg_exact_exploitability']:.4f}")
    result = {
        'val_metrics': predictor_metrics,
        'config': config_to_dict(config),
        'omega_conf': OmegaConf.to_container(args),
    }
    if config.compare_to_control:
        result['control_metrics'] = control_metrics
    return result
