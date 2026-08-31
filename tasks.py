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

import os
import numpy as np
import logging
from typing import List
from omegaconf import OmegaConf
from open_spiel.python.policy import Policy, UniformRandomPolicy

from config import TaskAConfig, TaskBConfig, TaskCConfig, TaskDConfig, TaskEConfig, TaskFConfig, ExperimentInfo, config_to_dict
from downstream import (
    BestResponseLearner,
    PayoffPredictor,
    StatePayoffPredictor,
    ExploitabilityPredictor,
    EmbeddingEquilibriumSolver,
    compute_best_response_value,
    compute_nash_conv,
    matrix_game_baseline,
    plot_solve_value_curves,
    plot_task_f_combined,
)
from utils import get_expected_payoffs


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
    import os
    from datetime import datetime
    import random as _random
    _ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _suffix = _random.randint(10, 99)
    plot_dir = os.path.join('results', 'training_metrics', f'{_ts}_{_suffix}')

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
        device=device,
    )

    predictor.train_best_responder(
        optimizer_type=config.model_config.optimizer_type,
        epochs=config.epochs,
        max_batch_size=config.max_batch_size,
        num_trajectories_per_policy_per_epoch=config.num_trajectories_per_policy_per_epoch,
        experiment_info=experiment_info,
        is_control=False,
        plot_dir=plot_dir,
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
            is_control=True,
            plot_dir=plot_dir,
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


def _value_function_fingerprint(p1_embeddings, p2_embeddings, config: TaskFConfig, game) -> str:
    """Stable hash of the inputs that determine the trained value function.

    The checkpoint identity is carried by the cache filename (experiment label + checkpoint
    id), so this only guards the value-function *configuration* and the embedding dimensionality
    (which sets the model's input size). It deliberately ignores the sampled embedding
    values: NeuPL pools are re-drawn from the same checkpoint each run, and a value
    function over embedding space need not be retrained just because a fresh sample of
    points was drawn from the same distribution.
    """
    import hashlib
    h = hashlib.sha256()
    d1 = np.array(p1_embeddings).shape[1]
    d2 = np.array(p2_embeddings).shape[1]
    mc = config.value_function_model_config
    key = (str(getattr(game, "get_type", lambda: game)()), d1, d2, mc.model_type,
           tuple(mc.hidden_dims or ()), mc.dropout, mc.num_epochs, mc.learning_rate,
           config.value_function_num_pairs, config.value_function_validation_split,
           config.exact_payoff_targets)
    h.update(repr(key).encode())
    return h.hexdigest()


def _log_pool_exploitability(game, p1_policies, p2_policies):
    """Log exact-exploitability (best-response value) stats over each player's pool policies.

    For each pool, computes BRV against every sampled policy and reports min / quartiles /
    max / mean. The min is the least-exploitable single pure policy in the pool -- a
    baseline the Task F solve should aim to beat.
    """
    def _stats(policies, cid, name):
        if not policies:
            return
        vals = np.array([compute_best_response_value(game, p, committed_player_id=cid)
                         for p in policies])
        q = np.percentile(vals, [0, 25, 50, 75, 100])
        logger.info(
            "  Pool exploitability (%s, n=%d): min=%.4f  q25=%.4f  median=%.4f  q75=%.4f  "
            "max=%.4f  mean=%.4f", name, len(vals), q[0], q[1], q[2], q[3], q[4], vals.mean())

    _stats(list(p1_policies), 0, "P1")
    _stats(list(p2_policies), 1, "P2")


def _archive_plot(path, run_ts):
    """Copy a saved Task F artifact into a sibling ``archive/`` dir, timestamp-prefixed.

    Keeps a per-run snapshot alongside the latest-overwriting canonical path. Best-effort:
    logs and swallows any error rather than failing the task.
    """
    import shutil
    from pathlib import Path as _Path
    try:
        p = _Path(path)
        adir = p.parent / "archive"
        adir.mkdir(parents=True, exist_ok=True)
        dest = adir / f"{run_ts}_{p.name}"
        shutil.copy2(p, dest)
        return str(dest)
    except Exception as exc:
        logger.warning(f"Could not archive {path}: {exc}")
        return None


def run_task_f(
    game,
    p1_policies, p1_embeddings,
    p2_policies, p2_embeddings,
    p1_decoder, p2_decoder,
    config: TaskFConfig,
    experiment_info: ExperimentInfo,
    device: str = "cpu",
    p1_original_embeddings=None, p2_original_embeddings=None,
    checkpoint_id=None,
) -> dict:
    """Task F: find an equilibrium by descent-ascent in embedding space; report NashConv."""
    logger.info(f"Running Task F: {experiment_info.label_string}")

    # 1. Pretrain the value function V(e_p1, e_p2) (Task B).
    predictor = PayoffPredictor(
        game=game, p1_policies=p1_policies, p2_policies=p2_policies,
        p1_embeddings=p1_embeddings, p2_embeddings=p2_embeddings,
        model_config=config.value_function_model_config, device=device, num_pairs=config.value_function_num_pairs)

    # Optional cache: skip ground-truth payoff computation + training on a fingerprint hit.
    cache_path = None
    fingerprint = None
    loaded_from_cache = False
    val_metrics = None
    if config.value_function_cache_dir:
        import os as _os
        import re as _re
        fingerprint = _value_function_fingerprint(p1_embeddings, p2_embeddings, config, game)
        safe_label = _re.sub(r"[^0-9A-Za-z._-]+", "_", experiment_info.label_string).strip("_") or "task_f"
        # Checkpoint identity in the FILENAME so two different training runs with the same
        # label (e.g. same game/randloss/N but a different NeuPL run_dir) don't collide.
        ckpt_tag = _re.sub(r"[^0-9A-Za-z._-]+", "_",
                           _os.path.basename(str(checkpoint_id))).strip("_") if checkpoint_id else "nockpt"
        cache_path = _os.path.join(config.value_function_cache_dir, f"{safe_label}_{ckpt_tag}_value_fn.pt")
        if config.value_function_cache_ignore and _os.path.exists(cache_path):
            logger.info(f"Ignoring cached value function at {cache_path} (retraining from scratch)")
        if _os.path.exists(cache_path) and not config.value_function_cache_ignore:
            import torch as _torch
            blob = _torch.load(cache_path, map_location=device, weights_only=False)
            if blob.get("fingerprint") == fingerprint:
                predictor.trainer.model.load_state_dict(blob["state_dict"])
                predictor.trainer.model.to(device)
                val_metrics = blob.get("val_metrics")
                loaded_from_cache = True
                logger.info(f"Loaded cached value function from {cache_path}")
            else:
                logger.warning(f"Value-function cache at {cache_path} has a stale "
                               f"fingerprint; retraining.")

    if not loaded_from_cache:
        predictor.compute_ground_truth_payoffs(exact=config.exact_payoff_targets)
        predictor.train_with_agent_level_split(config.value_function_validation_split)
        val_metrics = predictor.evaluate(eval_set="val")
        # Exact exploitability (best-response value) of the sampled pool policies -- a
        # reference baseline for what the solve should beat. Only when training from scratch.
        _log_pool_exploitability(game, p1_policies, p2_policies)
        if cache_path is not None:
            import os as _os
            import torch as _torch
            _os.makedirs(config.value_function_cache_dir, exist_ok=True)
            _torch.save({"state_dict": predictor.trainer.model.state_dict(),
                         "fingerprint": fingerprint,
                         "val_metrics": val_metrics}, cache_path)
            logger.info(f"Saved trained value function to {cache_path}")

    pool_p1 = np.array(p1_embeddings)
    pool_p2 = np.array(p2_embeddings)

    # NeuPL passes its original (pre-sampling) anchor embeddings here. Report the
    # exact opponent best-response value against every anchor for the committed
    # player. NeuPL policy index 0 is uniform random, so these anchors retain their
    # native policy indices starting at 1.
    committed_player_id = 0 if config.optimizing_player == "p1" else 1
    committed_decoder = p1_decoder if committed_player_id == 0 else p2_decoder
    committed_originals = (
        p1_original_embeddings if committed_player_id == 0 else p2_original_embeddings
    )
    if committed_originals is not None:
        committed_player_label = "p1" if committed_player_id == 0 else "p2"
        original_policy_brvs = [
            compute_best_response_value(
                game, committed_decoder(embedding),
                committed_player_id=committed_player_id,
            )
            for embedding in committed_originals
        ]
        logger.info(
            "Task F NeuPL BRV against each original player_id=%s (%s) policy (N=%s): %s",
            committed_player_id,
            committed_player_label,
            len(original_policy_brvs),
            ", ".join(
                f"policy {policy_index}={brv:.6f}"
                for policy_index, brv in enumerate(original_policy_brvs, start=1)
            ),
        )

    # Restricted-matrix-game baseline (computed before the solve): over the sampled pool, pick
    # the committed-player policy the value function deems least exploitable, then report its
    # exact ground-truth BRV -- the bar the ascent-descent solve needs to beat.
    try:
        matrix_baseline = matrix_game_baseline(
            game, predictor.trainer.model, p1_policies, p2_policies, pool_p1, pool_p2,
            committed_player_id=committed_player_id, device=device)
        logger.info(
            "Task F matrix-game baseline (committed=%s): pool policy #%d is least exploitable "
            "by the value function (predicted BR value=%.6f); its exact ground-truth BRV=%.6f",
            matrix_baseline["committed_player"], matrix_baseline["selected_pool_index"],
            matrix_baseline["predicted_exploitability"], matrix_baseline["ground_truth_brv"])
    except Exception as exc:  # best-effort; never fail the task on the baseline
        logger.warning(f"Could not compute Task F matrix-game baseline: {exc}")
        matrix_baseline = None

    # 2. Solve for an equilibrium, keeping the restart with lowest decoded NashConv.
    solver = EmbeddingEquilibriumSolver(
        game, predictor.trainer.model, pool_p1, pool_p2, p1_decoder, p2_decoder, config, device=device)

    # Live web viewer: stream each solve step's projected embeddings to figures/task_f/live/.
    if getattr(config, "live_view", False):
        live_dir = os.path.join("figures", "task_f", "live")
        solver.setup_live(live_dir, originals_p1=p1_original_embeddings,
                          originals_p2=p2_original_embeddings)
        logger.info(f"Live viewer streaming to {live_dir}/ (run: uv run python3 serve_live.py)")

    def score(res):
        return compute_nash_conv(game, p1_decoder(res.e_p1), p2_decoder(res.e_p2))

    # Online posttraining co-trains the value model during each solve, anchored on the
    # pretraining pool rows, with the model reset to its pretrained state per restart. On a
    # value-function cache hit the ground truth was skipped, so compute it now (posttrain
    # needs the anchor rows).
    X_pre = y_pre = None
    if config.posttrain:
        if predictor.ground_truth_payoffs is None:
            predictor.compute_ground_truth_payoffs(exact=config.exact_payoff_targets)
        X_pre = predictor._prepare_training_data()
        y_pre = predictor.ground_truth_payoffs

    result = solver.solve_best_of_restarts(
        score, X_pre=X_pre, y_pre=y_pre, exact=config.exact_payoff_targets)

    # Dump the raw per-step stats to JSON, and render one combined figure: one row per
    # restart with [solve curves | P1 exploitability landscape | P2 path]. When the
    # landscape is disabled, fall back to just the solve-curves figure.
    plot_path = None
    solve_stats_path = None
    # Shared per-run timestamp so all archived artifacts from this run group together.
    from datetime import datetime as _datetime
    run_ts = _datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        import re as _re
        import json as _json
        from pathlib import Path as _Path
        label = experiment_info.label_string
        safe_label = _re.sub(r"[^0-9A-Za-z._-]+", "_", label).strip("_") or "task_f"
        out_dir = _Path("figures") / "task_f"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Detect whether the source checkpoint is an APSRO run (its config.json carries an
        # 'apsro' flag, written during NeuPL/APSRO training). None = couldn't determine.
        _ckpt_apsro = None
        _ckpt_meta = {}
        if checkpoint_id:
            _ckpt_cfg = _Path(str(checkpoint_id)) / "config.json"
            if _ckpt_cfg.exists():
                try:
                    with open(_ckpt_cfg) as _cf:
                        _ckpt_meta = _json.load(_cf)
                    _ckpt_apsro = bool(_ckpt_meta.get("apsro", False))
                except Exception:
                    pass
        _ckpt_info = {
            "id": str(checkpoint_id) if checkpoint_id else None,
            "apsro": _ckpt_apsro,
            "apsro_exploited_player": _ckpt_meta.get("exploited_player"),
            "apsro_bandit": _ckpt_meta.get("bandit"),
        }

        # Serialize the full Task F config (nested dataclasses -> dicts) alongside the stats.
        import dataclasses as _dataclasses
        _config_dict = (_dataclasses.asdict(config)
                        if _dataclasses.is_dataclass(config) else vars(config))
        solve_stats_path = str(out_dir / f"{safe_label}_solve_stats.json")
        with open(solve_stats_path, "w") as _f:
            _json.dump({"label": label, "checkpoint": _ckpt_info, "config": _config_dict,
                        "matrix_game_baseline": matrix_baseline,
                        "restarts": solver.last_restart_stats}, _f,
                       indent=2, default=float)
        logger.info(f"Saved solve stats to {solve_stats_path}")
        _archive_plot(solve_stats_path, run_ts)

        # One-line summary of the optimization scheme, shown under the figure title.
        _sum_parts = [
            f"optimizer={config.optimizer}",
            f"lr(exploited/exploiter)={config.lr_exploited:g}/{config.lr_exploiter:g}",
        ]
        if config.optimizer in ("cem", "mppi"):
            _sum_parts.append(
                f"noise(exploited/exploiter)={config.cem_noise_exploited:g}/{config.cem_noise_exploiter:g}")
        if config.optimizer == "mppi":
            _sum_parts.append(f"mppi_temp={config.mppi_temperature:g}")
        _sum_parts.append(
            f"trust_region=on (q={config.trust_region_quantile:g}, scale={config.trust_region_scale:g})"
            if config.trust_region else "trust_region=off")
        _sum_parts.append(
            f"posttrain=on (lr={config.posttrain_lr:g}, anchor={config.posttrain_anchor_batch})"
            if config.posttrain else "posttrain=off")
        if _ckpt_apsro:
            _sum_parts.append(
                f"checkpoint=APSRO(exploited={_ckpt_meta.get('exploited_player')},"
                f"bandit={_ckpt_meta.get('bandit')})")
        elif _ckpt_apsro is False:
            _sum_parts.append("checkpoint=non-APSRO")
        else:
            _sum_parts.append("checkpoint=unknown")
        _config_summary = " | ".join(_sum_parts)

        # The committed (optimizing) player is whose exploitability is plotted.
        _cid = 0 if config.optimizing_player == "p1" else 1
        if config.plot_landscape:
            plot_path = str(out_dir / f"{safe_label}_task_f.png")
            # Kuhn poker: cap the shared exploitability color scale at 0.45.
            _vmax = 0.45 if game.get_type().short_name == "kuhn_poker" else None
            _cdec = p1_decoder if _cid == 0 else p2_decoder
            _cpool = pool_p1 if _cid == 0 else pool_p2
            _corig = p1_original_embeddings if _cid == 0 else p2_original_embeddings
            plot_task_f_combined(
                solver.last_restart_stats, solver.last_restart_results,
                game, _cdec, plot_path,
                dim_selection=config.landscape_dim_selection,
                grid_size=config.landscape_grid_size,
                committed_player_id=_cid, committed_pool=_cpool,
                committed_originals=_corig, landscape_vmax=_vmax,
                title=f"Task F — {label}", subtitle=_config_summary,
                # committed (outer) player uses the "exploited" sampling noise; only CEM/MPPI sample.
                sample_noise=(config.cem_noise_exploited
                              if config.optimizer in ("cem", "mppi") else None))
        else:
            plot_path = str(out_dir / f"{safe_label}_solve_curves.png")
            plot_solve_value_curves(
                solver.last_restart_stats, plot_path,
                title=f"Task F value curves — {label}",
                committed_player_id=_cid, subtitle=_config_summary)
        logger.info(f"Saved Task F figure to {plot_path}")
        _archive_plot(plot_path, run_ts)
    except Exception as exc:  # plotting/dumping is best-effort; never fail the task on it
        logger.warning(f"Could not save Task F figure/stats: {exc}")

    # 3. Evaluate the recovered profile.
    p1_star = p1_decoder(result.e_p1)
    p2_star = p2_decoder(result.e_p2)
    nashconv = compute_nash_conv(game, p1_star, p2_star)
    sampled_payoff = get_expected_payoffs(game, p1_star, p2_star)

    # random-embedding-pair baseline
    baseline_vals = []
    for _ in range(config.nashconv_baseline_samples):
        e1 = pool_p1[np.random.randint(len(pool_p1))]
        e2 = pool_p2[np.random.randint(len(pool_p2))]
        baseline_vals.append(compute_nash_conv(game, p1_decoder(e1), p2_decoder(e2)))
    nashconv_baseline = float(np.mean(baseline_vals))

    logger.info(f"Task F NashConv: {nashconv:.6f}  baseline: {nashconv_baseline:.6f}")

    return {
        "nashconv": float(nashconv),
        "nashconv_baseline": nashconv_baseline,
        "value_at_star": float(result.value),
        "sampled_payoff_at_star": float(sampled_payoff),
        "matrix_game_baseline": matrix_baseline,
        "val_metrics": val_metrics,
        "config": config_to_dict(config),
        "posttrain": bool(config.posttrain),
        "plot_path": plot_path,
        "solve_stats_path": solve_stats_path,
    }
