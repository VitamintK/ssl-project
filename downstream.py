"""
Downstream task predictors for self-supervised learning evaluation.

This module provides predictors for evaluating policy representations:
1. ModelTrainer: Handles training/evaluation for all model types
2. PayoffPredictorRefactored: Predicts expected payoffs between policies
3. StatePayoffPredictorRefactored: Predicts state-conditioned payoffs
4. ExploitabilityPredictorRefactored: Predicts policy exploitability

Supports multiple model types: MLP, linear regression, and random forest.
Uses composition pattern for clean separation of ML and domain logic.
"""

from abc import ABC, abstractmethod
import copy
import os
import time
from dataclasses import dataclass
from typing import List, Callable, Any, Optional, Literal
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pyspiel
import matplotlib.pyplot as plt
from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode
from open_spiel.python.policy import Policy, UniformRandomPolicy
from pyspiel import TabularBestResponse
from open_spiel.python.algorithms import policy_utils, best_response
from open_spiel.python.algorithms.psro_v2 import utils as psro_utils

from iig_rl_benchmark.algorithms.ppo.ppo import PPOConditionedOnPolicyRepresentationAgent
from iig_rl_benchmark.algorithms.ppo.ppo_wrapper import SimplePPOWrapper

from config import ModelConfig, ExperimentInfo
from utils import PPOAgentPolicy, PPONeuplAgentPolicy, PolicyAsAgent, get_device_string, get_expected_payoffs

# Import sklearn only when needed
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_data(total_len: int, validation_split: float):
    """
    Create train/val split with random permutation.

    Args:
        total_len: Total number of data points
        validation_split: Fraction for validation (0, 1)

    Returns:
        Tuple of (train_indices, val_indices)
    """
    n_val = max(1, int(total_len * validation_split))
    perm = np.random.permutation(total_len)
    return perm[n_val:], perm[:n_val]


def get_state_tensor(state):
    """Concatenate both player's info for full state representation."""
    state_tensor_p0 = state.information_state_tensor(0)
    state_tensor_p1 = state.information_state_tensor(1)
    return state_tensor_p0 + state_tensor_p1


def sample_random_states(game, num_states: int, max_depth: int = 10):
    """
    Sample random game states by doing random rollouts.

    Args:
        game: OpenSpiel game instance
        num_states: Number of states to sample
        max_depth: Maximum depth to sample states from

    Returns:
        List of game states
    """
    states = []
    attempts = 0
    max_attempts = num_states * 10  # Avoid infinite loops

    while len(states) < num_states and attempts < max_attempts:
        attempts += 1
        state = game.new_initial_state()

        # Random rollout to some depth
        depth = random.randint(1, max_depth)
        for _ in range(depth):
            if state.is_terminal():
                break
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                action = random.choices(actions, weights=probs)[0]
            else:
                legal_actions = state.legal_actions()
                action = random.choice(legal_actions)
            state.apply_action(action)

        # Only keep non-terminal states
        if not state.is_terminal():
            states.append(state.clone())

    if len(states) < num_states:
        print(f"Warning: Could only sample {len(states)} states out of {num_states} requested")

    return states


class PayoffModel(nn.Module):
    """Simple MLP to predict payoffs from agent embeddings."""

    def __init__(self, embedding_dim: int, hidden_dims: Optional[List[int]] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ModelTrainer:
    """
    Handles model training and evaluation for all predictor types.

    Provides a unified interface for training MLP, linear, and random forest models.
    Predictors delegate ML operations to this class, focusing on domain logic themselves.
    """

    def __init__(
        self,
        embedding_dim: int,
        model_config: ModelConfig,
        device: str = "cpu"
    ):
        """
        Initialize model trainer.

        Args:
            embedding_dim: Dimension of input embeddings
            model_config: Model configuration (type, hyperparameters)
            device: Device for computation (cpu, cuda, mps)
        """
        self.embedding_dim = embedding_dim
        self.model_config = model_config
        self.device = device
        self.model = self._create_model()
        self.train_indices = None
        self.val_indices = None

    def _create_model(self):
        """
        Factory method: Create model based on config.

        Returns:
            Model instance (RandomForestRegressor or PayoffModel)
        """
        if self.model_config.model_type == "random_forest":
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn is required for random_forest model type")
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Neural network (mlp or linear)
            return PayoffModel(
                self.embedding_dim,
                hidden_dims=self.model_config.hidden_dims,
                dropout=self.model_config.dropout
            ).to(self.device)

    def _split_data(self, total_len: int, validation_split: float):
        """
        Create train/val split with random permutation.

        Args:
            total_len: Total number of data points
            validation_split: Fraction for validation (0, 1)

        Returns:
            Tuple of (train_indices, val_indices)
        """
        return split_data(total_len, validation_split)

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train model on provided data.

        Args:
            X: Input embeddings (N, embedding_dim)
            y: Ground truth labels (N,)
            validation_split: Fraction for validation

        Returns:
            dict: Training history with 'train_loss' and 'val_loss' keys
        """
        # Create train/val split
        train_indices, val_indices = self._split_data(len(X), validation_split)
        self.train_indices = train_indices
        self.val_indices = val_indices

        # Dispatch to appropriate training method
        if self.model_config.model_type == "random_forest":
            return self._train_random_forest(X, y, train_indices, val_indices)
        else:
            return self._train_neural_network(X, y, train_indices, val_indices)

    def _train_random_forest(self, X, y, train_indices, val_indices):
        """
        Train random forest with progressive n_estimators and early stopping.

        Args:
            X: Input embeddings
            y: Ground truth labels
            train_indices: Indices for training
            val_indices: Indices for validation

        Returns:
            dict: Training history
        """
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        history = {'train_loss': [], 'val_loss': []}
        best_val_mse = float('inf')
        best_rf_model = None
        patience_counter = 0
        patience = 5

        for n_trees in [10, 25, 50, 100, 150, 200, 300]:
            rf_model = RandomForestRegressor(
                n_estimators=n_trees,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)

            train_pred = rf_model.predict(X_train)
            val_pred = rf_model.predict(X_val)
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)

            history['train_loss'].append(train_mse)
            history['val_loss'].append(val_mse)

            print(f"n_estimators={n_trees}: Train MSE={train_mse:.6f}, Val MSE={val_mse:.6f}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_rf_model = rf_model
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at n_estimators={n_trees}")
                    break

        self.model = best_rf_model
        return history

    def _train_neural_network(self, X, y, train_indices, val_indices):
        """
        Single implementation of neural network training with early stopping.

        Uses Adam optimizer, MSE loss, mini-batch SGD.

        Args:
            X: Input embeddings
            y: Ground truth labels
            train_indices: Indices for training
            val_indices: Indices for validation

        Returns:
            dict: Training history
        """
        cfg = self.model_config
        X_train = torch.FloatTensor(X[train_indices]).to(self.device)
        y_train = torch.FloatTensor(y[train_indices]).to(self.device)
        X_val = torch.FloatTensor(X[val_indices]).to(self.device)
        y_val = torch.FloatTensor(y[val_indices]).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(cfg.num_epochs):
            self.model.train()

            # Shuffle and mini-batch
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            train_losses = []
            for i in range(0, len(X_train), cfg.batch_size):
                batch_X = X_train_shuffled[i:i+cfg.batch_size]
                batch_y = y_train_shuffled[i:i+cfg.batch_size]

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = criterion(val_predictions, y_val)

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = val_loss.item()
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.num_epochs} - "
                      f"Train: {epoch_train_loss:.6f}, Val: {epoch_val_loss:.6f}")

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                 train_y: np.ndarray) -> dict:
        """
        Evaluate model on specified indices.

        Uses training set mean for baseline computation to prevent information leakage.

        Args:
            X: All input features
            y: All ground truth labels
            indices: Which indices to evaluate
            train_y: Ground truth for training set (for baseline)

        Returns:
            dict with mse, mae, baseline_mse, predictions, ground_truth
        """
        predictions = self.predict(X[indices])
        ground_truth = y[indices]

        # Compute metrics
        mse = np.mean((predictions - ground_truth) ** 2)
        mae = np.mean(np.abs(predictions - ground_truth))

        # Baseline uses training set mean
        train_mean = np.mean(train_y)
        baseline_mse = np.mean((ground_truth - train_mean) ** 2)

        return {
            'mse': mse,
            'mae': mae,
            'baseline_mse': baseline_mse,
            # 'predictions': predictions,
            # 'ground_truth': ground_truth
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for given inputs."""
        if self.model_config.model_type == "random_forest":
            return self.model.predict(X)
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                return self.model(X_tensor).cpu().numpy()


class PayoffPredictor:
    """
    Predicts expected payoff between P1 and P2 policies.

    Handles both single fixed opponent (Task A) and agent vs agent (Task B) cases.
    """

    def __init__(
        self,
        game,
        p1_policies: List[Policy],
        p2_policies: List[Policy],
        p1_embeddings: np.ndarray,
        p2_embeddings: np.ndarray,
        model_config: ModelConfig,
        device: str = "cpu",
        num_pairs: Optional[int] = None,
    ):
        """
        Initialize PayoffPredictor.

        Args:
            game: OpenSpiel game instance
            p1_policies: List of P1 policies
            p2_policies: List of P2 policies (can be single-element list for fixed opponent)
            p1_embeddings: P1 agent embeddings
            p2_embeddings: P2 agent embeddings
            model_config: Model configuration
            device: Device for computation
            num_pairs: If set, compute ground truth for a random sample of this many
                (P1, P2) pairs instead of the full N x M grid. None (default) uses all pairs.
        """
        # Domain setup
        self.game = game
        self.p1_policies = p1_policies
        self.p2_policies = p2_policies
        self.p1_embeddings = np.array(p1_embeddings)
        self.p2_embeddings = np.array(p2_embeddings)
        self.num_pairs = num_pairs

        # Create trainer via composition
        embedding_dim = self.p1_embeddings.shape[1] + self.p2_embeddings.shape[1]
        self.trainer = ModelTrainer(embedding_dim, model_config, device)

        # Domain state
        self.ground_truth_payoffs = None
        self.pair_indices = None  # Will be set in compute_ground_truth_payoffs

    def compute_ground_truth_payoffs(self):
        """Compute ground truth payoffs for P1-P2 agent pairs.

        By default every (P1, P2) pair in the N x M grid is evaluated. If ``num_pairs``
        was set, a random sample of that many distinct pairs is evaluated instead.
        """
        n_p1, n_p2 = len(self.p1_policies), len(self.p2_policies)
        total_pairs = n_p1 * n_p2

        if self.num_pairs is None or self.num_pairs >= total_pairs:
            pairs = [(p1_idx, p2_idx) for p1_idx in range(n_p1) for p2_idx in range(n_p2)]
        else:
            # Sample distinct pairs by flat index to avoid materializing the full grid.
            flat = np.random.choice(total_pairs, size=self.num_pairs, replace=False)
            pairs = [(int(f // n_p2), int(f % n_p2)) for f in flat]

        print(f"Computing ground truth payoffs for {len(pairs)} of {total_pairs} "
              f"({n_p1} x {n_p2}) agent pairs...")
        payoffs = []
        self.pair_indices = []
        for p1_idx, p2_idx in tqdm(pairs, desc="Agent pairs"):
            payoff = get_expected_payoffs(self.game, self.p1_policies[p1_idx], self.p2_policies[p2_idx])
            payoffs.append(payoff)
            self.pair_indices.append((p1_idx, p2_idx))

        self.ground_truth_payoffs = np.array(payoffs)
        return self.ground_truth_payoffs

    def train_with_agent_level_split(self, validation_split: float = 0.2):
        """
        Train with agent-level splitting to avoid leakage.

        This method overrides the base train() to implement custom agent-level
        splitting instead of random pair-level splitting.

        Args:
            validation_split: Fraction of agents for validation

        Returns:
            dict: Training history
        """
        # Compute ground truth if needed
        if self.ground_truth_payoffs is None:
            self.compute_ground_truth_payoffs()

        # Create concatenated embeddings for all pairs
        print("Creating concatenated embeddings for pairs...")
        X = np.array([
            np.concatenate([self.p1_embeddings[p1_idx], self.p2_embeddings[p2_idx]])
            for p1_idx, p2_idx in self.pair_indices
        ])
        y = self.ground_truth_payoffs

        # Agent-level split (prevents leakage)
        n_p1 = len(self.p1_policies)
        n_val_p1 = max(1, int(n_p1 * validation_split))
        if n_val_p1 >= n_p1:
            raise ValueError(f"Not enough P1 agents ({n_p1}) for train/val split. Need at least 2.")

        p1_perm = np.random.permutation(n_p1)
        val_p1_set = set(p1_perm[:n_val_p1])
        train_p1_set = set(p1_perm[n_val_p1:])

        # Handle single P2 (fixed opponent) vs multiple P2
        if len(self.p2_policies) == 1:
            # Single opponent: split only on P1
            print("Single P2 agent detected - using same opponent for train and val")
            train_indices = np.array([
                i for i, (p1_idx, _) in enumerate(self.pair_indices)
                if p1_idx in train_p1_set
            ])
            val_indices = np.array([
                i for i, (p1_idx, _) in enumerate(self.pair_indices)
                if p1_idx in val_p1_set
            ])
        else:
            # Multiple opponents: split on both
            n_p2 = len(self.p2_policies)
            n_val_p2 = max(1, int(n_p2 * validation_split))
            if n_val_p2 >= n_p2:
                raise ValueError(f"Not enough P2 agents ({n_p2}) for train/val split. Need at least 2.")

            p2_perm = np.random.permutation(n_p2)
            val_p2_set = set(p2_perm[:n_val_p2])
            train_p2_set = set(p2_perm[n_val_p2:])

            train_indices = np.array([
                i for i, (p1_idx, p2_idx) in enumerate(self.pair_indices)
                if p1_idx in train_p1_set and p2_idx in train_p2_set
            ])
            val_indices = np.array([
                i for i, (p1_idx, p2_idx) in enumerate(self.pair_indices)
                if p1_idx in val_p1_set and p2_idx in val_p2_set
            ])

        print(f"Split summary: {len(train_indices)} training pairs, {len(val_indices)} validation pairs")

        # Set indices on trainer
        self.trainer.train_indices = train_indices
        self.trainer.val_indices = val_indices

        # Delegate training to ModelTrainer
        if self.trainer.model_config.model_type == "random_forest":
            return self.trainer._train_random_forest(X, y, train_indices, val_indices)
        else:
            return self.trainer._train_neural_network(X, y, train_indices, val_indices)

    def evaluate(self, eval_set: str = "val"):
        """Prepare data and delegate evaluation to ModelTrainer."""
        X = self._prepare_training_data()
        y = self.ground_truth_payoffs

        if eval_set == "train":
            indices = self.trainer.train_indices
        elif eval_set == "val":
            indices = self.trainer.val_indices
        elif eval_set == "all":
            indices = np.arange(len(y))
        else:
            raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'train', 'val', or 'all'")

        train_y = y[self.trainer.train_indices]
        return self.trainer.evaluate(X, y, indices, train_y)

    def _prepare_training_data(self):
        """Convert pair_indices to concatenated embeddings."""
        return np.array([
            np.concatenate([self.p1_embeddings[p1_idx], self.p2_embeddings[p2_idx]])
            for p1_idx, p2_idx in self.pair_indices
        ])


class StatePayoffPredictor:
    """
    Predicts expected payoff for (P1, P2, state) triples.

    Evaluates policies from specific game states rather than initial state.
    """

    def __init__(
        self,
        game,
        p1_policies: List[Policy],
        p2_policies: List[Policy],
        p1_embeddings: np.ndarray,
        p2_embeddings: np.ndarray,
        model_config: ModelConfig,
        num_states: int,
        max_depth: int,
        device: str = "cpu"
    ):
        """
        Initialize StatePayoffPredictor.

        Args:
            game: OpenSpiel game instance
            p1_policies: List of P1 policies
            p2_policies: List of P2 policies
            p1_embeddings: P1 embeddings
            p2_embeddings: P2 embeddings
            model_config: Model configuration
            num_states: Number of states to sample
            max_depth: Maximum depth for state sampling
            device: Device for computation
        """
        # Domain setup
        self.game = game
        self.p1_policies = p1_policies
        self.p2_policies = p2_policies
        self.p1_embeddings = np.array(p1_embeddings)
        self.p2_embeddings = np.array(p2_embeddings)

        # Sample states
        print(f"Sampling {num_states} game states (max depth {max_depth})...")
        self.states = sample_random_states(game, num_states, max_depth)

        # Encode states
        print("Encoding states...")
        self.state_tensors = np.array([
            get_state_tensor(state) for state in tqdm(self.states, desc="States")
        ])

        # Create trainer via composition
        embedding_dim = (self.p1_embeddings.shape[1] +
                        self.p2_embeddings.shape[1] +
                        self.state_tensors.shape[1])
        self.trainer = ModelTrainer(embedding_dim, model_config, device)

        # Domain state
        self.ground_truth_payoffs = None
        self.triple_indices = None  # Will be set in compute_ground_truth_payoffs

    def compute_ground_truth_payoffs(self):
        """Compute ground truth payoffs for all (P1, P2, state) triples."""
        print(f"Computing ground truth payoffs for {len(self.p1_policies)} x {len(self.p2_policies)} x {len(self.states)} triples...")
        payoffs = []
        self.triple_indices = []

        for p1_idx, p1_policy in enumerate(tqdm(self.p1_policies, desc="P1 policies")):
            for p2_idx, p2_policy in enumerate(self.p2_policies):
                for state_idx, state in enumerate(self.states):
                    # Compute payoff starting from this state
                    payoff = self._compute_payoff_from_state(state, p1_policy, p2_policy)
                    payoffs.append(payoff)
                    self.triple_indices.append((p1_idx, p2_idx, state_idx))

        self.ground_truth_payoffs = np.array(payoffs)
        return self.ground_truth_payoffs

    def _compute_payoff_from_state(self, state, p1_policy: Policy, p2_policy: Policy):
        """Compute expected payoff starting from a given state."""
        policies = [p1_policy, p2_policy]

        # Sample episodes starting from this state
        payoffs = []
        from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode
        for _ in range(100):  # 100 episodes per state
            payoff = sample_episode(state.clone(), policies)[0]
            payoffs.append(payoff)

        return np.mean(payoffs)

    def train_with_agent_level_split(self, validation_split: float = 0.2):
        """
        Train with agent-level splitting to avoid leakage.

        Args:
            validation_split: Fraction of agents for validation

        Returns:
            dict: Training history
        """
        # Compute ground truth if needed
        if self.ground_truth_payoffs is None:
            self.compute_ground_truth_payoffs()

        # Create concatenated embeddings for all triples
        print("Creating concatenated embeddings for triples...")
        X = np.array([
            np.concatenate([
                self.p1_embeddings[p1_idx],
                self.p2_embeddings[p2_idx],
                self.state_tensors[state_idx]
            ])
            for p1_idx, p2_idx, state_idx in self.triple_indices
        ])
        y = self.ground_truth_payoffs

        # Agent-level split on P1 policies
        n_p1 = len(self.p1_policies)
        n_val_p1 = max(1, int(n_p1 * validation_split))
        if n_val_p1 >= n_p1:
            raise ValueError(f"Not enough P1 policies ({n_p1}) for train/val split. Need at least 2.")

        p1_perm = np.random.permutation(n_p1)
        val_p1_set = set(p1_perm[:n_val_p1])
        train_p1_set = set(p1_perm[n_val_p1:])

        # Split on P2 policies as well
        n_p2 = len(self.p2_policies)
        n_val_p2 = max(1, int(n_p2 * validation_split))
        if n_val_p2 >= n_p2:
            raise ValueError(f"Not enough P2 policies ({n_p2}) for train/val split. Need at least 2.")

        p2_perm = np.random.permutation(n_p2)
        val_p2_set = set(p2_perm[:n_val_p2])
        train_p2_set = set(p2_perm[n_val_p2:])

        # States are reused in both train and val
        train_indices = np.array([
            i for i, (p1_idx, p2_idx, _) in enumerate(self.triple_indices)
            if p1_idx in train_p1_set and p2_idx in train_p2_set
        ])
        val_indices = np.array([
            i for i, (p1_idx, p2_idx, _) in enumerate(self.triple_indices)
            if p1_idx in val_p1_set and p2_idx in val_p2_set
        ])

        print(f"Split summary: {len(train_indices)} training triples, {len(val_indices)} validation triples")

        # Set indices on trainer
        self.trainer.train_indices = train_indices
        self.trainer.val_indices = val_indices

        # Delegate training to ModelTrainer
        if self.trainer.model_config.model_type == "random_forest":
            return self.trainer._train_random_forest(X, y, train_indices, val_indices)
        else:
            return self.trainer._train_neural_network(X, y, train_indices, val_indices)

    def evaluate(self, eval_set: str = "val"):
        """Prepare data and delegate evaluation to ModelTrainer."""
        X = self._prepare_training_data()
        y = self.ground_truth_payoffs

        if eval_set == "train":
            indices = self.trainer.train_indices
        elif eval_set == "val":
            indices = self.trainer.val_indices
        elif eval_set == "all":
            indices = np.arange(len(y))
        else:
            raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'train', 'val', or 'all'")

        train_y = y[self.trainer.train_indices]
        return self.trainer.evaluate(X, y, indices, train_y)

    def _prepare_training_data(self):
        """Convert triple_indices to concatenated embeddings."""
        return np.array([
            np.concatenate([
                self.p1_embeddings[p1_idx],
                self.p2_embeddings[p2_idx],
                self.state_tensors[state_idx]
            ])
            for p1_idx, p2_idx, state_idx in self.triple_indices
        ])


class ExploitabilityPredictor:
    """
    Predicts exploitability (best response payoff) for policies.

    Evaluates how much a policy can be exploited by a best-responding opponent.
    """

    def __init__(
        self,
        game,
        policies: List[Policy],
        embeddings: np.ndarray,
        model_config: ModelConfig,
        player_id: int = 0,
        device: str = "cpu"
    ):
        """
        Initialize ExploitabilityPredictor.

        Args:
            game: OpenSpiel game instance
            policies: List of policies to evaluate
            embeddings: Policy embeddings
            model_config: Model configuration
            player_id: Which player perspective (0 or 1)
            device: Device for computation
        """
        # Domain setup
        self.game = game
        self.initial_state = game.new_initial_state()
        self.policies = policies
        self.player_id = player_id
        self.embeddings_array = np.array(embeddings)

        # Create trainer via composition
        embedding_dim = self.embeddings_array.shape[1]
        self.trainer = ModelTrainer(embedding_dim, model_config, device)

        # Domain state
        self.ground_truth_payoffs = None
        self.all_states = None
        self.state_to_information_state = None
        self.best_response_processor = None

    def compute_ground_truth_payoffs(self):
        """Compute ground truth exploitability using best response oracle."""
        print(f"Computing best response payoffs for {len(self.policies)} policies...")

        best_responder_id = 1 - self.player_id

        # Compute all states and info states
        self.all_states, self.state_to_information_state = (
            psro_utils.compute_states_and_info_states_if_none(
                self.game, None, None
            )
        )

        # Create dummy policy for initializing best response
        policy = UniformRandomPolicy(self.game)
        policy_to_dict = policy_utils.policy_to_dict(
            policy, self.game, self.all_states, self.state_to_information_state
        )

        # Create best response processor
        self.best_response_processor = TabularBestResponse(
            self.game, best_responder_id, policy_to_dict
        )

        # Compute best response value for each policy
        payoffs = []
        for policy in tqdm(self.policies, desc="Computing best responses"):
            self.best_response_processor.set_policy(
                policy_utils.policy_to_dict(
                    policy, self.game, self.all_states, self.state_to_information_state
                )
            )
            best_responder = best_response.CPPBestResponsePolicy(
                self.game, best_responder_id, policy, self.all_states,
                self.state_to_information_state, self.best_response_processor
            )
            payoff = best_responder.value(self.initial_state)
            payoffs.append(payoff)

        self.ground_truth_payoffs = np.array(payoffs)
        return self.ground_truth_payoffs

    def train_with_agent_level_split(self, validation_split: float = 0.2):
        """
        Train with agent-level splitting to avoid leakage.

        Args:
            validation_split: Fraction of policies for validation

        Returns:
            dict: Training history
        """
        # Compute ground truth if needed
        if self.ground_truth_payoffs is None:
            self.compute_ground_truth_payoffs()

        # Prepare data
        X = self.embeddings_array
        y = self.ground_truth_payoffs

        # Agent-level split (simple for exploitability - just policies, not pairs)
        n_policies = len(self.policies)
        n_val = max(1, int(n_policies * validation_split))
        if n_val >= n_policies:
            raise ValueError(f"Not enough policies ({n_policies}) for train/val split. Need at least 2.")

        policy_perm = np.random.permutation(n_policies)
        val_indices = policy_perm[:n_val]
        train_indices = policy_perm[n_val:]

        print(f"Split summary: {len(train_indices)} training policies, {len(val_indices)} validation policies")

        # Set indices on trainer
        self.trainer.train_indices = train_indices
        self.trainer.val_indices = val_indices

        # Delegate training to ModelTrainer
        if self.trainer.model_config.model_type == "random_forest":
            return self.trainer._train_random_forest(X, y, train_indices, val_indices)
        else:
            return self.trainer._train_neural_network(X, y, train_indices, val_indices)

    def evaluate(self, eval_set: str = "val"):
        """Prepare data and delegate evaluation to ModelTrainer."""
        X = self.embeddings_array
        y = self.ground_truth_payoffs

        if eval_set == "train":
            indices = self.trainer.train_indices
        elif eval_set == "val":
            indices = self.trainer.val_indices
        elif eval_set == "all":
            indices = np.arange(len(y))
        else:
            raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'train', 'val', or 'all'")

        train_y = y[self.trainer.train_indices]
        return self.trainer.evaluate(X, y, indices, train_y)

# Best Response Learner ################################################################
from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_environment import ChanceEventSampler
from open_spiel.python.vector_env import SyncVectorEnv

from iig_rl_benchmark.algorithms.ppo.ppo import PPO

def make_single_env(game_name, seed, config):
    def gen_env():
        game = pyspiel.load_game(game_name)
        return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

    return gen_env

class BestResponseLearner:
    def __init__(
            self,
            game,
            policies: list[Policy],
            embeddings,
            config,
            policy_player_id,
            validation_split: float = 0.2,
            device: str = "cpu",
    ):
        self.device = device
        self.game = game
        self.policies = policies
        if isinstance(embeddings[0], np.ndarray):
            self.embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)
        else:
            self.embeddings = embeddings
        self.policy_player_id = policy_player_id
        self.best_responder_player_id = 1 - policy_player_id
        self.config = config.algorithm
        self.meta_config = config

        # Create train/val split
        self.train_indices, self.val_indices = split_data(len(policies), validation_split)
        self.train_policies = [policies[i] for i in self.train_indices]
        self.train_embeddings = self.embeddings[self.train_indices]
        self.val_policies = [policies[i] for i in self.val_indices]
        self.val_embeddings = self.embeddings[self.val_indices]
        print(f"BestResponseLearner split: {len(self.train_policies)} train, {len(self.val_policies)} val")

    def train_best_responder(self, optimizer_type="adam", epochs=1, max_batch_size=20, num_trajectories_per_policy_per_epoch=2, experiment_info: ExperimentInfo = None, is_control: bool = False, plot_dir: str = None):
        NUM_ENVS = 1
        # num_steps_per_batch = 20 # self.config.num_steps # IS THIS USED?
        info_state_shape = self.game.information_state_tensor_shape()
        game = self.game
        # device = get_device_string()
        device = self.device
        # best_responder = PPOConditionedOnPolicyRepresentationAgent(
        #     num_actions=game.num_distinct_actions(),
        #     observation_shape=info_state_shape,
        #     device=device,
        #     num_policies=1, # we don't use the embedding layer, since we already have the embeddings
        #     policy_embedding_size=self.embeddings[0].shape,
        # )
        # batch_size = int(NUM_ENVS * max_batch_size)
        # num_updates = self.meta_config.max_steps // batch_size + 1 # THIS IS NOT USED
        # envs = SyncVectorEnv(
        #     [
        #         make_single_env(
        #             str(self.game),
        #             self.meta_config.seed + i,
        #             self.meta_config,
        #         )()
        #         for i in range(self.config.num_envs)
        #     ]
        # )
        env = make_single_env(str(self.game), self.meta_config.seed, self.meta_config)()
        self.agent = PPO(
            input_shape=info_state_shape,
            num_actions=game.num_distinct_actions(),
            num_players=game.num_players(),
            num_envs=NUM_ENVS,
            steps_per_batch=max_batch_size,
            num_minibatches=self.config.num_minibatches,
            update_epochs=self.config.update_epochs,
            learning_rate=self.config.learning_rate,
            gae=self.config.gae,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            normalize_advantages=self.config.norm_adv,
            clip_coef=self.config.clip_coef,
            clip_vloss=self.config.clip_vloss,
            entropy_coef=self.config.ent_coef,
            value_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            target_kl=self.config.target_kl,
            device=device,
            agent_fn=PPOConditionedOnPolicyRepresentationAgent,
            neupl_ppo_policy_embedding=self.embeddings[0], # we'll provide this later
            neupl_ppo_kwargs={"num_policies": 1, # this doesn't matter because we don't use the embedding layer
                "policy_embedding_size": self.embeddings[0].shape[0],},
            use_joint_obs_for_critic=True,
            optimizer_type=optimizer_type,
            log_file=os.path.join(self.meta_config.experiment_dir, 'train_log.csv'),
        )
        agents = [None, None]
        agents[self.best_responder_player_id] = SimplePPOWrapper(self.best_responder_player_id, self.agent)

        # Track training metrics
        trajectory_payoffs = []
        ppo_metrics_list = []

        initial_lr = self.config.learning_rate
        final_lr = getattr(self.config, 'final_learning_rate', initial_lr)

        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"Epoch {epoch} of {epochs}")
            if epochs > 1:
                lr = initial_lr + (final_lr - initial_lr) * epoch / (epochs - 1)
            else:
                lr = initial_lr
            self.agent.optimizer.param_groups[0]["lr"] = lr
            # Add a tqdm progress bar to show progress through policies in the current epoch
            pbar = tqdm(
                zip(self.train_policies, self.train_embeddings),
                total=len(self.train_policies),
                desc=f"Epoch {epoch+1}/{epochs} Policy Progress",
            )
            for policy, embedding in pbar:
                self.agent.set_policy_embedding(embedding)
                agents[self.policy_player_id] = PolicyAsAgent(self.policy_player_id, self.game.num_distinct_actions(), np.random.default_rng(), policy)
                for _ in range(num_trajectories_per_policy_per_epoch):
                    time_step = env.reset()
                    cumulative_rewards = 0.0
                    while not time_step.last():
                        # self._num_steps += 1
                        player_id = time_step.observations["current_player"]
                        # is_evaluation is a boolean that, when False, lets policies train. The
                        # setting of PSRO requires that all policies be static aside from those
                        # being trained by the oracle. is_evaluation could be used to prevent
                        # policies from training, yet we have opted for adding frozen attributes
                        # that prevents policies from training, for all values of is_evaluation.
                        # Since all policies returned by the oracle are frozen before being
                        # returned, only currently-trained policies can effectively learn.
                        # if isinstance(agents[0]._policy, ppo_wrapper.PPOWrapper) and isinstance(agents[1]._policy, ppo_wrapper.PPOWrapper):
                        #     pass
                        # agent_output = agents[player_id].step(
                        #     time_step, is_evaluation=is_evaluation
                        # )
                        is_evaluation = (player_id == self.policy_player_id)
                        agent_output = agents[player_id].step(time_step, is_evaluation=is_evaluation)
                        action_list = [agent_output.action]
                        time_step = env.step(action_list)
                        cumulative_rewards += np.array(time_step.rewards)
                        if isinstance(
                            agents[player_id], SimplePPOWrapper
                        ):  # self._best_response_kwargs['oracle_type'] == "ppo":
                            agents[player_id].post_step(time_step, is_evaluation=is_evaluation)

                    # If the last player to act was the non-training PPO agent, then we need to post-step the training PPO agent.
                    for pid, agent in enumerate(agents):
                        if isinstance(agent, SimplePPOWrapper):
                            if pid == 1 - player_id:
                                assert pid == self.best_responder_player_id
                                agent.post_step(time_step, is_evaluation=False)
                        else:
                            agent.step(time_step)

                    # Track the payoff for this trajectory
                    trajectory_payoffs.append(cumulative_rewards[self.best_responder_player_id])
                

                fixed_obs_dict = copy.copy(time_step.observations)
                fixed_obs_dict["current_player"] = self.best_responder_player_id
                from open_spiel.python.rl_environment import TimeStep
                fixed_time_step = TimeStep(
                    observations=fixed_obs_dict,
                    rewards=time_step.rewards,
                    discounts=time_step.discounts,
                    step_type=time_step.step_type,
                )
                metrics = None
                if self.agent.cur_batch_idx >= 2:
                    metrics = self.agent.learn([fixed_time_step])
                else:
                    self.agent.cur_batch_idx = 0
                # Snap total_steps_done to a multiple of steps_per_batch so the auto-learn
                # trigger fires only after a full buffer accumulates, not after 1 step.
                self.agent.total_steps_done = (self.agent.total_steps_done // self.agent.steps_per_batch) * self.agent.steps_per_batch
                if metrics is not None:
                    ppo_metrics_list.append(metrics)

            print(f"Epoch {epoch} took {time.time() - epoch_start:.2f}s")

        # Store metrics for later access
        self.training_metrics = {
            'trajectory_payoffs': trajectory_payoffs,
            'ppo_metrics': ppo_metrics_list,
        }

        # Plot training metrics
        self._plot_training_metrics(trajectory_payoffs, ppo_metrics_list, window_size=450, experiment_info=experiment_info, is_control=is_control, plot_dir=plot_dir)
                # self.agent.learn([time_step])
                # return cumulative_rewards
                # update = -1
                # t0 = time.time()
                # time_steps = envs.reset()
                # cp_step = 0
                # self.agent.total_steps_done = 0
                # while self.agent.total_steps_done < num_steps_per_policy_per_epoch:
                #     update += 1
                #     for _ in range(num_steps_per_batch):
                #         # Output of current player in each of the envs
                #         agent_outputs = self.agent.step(time_steps)

                #         # Advance all envs
                #         time_steps, rewards, dones, unreset_time_steps = envs.step(
                #             agent_outputs, reset_if_done=True
                #         )
                #         self.agent.post_step([reward[0] for reward in rewards], dones)
                #     if self.config.anneal_lr:
                #         self.agent.anneal_learning_rate(update, num_updates)
                #     self.agent.learn(time_steps)

                #     if self.agent.total_steps_done > cp_step + self.meta_config.compute_exploitability_every:
                #         cp_step = cp_step + self.meta_config.compute_exploitability_every
                #         if self.expl_callback is not None:
                #             self.expl_callback(
                #                 self.get_model(), self.get_model(), self.agent.total_steps_done
                #             )
                #         self.agent.save(f"{self.meta_config.experiment_dir}/agent.pth")

                #     if update % self.config.eval_every == 0:
                #         time_elapsed = time.time() - t0
                #         time_remaining_est = (
                #             # (self.meta_config.max_steps - self.agent.total_steps_done)
                #             (num_steps_per_policy_per_epoch - self.agent.total_steps_done)
                #             * time_elapsed
                #             / self.agent.total_steps_done
                #         )
                #         # print(f"step {self.agent.total_steps_done}/{self.meta_config.max_steps} ; elapsed: {time_elapsed/60:.1f}min ; remaining: {time_remaining_est/60:.1f}min")
                #         # print(f"step {self.agent.total_steps_done}/{num_steps_per_policy_per_epoch} ; elapsed: {time_elapsed/60:.1f}min ; remaining: {time_remaining_est/60:.1f}min")
                #         # pbar.set_postfix(elapsed=f"{time_elapsed/60:.1f}min", remaining=f"{time_remaining_est/60:.1f}min")
                #         pbar.set_description(f"step {self.agent.total_steps_done}/{num_steps_per_policy_per_epoch} ; elapsed: {time_elapsed/60:.1f}min ; remaining: {time_remaining_est/60:.1f}min")

    def _plot_training_metrics(self, trajectory_payoffs: list, ppo_metrics: list, window_size: int = 100, experiment_info: ExperimentInfo = None, is_control: bool = False, plot_dir: str = None):
        """
        Plot training metrics with moving average.

        Args:
            trajectory_payoffs: List of payoffs from each trajectory
            ppo_metrics: List of dicts with PPO training metrics (value_loss, policy_loss, etc.)
            window_size: Window size for moving average
            experiment_info: Experiment info for labeling the plot
        """
        if len(trajectory_payoffs) == 0:
            print("No training data to plot")
            return

        payoffs = np.array(trajectory_payoffs)

        # Compute moving average helper
        def compute_moving_avg(data, window):
            actual_window = min(window, len(data))
            if actual_window == 0:
                return np.array([]), np.array([])
            moving_avg = np.convolve(data, np.ones(actual_window) / actual_window, mode='valid')
            ma_x = np.arange(actual_window - 1, actual_window - 1 + len(moving_avg))
            return ma_x, moving_avg

        # Create 3x2 subplot figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))

        # Row 0, Col 0: Trajectory payoffs
        ax = axes[0, 0]
        ax.plot(payoffs, alpha=0.3, color='blue', label='Raw payoffs')
        ma_x, moving_avg = compute_moving_avg(payoffs, window_size)
        if len(moving_avg) > 0:
            ax.plot(ma_x, moving_avg, color='red', linewidth=2, label=f'Moving avg (window={min(window_size, len(payoffs))})')
        ax.set_xlabel('Trajectory')
        ax.set_ylabel('Payoff')
        ax.set_title('Trajectory Payoffs')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Extract PPO metrics if available
        if len(ppo_metrics) > 0:
            value_losses = np.array([m['value_loss'] for m in ppo_metrics])
            explained_variances = np.array([m['explained_variance'] for m in ppo_metrics])
            entropies = np.array([m['entropy'] for m in ppo_metrics])
            grad_norms = np.array([m.get('grad_norm', 0) for m in ppo_metrics])

            # Debug: print explained variance stats
            print(f"Explained Variance stats: min={np.min(explained_variances):.4f}, max={np.max(explained_variances):.4f}, "
                  f"mean={np.mean(explained_variances):.4f}, median={np.median(explained_variances):.4f}")
            print(f"Explained Variance first 10: {explained_variances[:10]}")
            print(f"Explained Variance last 10: {explained_variances[-10:]}")

            # Top-right: Value loss
            ax = axes[0, 1]
            ax.plot(value_losses, alpha=0.3, color='orange', label='Raw')
            ma_x, moving_avg = compute_moving_avg(value_losses, window_size)
            if len(moving_avg) > 0:
                ax.plot(ma_x, moving_avg, color='red', linewidth=2, label='Moving avg')
            ax.set_xlabel('Update')
            ax.set_ylabel('Value Loss')
            ax.set_title('Value Loss (should decrease)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Bottom-left: Explained variance
            ax = axes[1, 0]
            ax.plot(explained_variances, alpha=0.3, color='green', label='Raw')
            ma_x, moving_avg = compute_moving_avg(explained_variances, window_size)
            if len(moving_avg) > 0:
                ax.plot(ma_x, moving_avg, color='red', linewidth=2, label='Moving avg')
            ax.set_xlabel('Update')
            ax.set_ylabel('Explained Variance')
            ax.set_title('Explained Variance (should approach 1.0)')
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(-1, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Row 1, Col 1: Entropy
            ax = axes[1, 1]
            ax.plot(entropies, alpha=0.3, color='purple', label='Raw')
            ma_x, moving_avg = compute_moving_avg(entropies, window_size)
            if len(moving_avg) > 0:
                ax.plot(ma_x, moving_avg, color='red', linewidth=2, label='Moving avg')
            ax.set_xlabel('Update')
            ax.set_ylabel('Entropy')
            ax.set_title('Policy Entropy (should not collapse to 0)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Row 2, Col 0: Gradient Norm
            ax = axes[2, 0]
            ax.plot(grad_norms, alpha=0.3, color='brown', label='Raw')
            ma_x, moving_avg = compute_moving_avg(grad_norms, window_size)
            if len(moving_avg) > 0:
                ax.plot(ma_x, moving_avg, color='red', linewidth=2, label='Moving avg')
            ax.set_xlabel('Update')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm (before clipping)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Row 2, Col 1: Empty or future use
            axes[2, 1].set_axis_off()
        else:
            # No PPO metrics - add text to remaining subplots
            for ax in [axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]]:
                ax.text(0.5, 0.5, 'No PPO metrics available',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()

        # Add experiment label as title at top of figure
        if experiment_info:
            run_type = "Control" if is_control else "Experiment"
            fig.suptitle(f"[{run_type}] {experiment_info.label_string}", fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save plot to the provided plot_dir, or fall back to a timestamped directory
        from datetime import datetime
        if plot_dir is not None:
            run_dir = plot_dir
        else:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            random_suffix = random.randint(10, 99)
            run_dir = os.path.join('results', 'training_metrics', f'{timestamp}_{random_suffix}')
        os.makedirs(run_dir, exist_ok=True)
        filename = 'control.png' if is_control else 'experiment.png'
        plot_path = os.path.join(run_dir, filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {plot_path}")
        plt.close(fig)

    def evaluate(self, eval_set: str = "val", num_episodes_per_policy: int = 200):
        """
        Evaluate the trained best-responder on the specified set.

        Args:
            eval_set: Which set to evaluate on ('val' or 'train')
            num_episodes_per_policy: Number of episodes to run per policy for empirical evaluation

        Returns:
            dict with empirical_payoffs, exact_exploitabilities, and averages
        """
        if eval_set == "val":
            policies = self.val_policies
            embeddings = self.val_embeddings
        elif eval_set == "train":
            policies = self.train_policies
            embeddings = self.train_embeddings
        else:
            raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'val' or 'train'")

        # Empirically run the trained best-responder against the policies in the validation set
        # collect the empirical payoffs for each trajectory against each policy, and the average payoff against each policy,
        # and the average payoff against all policies.
        empirical_payoffs_per_policy = []
        empirical_payoffs_all_trajectories = []

        # best_responder_player_id = 0  # The best-responder plays as player 0

        for policy, embedding in tqdm(zip(policies, embeddings), total=len(policies), desc="Empirical evaluation"):
            # Set the embedding for the best-responder
            self.agent.set_policy_embedding(embedding)

            # Create a policy wrapper for the trained best-responder
            br_policy = PPONeuplAgentPolicy(
                self.game,
                self.agent.network,  # Access the underlying PPOConditionedOnPolicyRepresentationAgent
                self.best_responder_player_id,
                use_observation=False,
                embedding=embedding
            )

            # Run episodes and collect payoffs
            episode_payoffs = []
            policies_to_sample = [None, None]
            policies_to_sample[self.best_responder_player_id] = br_policy
            policies_to_sample[self.policy_player_id] = policy
            for _ in range(num_episodes_per_policy):
                payoff = sample_episode(self.game.new_initial_state(), policies_to_sample)[self.best_responder_player_id]
                episode_payoffs.append(payoff)

            avg_payoff = np.mean(episode_payoffs)
            empirical_payoffs_per_policy.append(avg_payoff)
            empirical_payoffs_all_trajectories.extend(episode_payoffs)

        avg_empirical_payoff = np.mean(empirical_payoffs_per_policy)

        # Compute exact exploitability against each policy in the validation set, and the average exact exploitability for all policies
        exact_exploitabilities = []
        # best_responder_id = 0  # We compute best response value for player 0

        # Compute all states and info states (needed for tabular best response)
        all_states, state_to_information_state = psro_utils.compute_states_and_info_states_if_none(
            self.game, None, None
        )

        # Create dummy policy for initializing best response processor
        dummy_policy = UniformRandomPolicy(self.game)
        policy_to_dict = policy_utils.policy_to_dict(
            dummy_policy, self.game, all_states, state_to_information_state
        )

        # Create best response processor
        best_response_processor = TabularBestResponse(
            self.game, self.best_responder_player_id, policy_to_dict
        )

        initial_state = self.game.new_initial_state()

        for policy in tqdm(policies, desc="Computing exact exploitabilities"):
            # Set the policy we're computing exploitability against
            best_response_processor.set_policy(
                policy_utils.policy_to_dict(
                    policy, self.game, all_states, state_to_information_state
                )
            )
            # Compute best response policy and its value
            br = best_response.CPPBestResponsePolicy(
                self.game, self.best_responder_player_id, policy, all_states,
                state_to_information_state, best_response_processor
            )
            exact_value = br.value(initial_state)
            exact_exploitabilities.append(exact_value)

        avg_exact_exploitability = np.mean(exact_exploitabilities)

        results = {
            'empirical_payoffs_per_policy': empirical_payoffs_per_policy,
            'empirical_payoffs_all_trajectories': empirical_payoffs_all_trajectories,
            'avg_empirical_payoff': avg_empirical_payoff,
            'exact_exploitabilities': exact_exploitabilities,
            'avg_exact_exploitability': avg_exact_exploitability,
            'num_policies': len(policies),
        }
        print(results['avg_empirical_payoff'])
        print(results['avg_exact_exploitability'])
        return results


def continue_train_value_model(model, X, y, epochs, lr, device="cpu"):
    """In-place MSE SGD on a torch value model. X: (n, D), y: (n,).

    NOTE: EmbeddingEquilibriumSolver.__init__ freezes the value model's
    parameters (requires_grad_(False)) since it only needs gradients w.r.t.
    the embeddings, not the model weights. run_task_f passes the same model
    object (predictor.trainer.model) into the solver, so by the time this
    function is later called to posttrain that model, its parameters are
    frozen. Re-enable gradients here before optimizing, or Adam.step() will
    silently be a no-op.
    """
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()


def compute_nash_conv(game, p1_policy, p2_policy) -> float:
    """NashConv of the joint profile (p1_policy plays player 0, p2_policy plays player 1)."""
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability as _exploitability

    joint = policy_lib.TabularPolicy(game)
    for state in joint.states:
        pid = state.current_player()
        src = p1_policy if pid == 0 else p2_policy
        probs = src.action_probabilities(state)
        row = joint.action_probability_array[joint.state_index(state)]
        row[:] = 0.0
        for action, p in probs.items():
            row[action] = p
    return float(_exploitability.nash_conv(game, joint))


@dataclass
class SolveResult:
    e_p1: np.ndarray
    e_p2: np.ndarray
    value: float
    visited: list  # list[tuple[np.ndarray, np.ndarray]]
    stats: list = None  # list[dict]: per inner-step records, e.g. {'outer_step', 'inner_step', 'v'}


class EmbeddingEquilibriumSolver:
    """Two-timescale gradient descent-ascent on a frozen value model, in embedding space.

    Player 0 (P1) maximizes V; player 1 (P2) minimizes V.
    """

    def __init__(self, value_model, pool_p1, pool_p2, config, device="cpu"):
        self.model = value_model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.config = config
        self.pool_p1 = np.asarray(pool_p1, dtype=np.float32)
        self.pool_p2 = np.asarray(pool_p2, dtype=np.float32)
        self.d1 = self.pool_p1.shape[1]
        self.d2 = self.pool_p2.shape[1]
        self.lo1 = torch.tensor(self.pool_p1.min(0), device=device)
        self.hi1 = torch.tensor(self.pool_p1.max(0), device=device)
        self.lo2 = torch.tensor(self.pool_p2.min(0), device=device)
        self.hi2 = torch.tensor(self.pool_p2.max(0), device=device)

    def _value(self, e1, e2):
        return self.model(torch.cat([e1, e2]).unsqueeze(0)).squeeze(0)

    def _clamp(self, e, lo, hi):
        if self.config.bound_embeddings:
            return torch.max(torch.min(e, hi), lo)
        return e

    def solve(self, init_p1=None, init_p2=None) -> SolveResult:
        cfg = self.config
        rng = np.random
        if init_p1 is None:
            init_p1 = self.pool_p1[rng.randint(len(self.pool_p1))]
        if init_p2 is None:
            init_p2 = self.pool_p2[rng.randint(len(self.pool_p2))]
        e1 = torch.tensor(np.asarray(init_p1, np.float32), device=self.device, requires_grad=True)
        e2 = torch.tensor(np.asarray(init_p2, np.float32), device=self.device, requires_grad=True)
        visited = []
        stats = []  # generic per-step log; currently records v at each inner step
        for outer_step in range(cfg.outer_steps):
            # inner loop: P2 minimizes V (fast)
            for inner_step in range(cfg.inner_steps):
                v = self._value(e1, e2)
                stats.append({'outer_step': outer_step, 'inner_step': inner_step,
                              'v': float(v.detach())})
                (g2,) = torch.autograd.grad(v, e2)
                with torch.no_grad():
                    e2 = self._clamp(e2 - cfg.lr_p2 * g2, self.lo2, self.hi2)
                e2.requires_grad_(True)
            # outer step: P1 maximizes V (slow)
            v = self._value(e1, e2)
            (g1,) = torch.autograd.grad(v, e1)
            with torch.no_grad():
                e1 = self._clamp(e1 + cfg.lr_p1 * g1, self.lo1, self.hi1)
            e1.requires_grad_(True)
            visited.append((e1.detach().cpu().numpy().copy(),
                            e2.detach().cpu().numpy().copy()))
        with torch.no_grad():
            final_v = float(self._value(e1, e2))
        return SolveResult(e1.detach().cpu().numpy(), e2.detach().cpu().numpy(),
                           final_v, visited, stats)

    def solve_best_of_restarts(self, score_fn) -> SolveResult:
        """Run num_restarts solves; return the result minimizing score_fn(result) (lower=better).

        The per-solve stats logs for every restart of the most recent call are kept on
        ``self.last_restart_stats`` (one element per restart) for inspection/plotting.
        """
        best, best_score = None, float("inf")
        self.last_restart_stats = []
        for _ in range(self.config.num_restarts):
            res = self.solve()
            self.last_restart_stats.append(res.stats)
            s = score_fn(res)
            if s < best_score:
                best, best_score = res, s
        return best


def plot_solve_value_curves(restart_stats, save_path, title=None):
    """Plot the value trajectory of each solve run, one subplot per restart.

    Args:
        restart_stats: list where each element is one solve()'s stats list (dicts with
            'outer_step', 'inner_step', 'v'); e.g. ``solver.last_restart_stats``.
        save_path: file path to write the PNG to.
        title: optional overall figure title.

    Returns:
        The save_path written, or None if there was nothing to plot.
    """
    import math
    from pathlib import Path as _Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [s for s in restart_stats if s]
    if not runs:
        return None

    n = len(runs)
    ncols = 2 if n > 1 else 1
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    for k, stats in enumerate(runs):
        ax = axes[k // ncols][k % ncols]
        vs = [rec['v'] for rec in stats]
        ax.plot(range(len(vs)), vs, lw=1)
        ax.set_title(f"restart {k}")
        ax.set_xlabel("inner step")
        ax.set_ylabel("v")
        ax.grid(True, alpha=0.3)
    # hide any unused axes in the grid
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    _Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
