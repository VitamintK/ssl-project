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
from typing import List, Callable, Any, Optional, Literal
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pyspiel
from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode
from open_spiel.python.policy import Policy, UniformRandomPolicy
from pyspiel import TabularBestResponse
from open_spiel.python.algorithms import policy_utils, best_response
from open_spiel.python.algorithms.psro_v2 import utils as psro_utils

from config import ModelConfig
from utils import PPOAgentPolicy, get_expected_payoffs

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
        n_val = max(1, int(total_len * validation_split))
        perm = np.random.permutation(total_len)
        return perm[n_val:], perm[:n_val]

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
            'predictions': predictions,
            'ground_truth': ground_truth
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


class PayoffPredictorRefactored:
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
        device: str = "cpu"
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
        """
        # Domain setup
        self.game = game
        self.p1_policies = p1_policies
        self.p2_policies = p2_policies
        self.p1_embeddings = np.array(p1_embeddings)
        self.p2_embeddings = np.array(p2_embeddings)

        # Create trainer via composition
        embedding_dim = self.p1_embeddings.shape[1] + self.p2_embeddings.shape[1]
        self.trainer = ModelTrainer(embedding_dim, model_config, device)

        # Domain state
        self.ground_truth_payoffs = None
        self.pair_indices = None  # Will be set in compute_ground_truth_payoffs

    def compute_ground_truth_payoffs(self):
        """Compute ground truth payoffs for all P1-P2 agent pairs."""
        print(f"Computing ground truth payoffs for {len(self.p1_policies)} x {len(self.p2_policies)} agent pairs...")
        payoffs = []
        self.pair_indices = []

        for p1_idx, p1_policy in enumerate(tqdm(self.p1_policies, desc="P1 policies")):
            for p2_idx, p2_policy in enumerate(self.p2_policies):
                payoff = get_expected_payoffs(self.game, p1_policy, p2_policy)
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


class StatePayoffPredictorRefactored:
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


class ExploitabilityPredictorRefactored:
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
