"""
Downstream task: Train a model to predict expected payoff between PPO agents and a fixed policy.

PayoffPredictor:
1. Takes PPO agents and encodes them using a custom encoder function
2. Trains a neural network to predict expected payoffs against a fixed opponent policy
3. Can be used to evaluate agent performance without running full game simulations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Callable, Any, Optional
from tqdm import tqdm

from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode
from open_spiel.python.policy import Policy
from policy_repr.utils import PPOAgentPolicy, get_expected_payoffs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_state_tensor(state):
    """
    Concatenate both player's info for full state representation.
    """
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
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

class PayoffPredictor:
    """
    Trains a model to predict expected payoff between PPO agents and a fixed policy.

    Pipeline:
    - Encoding PPO agents into embeddings
    - Generating training data using ground-truth payoff calculations
    - Training a regression model to predict payoffs from embeddings
    - Evaluating the trained model
    """

    def __init__(
        self,
        game,
        p1_policies: list[Policy],
        p2_policies: list[Policy],
        p1_embeddings: list[np.ndarray],
        p2_embeddings: list[np.ndarray],
        hidden_dims: List[int],
        dropout: float,
        device: str = "cpu"
    ):
        """
        Initialize the PayoffPredictor.

        Args:
            game: OpenSpiel game instance
            p1_agents: List of P1 agents
            p2_agents: List of P2 agents (can be a single-element list for fixed opponent)
            p1_encoder_fn: Function that maps a P1 agent to an embedding vector
            p2_encoder_fn: Function that maps a P2 agent to an embedding vector
            hidden_dims: Hidden layer dimensions for the predictor network
            dropout: Dropout rate for the predictor network
            device: cpu or cuda
        """
        self.game = game
        self.p1_policies = p1_policies
        self.p2_policies = p2_policies
        self.device = device
        self.p1_embeddings = np.array(p1_embeddings)
        self.p2_embeddings = np.array(p2_embeddings)
        # Concatenate P1 and P2 embeddings for input
        self.embedding_dim = self.p1_embeddings.shape[1] + self.p2_embeddings.shape[1]
        # Initialize the predictor model
        self.model = PayoffModel(
            self.embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.ground_truth_payoffs = None
        self.train_indices = None
        self.val_indices = None

    def compute_ground_truth_payoffs(self):
        """Compute ground truth payoffs for all P1-P2 agent pairs."""

        print(f"Computing ground truth payoffs for {len(self.p1_policies)} x {len(self.p2_policies)} agent pairs...")
        payoffs = []
        pair_indices = []  # Store (p1_idx, p2_idx) for each payoff

        for p1_idx, p1_policy in enumerate(tqdm(self.p1_policies, desc="P1 policies")):
            for p2_idx, p2_policy in enumerate(self.p2_policies):
                payoff = get_expected_payoffs(self.game, p1_policy, p2_policy)
                payoffs.append(payoff)
                pair_indices.append((p1_idx, p2_idx))

        self.ground_truth_payoffs = np.array(payoffs)
        self.pair_indices = pair_indices
        return self.ground_truth_payoffs

    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        verbose: bool = True
    ):
        """
        Train the payoff predictor model.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            seed: Random seed for reproducibility

        Returns:
            dict: Training history with train/val losses
        """
        # Compute ground truth if not already done
        if self.ground_truth_payoffs is None:
            self.compute_ground_truth_payoffs()

        # Create concatenated embeddings for all pairs
        print("Creating concatenated embeddings for pairs...")
        embeddings = []
        for p1_idx, p2_idx in self.pair_indices:
            p1_emb = self.p1_embeddings[p1_idx]
            p2_emb = self.p2_embeddings[p2_idx]
            concat_emb = np.concatenate([p1_emb, p2_emb])
            embeddings.append(concat_emb)
        embeddings = np.array(embeddings)

        # Split P1 agents to avoid leakage
        n_p1_policies = len(self.p1_policies)
        n_val_p1_policies = max(1, int(n_p1_policies * validation_split))  # Ensure at least 1 val agent
        if n_val_p1_policies >= n_p1_policies:
            raise ValueError(f"Not enough P1 agents ({n_p1_policies}) to create train/val split. Need at least 2.")
        p1_policy_indices = np.random.permutation(n_p1_policies)
        val_p1_policy_set = set(p1_policy_indices[:n_val_p1_policies])
        train_p1_policy_set = set(p1_policy_indices[n_val_p1_policies:])

        # Check if we should split P2 agents or use all P2 agents in both sets
        n_p2_policies = len(self.p2_policies)
        if n_p2_policies == 1:
            # Single fixed opponent: use in both train and val
            print("Single P2 agent detected - using same opponent for train and val")
            train_indices = np.array([i for i, (p1_idx, p2_idx) in enumerate(self.pair_indices)
                                      if p1_idx in train_p1_policy_set], dtype=int)
            val_indices = np.array([i for i, (p1_idx, p2_idx) in enumerate(self.pair_indices)
                                    if p1_idx in val_p1_policy_set], dtype=int)
            print(f"Split summary: {len(train_indices)} training pairs, {len(val_indices)} validation pairs")
            print(f"Training P1 agents: {len(train_p1_policy_set)}, Validation P1 agents: {len(val_p1_policy_set)}")
        else:
            # Multiple P2 agents: split them too
            n_val_p2_policies = max(1, int(n_p2_policies * validation_split))
            if n_val_p2_policies >= n_p2_policies:
                raise ValueError(f"Not enough P2 agents ({n_p2_policies}) to create train/val split. Need at least 2.")
            p2_policy_indices = np.random.permutation(n_p2_policies)
            val_p2_policy_set = set(p2_policy_indices[:n_val_p2_policies])
            train_p2_policy_set = set(p2_policy_indices[n_val_p2_policies:])

            # A pair is in validation only if BOTH P1 and P2 are in validation sets
            # A pair is in training only if BOTH P1 and P2 are in training sets
            train_indices = np.array([i for i, (p1_idx, p2_idx) in enumerate(self.pair_indices)
                                      if p1_idx in train_p1_policy_set and p2_idx in train_p2_policy_set], dtype=int)
            val_indices = np.array([i for i, (p1_idx, p2_idx) in enumerate(self.pair_indices)
                                    if p1_idx in val_p1_policy_set and p2_idx in val_p2_policy_set], dtype=int)

            print(f"Split summary: {len(train_indices)} training pairs, {len(val_indices)} validation pairs")
            print(f"Training P1 agents: {len(train_p1_policy_set)}, Validation P1 agents: {len(val_p1_policy_set)}")
            print(f"Training P2 agents: {len(train_p2_policy_set)}, Validation P2 agents: {len(val_p2_policy_set)}")

        # Store indices for later evaluation
        self.train_indices = train_indices
        self.val_indices = val_indices

        X_train = torch.FloatTensor(embeddings[train_indices]).to(self.device)
        y_train = torch.FloatTensor(self.ground_truth_payoffs[train_indices]).to(self.device)
        X_val = torch.FloatTensor(embeddings[val_indices]).to(self.device)
        y_val = torch.FloatTensor(self.ground_truth_payoffs[val_indices]).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()

            # Shuffle training data
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            train_losses = []

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(val_predictions, y_val)

            # Record history
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = val_loss.item()
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {epoch_train_loss:.6f}, "
                      f"Val Loss: {epoch_val_loss:.6f}")

        return history

    def predict(self, p1_embedding: np.ndarray, p2_embedding: np.ndarray):
        """
        Predict the expected payoff for a P1-P2 agent pair.

        Args:
            p1_agent: A P1 agent object
            p2_agent: A P2 agent object

        Returns:
            float: Predicted expected payoff
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(p1_embedding, torch.Tensor):
                p1_embedding = p1_embedding.detach().cpu().numpy()
            if isinstance(p2_embedding, torch.Tensor):
                p2_embedding = p2_embedding.detach().cpu().numpy()
            # Concatenate embeddings
            concat_embedding = np.concatenate([p1_embedding, p2_embedding])
            embedding_tensor = torch.FloatTensor(concat_embedding).unsqueeze(0).to(self.device)
            prediction = self.model(embedding_tensor)
            return prediction.item()

    def evaluate(self, eval_set: str = "all"):
        """
        Evaluate the model on agent pairs.

        Args:
            eval_set: Which set to evaluate on: "all", "train", or "val".

        Returns:
            dict: Evaluation metrics (MSE, MAE, R2)
        """
        # Determine which pairs to evaluate
        if eval_set == "train" and self.train_indices is not None:
            indices = self.train_indices
            test_payoffs = self.ground_truth_payoffs[indices]
            test_pairs = [self.pair_indices[i] for i in indices]
            print(f"Evaluating on training set ({len(indices)} pairs)...")
        elif eval_set == "val" and self.val_indices is not None:
            indices = self.val_indices
            test_payoffs = self.ground_truth_payoffs[indices]
            test_pairs = [self.pair_indices[i] for i in indices]
            print(f"Evaluating on validation set ({len(indices)} pairs)...")
        elif eval_set == "all":
            test_payoffs = self.ground_truth_payoffs
            test_pairs = self.pair_indices
            print(f"Evaluating on all pairs ({len(test_pairs)} pairs)...")
        else:
            raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'all', 'train', or 'val'. "
                           f"Also ensure train() has been called to create the split.")

        # Get predictions for all pairs
        print("Computing predictions...")
        predictions = []
        for p1_idx, p2_idx in tqdm(test_pairs):
            p1_embedding = self.p1_embeddings[p1_idx]
            p2_embedding = self.p2_embeddings[p2_idx]
            pred = self.predict(p1_embedding, p2_embedding)
            predictions.append(pred)
        predictions = np.array(predictions)

        # Compute metrics
        mse = np.mean((predictions - test_payoffs) ** 2)
        mae = np.mean(np.abs(predictions - test_payoffs))

        # R2 score
        ss_res = np.sum((test_payoffs - predictions) ** 2)
        ss_tot = np.sum((test_payoffs - np.mean(test_payoffs)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'ground_truth': test_payoffs
        }

        return metrics

    def save(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")


class StatePayoffPredictor(PayoffPredictor):
    """
    Payoff predictor that conditions on game states.

    Predicts payoffs for (P1_agent, P2_agent, state) triples instead of just agent pairs.
    """

    def __init__(
        self,
        game,
        p1_agents: List[Any],
        p2_agents: List[Any],
        p1_encoder_fn: Callable,
        p2_encoder_fn: Callable,
        state_sampler: Callable,
        num_states_per_pair: int,
        hidden_dims: List[int],
        dropout: float,
        device: str = "cpu"
    ):
        """
        Initialize the StatePayoffPredictor.

        Args:
            game: OpenSpiel game instance
            p1_agents: List of P1 agents
            p2_agents: List of P2 agents
            p1_encoder_fn: Function that maps a P1 agent to an embedding vector
            p2_encoder_fn: Function that maps a P2 agent to an embedding vector
            state_sampler: Function that generates states from the game (takes game as input, returns list of states)
            num_states_per_pair: Number of states to sample per agent pair
            hidden_dims: Hidden layer dimensions for the predictor network
            dropout: Dropout rate for the predictor network
            device: cpu or cuda
        """
        self.state_sampler = state_sampler
        self.num_states_per_pair = num_states_per_pair
        self.states = None
        self.state_tensors = None
        self.p1_agents = p1_agents
        self.p2_agents = p2_agents
        self.p1_encoder_fn = p1_encoder_fn
        self.p2_encoder_fn = p2_encoder_fn

        # Encode agents to get embeddings
        print("Encoding P1 agents...")
        p1_embeddings = [p1_encoder_fn(agent).cpu().numpy() for agent in tqdm(p1_agents)]
        p1_embeddings = np.array(p1_embeddings)

        print("Encoding P2 agents...")
        p2_embeddings = [p2_encoder_fn(agent).cpu().numpy() for agent in tqdm(p2_agents)]
        p2_embeddings = np.array(p2_embeddings)

        # Convert agents to policies
        from policy_repr.utils import PPOAgentPolicy
        p1_policies = [PPOAgentPolicy(game, agent, 0, use_observation=False) for agent in p1_agents]
        p2_policies = [PPOAgentPolicy(game, agent, 1, use_observation=False) for agent in p2_agents]

        # Call parent init with embeddings and policies
        super().__init__(
            game=game,
            p1_policies=p1_policies,
            p2_policies=p2_policies,
            p1_embeddings=p1_embeddings,
            p2_embeddings=p2_embeddings,
            hidden_dims=hidden_dims,
            dropout=dropout,
            device=device
        )

        # Sample and encode states
        print(f"Sampling {num_states_per_pair} states...")
        self.states = state_sampler(game, num_states_per_pair)

        print("Encoding states...")
        self.state_tensors = []
        for state in tqdm(self.states):
            # Use state tensor as embedding
            state_tensor = torch.FloatTensor(get_state_tensor(state))
            self.state_tensors.append(state_tensor.numpy())
        self.state_tensors = np.array(self.state_tensors)

        # Update embedding dimension to include state
        state_dim = self.state_tensors.shape[1]
        self.embedding_dim = self.p1_embeddings.shape[1] + self.p2_embeddings.shape[1] + state_dim

        # Reinitialize model with correct embedding dimension
        self.model = PayoffModel(
            self.embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(device)

    def compute_ground_truth_payoffs(self):
        """Compute ground truth payoffs for all (P1, P2, state) triples."""

        print(f"Computing ground truth payoffs for {len(self.p1_agents)} x {len(self.p2_agents)} x {len(self.states)} triples...")
        payoffs = []
        triple_indices = []  # Store (p1_idx, p2_idx, state_idx) for each payoff

        for p1_idx, p1_agent in enumerate(tqdm(self.p1_agents, desc="P1 agents")):
            for p2_idx, p2_agent in enumerate(self.p2_agents):
                for state_idx, state in enumerate(self.states):
                    # Compute payoff starting from this state
                    payoff = self._compute_payoff_from_state(state, p1_agent, p2_agent)
                    payoffs.append(payoff)
                    triple_indices.append((p1_idx, p2_idx, state_idx))

        self.ground_truth_payoffs = np.array(payoffs)
        self.triple_indices = triple_indices
        return self.ground_truth_payoffs

    def _compute_payoff_from_state(self, state, p1_agent, p2_agent):
        """Compute expected payoff starting from a given state."""

        # Wrap agents in policies
        p0_policy = PPOAgentPolicy(self.game, p1_agent, 0, False)

        if hasattr(p2_agent, 'actor'):
            # PPOAgent
            p1_policy = PPOAgentPolicy(self.game, p2_agent, 1, False)
        else:
            # Already a policy
            p1_policy = p2_agent

        policies = [p0_policy, p1_policy]

        # Sample episodes starting from this state
        payoffs = []
        for _ in range(100):  # 100 episodes per state
            payoff = sample_episode(state.clone(), policies)[0]
            payoffs.append(payoff)

        return np.mean(payoffs)

    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        verbose: bool = True,
    ):
        """
        Train the payoff predictor model.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
            seed: Random seed for reproducibility

        Returns:
            dict: Training history with train/val losses
        """
        # Compute ground truth if not already done
        if self.ground_truth_payoffs is None:
            self.compute_ground_truth_payoffs()

        # Create concatenated embeddings for all triples
        print("Creating concatenated embeddings for triples...")
        embeddings = []
        for p1_idx, p2_idx, state_idx in self.triple_indices:
            p1_emb = self.p1_embeddings[p1_idx]
            p2_emb = self.p2_embeddings[p2_idx]
            state_emb = self.state_tensors[state_idx]
            concat_emb = np.concatenate([p1_emb, p2_emb, state_emb])
            embeddings.append(concat_emb)
        embeddings = np.array(embeddings)

        # Split at P1 agent, P2 agent, AND state level to avoid leakage
        n_p1_agents = len(self.p1_agents)
        n_val_p1_agents = int(n_p1_agents * validation_split)
        p1_agent_indices = np.random.permutation(n_p1_agents)
        val_p1_agent_set = set(p1_agent_indices[:n_val_p1_agents])
        train_p1_agent_set = set(p1_agent_indices[n_val_p1_agents:])

        n_p2_agents = len(self.p2_agents)
        n_val_p2_agents = int(n_p2_agents * validation_split)
        p2_agent_indices = np.random.permutation(n_p2_agents)
        val_p2_agent_set = set(p2_agent_indices[:n_val_p2_agents])
        train_p2_agent_set = set(p2_agent_indices[n_val_p2_agents:])

        n_states = len(self.states)
        n_val_states = int(n_states * validation_split)
        state_indices = np.random.permutation(n_states)
        val_state_set = set(state_indices[:n_val_states])
        train_state_set = set(state_indices[n_val_states:])

        # A triple is in validation only if ALL of P1, P2, and state are in validation sets
        # A triple is in training only if ALL of P1, P2, and state are in training sets
        train_indices = np.array([i for i, (p1_idx, p2_idx, state_idx) in enumerate(self.triple_indices)
                                  if p1_idx in train_p1_agent_set and p2_idx in train_p2_agent_set and state_idx in train_state_set], dtype=int)
        val_indices = np.array([i for i, (p1_idx, p2_idx, state_idx) in enumerate(self.triple_indices)
                                if p1_idx in val_p1_agent_set and p2_idx in val_p2_agent_set and state_idx in val_state_set], dtype=int)

        print(f"Split summary: {len(train_indices)} training triples, {len(val_indices)} validation triples")
        print(f"Training P1 agents: {len(train_p1_agent_set)}, Validation P1 agents: {len(val_p1_agent_set)}")
        print(f"Training P2 agents: {len(train_p2_agent_set)}, Validation P2 agents: {len(val_p2_agent_set)}")
        print(f"Training states: {len(train_state_set)}, Validation states: {len(val_state_set)}")

        # Store indices for later evaluation
        self.train_indices = train_indices
        self.val_indices = val_indices

        X_train = torch.FloatTensor(embeddings[train_indices]).to(self.device)
        y_train = torch.FloatTensor(self.ground_truth_payoffs[train_indices]).to(self.device)
        X_val = torch.FloatTensor(embeddings[val_indices]).to(self.device)
        y_val = torch.FloatTensor(self.ground_truth_payoffs[val_indices]).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()

            # Shuffle training data
            perm = torch.randperm(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            train_losses = []

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train_shuffled[i:i+batch_size]
                batch_y = y_train_shuffled[i:i+batch_size]

                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)

                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val)
                val_loss = self.criterion(val_predictions, y_val)

            # Record history
            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = val_loss.item()
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {epoch_train_loss:.6f}, "
                      f"Val Loss: {epoch_val_loss:.6f}")

        return history

    def predict(self, p1_agent, p2_agent, state):
        """
        Predict the expected payoff for a (P1, P2, state) triple.

        Args:
            p1_agent: A P1 agent object
            p2_agent: A P2 agent object
            state: A game state

        Returns:
            float: Predicted expected payoff
        """
        self.model.eval()
        with torch.no_grad():
            # Encode P1 agent
            p1_embedding = self.p1_encoder_fn(p1_agent)
            if isinstance(p1_embedding, torch.Tensor):
                p1_embedding = p1_embedding.detach().cpu().numpy()

            # Encode P2 agent
            p2_embedding = self.p2_encoder_fn(p2_agent)
            if isinstance(p2_embedding, torch.Tensor):
                p2_embedding = p2_embedding.detach().cpu().numpy()

            # Encode state (both players' perspectives)
            state_tensor = torch.FloatTensor(get_state_tensor(state))
            state_embedding = state_tensor.numpy()

            # Concatenate embeddings
            concat_embedding = np.concatenate([p1_embedding, p2_embedding, state_embedding])
            embedding_tensor = torch.FloatTensor(concat_embedding).unsqueeze(0).to(self.device)
            prediction = self.model(embedding_tensor)
            return prediction.item()

    def evaluate(self, eval_set: str = "all"):
        """
        Evaluate the model on (agent, agent, state) triples.

        Args:
            eval_set: Which set to evaluate on: "all", "train", or "val".

        Returns:
            dict: Evaluation metrics (MSE, MAE, R2)
        """
        # Determine which triples to evaluate
        if eval_set == "train" and self.train_indices is not None:
            indices = self.train_indices
            test_payoffs = self.ground_truth_payoffs[indices]
            test_triples = [self.triple_indices[i] for i in indices]
            print(f"Evaluating on training set ({len(indices)} triples)...")
        elif eval_set == "val" and self.val_indices is not None:
            indices = self.val_indices
            test_payoffs = self.ground_truth_payoffs[indices]
            test_triples = [self.triple_indices[i] for i in indices]
            print(f"Evaluating on validation set ({len(indices)} triples)...")
        elif eval_set == "all":
            test_payoffs = self.ground_truth_payoffs
            test_triples = self.triple_indices
            print(f"Evaluating on all triples ({len(test_triples)} triples)...")
        else:
            raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'all', 'train', or 'val'. "
                           f"Also ensure train() has been called to create the split.")

        # Get predictions for all triples
        print("Computing predictions...")
        predictions = []
        for p1_idx, p2_idx, state_idx in tqdm(test_triples):
            p1_agent = self.p1_agents[p1_idx]
            p2_agent = self.p2_agents[p2_idx]
            state = self.states[state_idx]
            pred = self.predict(p1_agent, p2_agent, state)
            predictions.append(pred)
        predictions = np.array(predictions)

        # Compute metrics
        mse = np.mean((predictions - test_payoffs) ** 2)
        mae = np.mean(np.abs(predictions - test_payoffs))

        metrics = {
            'mse': mse,
            'mae': mae,
            'predictions': predictions,
            'ground_truth': test_payoffs
        }

        return metrics
