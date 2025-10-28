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
from typing import List, Callable, Any
from tqdm import tqdm

from utils import get_expected_payoffs


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

class PayoffModel(nn.Module):
    """Simple MLP to predict payoffs from agent embeddings."""

    def __init__(self, embedding_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()

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
        ppo_agents: List[Any],
        opponent_policy: Any,
        encoder_fn: Callable,
        hidden_dims: List[int],
        dropout: float,
        device: str = "cpu"
    ):
        """
        Initialize the PayoffPredictor.

        Args:
            game: OpenSpiel game instance
            ppo_agents: List of PPO agent objects
            opponent_policy: Opponent policy to evaluate against
            encoder_fn: Function that maps a PPO agent to an embedding vector
            hidden_dims: Hidden layer dimensions for the predictor network
            device: cpu or cuda
        """
        self.game = game
        self.ppo_agents = ppo_agents
        self.opponent_policy = opponent_policy
        self.encoder_fn = encoder_fn
        self.device = device

        # Extract embeddings and determine embedding dimension
        print("Encoding PPO agents...")
        self.embeddings = []
        for agent in tqdm(ppo_agents):
            embedding = encoder_fn(agent)
            # Convert to numpy if it's a torch tensor
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            self.embeddings.append(embedding)

        self.embeddings = np.array(self.embeddings)
        self.embedding_dim = self.embeddings.shape[1]

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
        """Compute ground truth payoffs for all agents."""

        print("Computing ground truth payoffs...")
        payoffs = []
        for agent in tqdm(self.ppo_agents):
            payoff = get_expected_payoffs(self.game, agent, self.opponent_policy)
            payoffs.append(payoff)
        self.ground_truth_payoffs = np.array(payoffs)
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

        Returns:
            dict: Training history with train/val losses
        """
        # Compute ground truth if not already done
        if self.ground_truth_payoffs is None:
            self.compute_ground_truth_payoffs()

        # Split into train/val
        n_samples = len(self.embeddings)
        n_val = int(n_samples * validation_split)
        indices = np.random.permutation(n_samples)

        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        # Store indices for later evaluation
        self.train_indices = train_indices
        self.val_indices = val_indices

        X_train = torch.FloatTensor(self.embeddings[train_indices]).to(self.device)
        y_train = torch.FloatTensor(self.ground_truth_payoffs[train_indices]).to(self.device)
        X_val = torch.FloatTensor(self.embeddings[val_indices]).to(self.device)
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

    def predict(self, ppo_agent):
        """
        Predict the expected payoff for a single PPO agent.

        Args:
            ppo_agent: A PPO agent object

        Returns:
            float: Predicted expected payoff
        """
        self.model.eval()
        with torch.no_grad():
            embedding = self.encoder_fn(ppo_agent)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            prediction = self.model(embedding_tensor)
            return prediction.item()

    def evaluate(self, test_agents: List[Any] = None, eval_set: str = "all"):
        """
        Evaluate the model on test agents.

        Args:
            test_agents: List of test agents. If None, uses agents based on eval_set.
            eval_set: Which set to evaluate on: "all", "train", or "val". Only used if test_agents is None.

        Returns:
            dict: Evaluation metrics (MSE, MAE, R2)
        """
        if test_agents is None:
            # Use stored train/val split if available
            if eval_set == "train" and self.train_indices is not None:
                indices = self.train_indices
                test_agents = [self.ppo_agents[i] for i in indices]
                test_payoffs = self.ground_truth_payoffs[indices]
                print(f"Evaluating on training set ({len(indices)} agents)...")
            elif eval_set == "val" and self.val_indices is not None:
                indices = self.val_indices
                test_agents = [self.ppo_agents[i] for i in indices]
                test_payoffs = self.ground_truth_payoffs[indices]
                print(f"Evaluating on validation set ({len(indices)} agents)...")
            elif eval_set == "all":
                test_agents = self.ppo_agents
                test_payoffs = self.ground_truth_payoffs
                print(f"Evaluating on all agents ({len(test_agents)} agents)...")
            else:
                raise ValueError(f"Invalid eval_set: {eval_set}. Must be 'all', 'train', or 'val'. "
                               f"Also ensure train() has been called to create the split.")
        else:
            print("Computing ground truth for test agents...")
            test_payoffs = np.array([
                get_expected_payoffs(self.game, agent, self.opponent_policy)
                for agent in tqdm(test_agents)
            ])

        # Get predictions
        print("Computing predictions...")
        predictions = np.array([self.predict(agent) for agent in tqdm(test_agents)])

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
