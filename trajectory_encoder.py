"""
Trajectory Transformer Encoder for Multi-Agent RL Policy Embeddings.

This module implements a transformer-based encoder that learns policy representations
from behavioral trajectories rather than policy weights. The key insight is that
policies should be encoded based on how they actually play, not their weight values.

Approach:
1. Collect trajectories by rolling out each policy against a uniform random opponent
2. Embed each transition (state, action, reward) into a vector
3. Process the sequence of transition embeddings through a Transformer
4. Train using contrastive learning to ensure similar behaviors get similar embeddings
5. Use frozen encoder for downstream task prediction

This provides game-agnostic behavioral representations that transfer across tasks.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pyspiel
from open_spiel.python import policy

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import PPOAgentPolicy, get_device_string


# ==============================================================================
# Trajectory Utilities
# ==============================================================================

def collect_trajectory_transitions(
    game: pyspiel.Game,
    policy_agent: PPOAgent,
    player_id: int = 0,
    num_transitions: int = 200,
    opponent_policy: Optional[policy.Policy] = None,
    opponent_pool: Optional[List[PPOAgent]] = None,
) -> List[Tuple[torch.Tensor, int, float]]:
    """
    Collect trajectory transitions by rolling out a policy against an opponent.

    Args:
        game: OpenSpiel game instance
        policy_agent: PPOAgent to collect trajectories from
        player_id: Which player the policy_agent controls (0 or 1)
        num_transitions: Number of transitions to collect
        opponent_policy: Opponent policy (defaults to uniform random if opponent_pool not provided)
        opponent_pool: Optional pool of PPOAgents to randomly sample opponent from

    Returns:
        List of (state_tensor, action, reward) tuples
    """
    if opponent_pool is not None:
        # Randomly select an opponent from the pool
        opponent_agent = random.choice(opponent_pool)
        opponent_policy = PPOAgentPolicy(game, opponent_agent, 1 - player_id, use_observation=False)
    elif opponent_policy is None:
        opponent_policy = policy.UniformRandomPolicy(game)

    # Create policies for both players
    agent_policy = PPOAgentPolicy(game, policy_agent, player_id, use_observation=False)
    if player_id == 0:
        policies = [agent_policy, opponent_policy]
    else:
        policies = [opponent_policy, agent_policy]

    transitions = []

    while len(transitions) < num_transitions:
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.is_chance_node():
                # Sample chance node
                outcomes, probs = zip(*state.chance_outcomes())
                action = outcomes[torch.multinomial(torch.tensor(probs), 1).item()]
                state.apply_action(action)
            else:
                current_player = state.current_player()

                # Get state representation BEFORE taking action
                info_state = torch.tensor(
                    state.information_state_tensor(current_player),
                    dtype=torch.float32
                )

                # Get action from appropriate policy
                action_probs = policies[current_player].action_probabilities(state)
                actions = list(action_probs.keys())
                probs = list(action_probs.values())
                action = actions[torch.multinomial(torch.tensor(probs), 1).item()]

                # Store transition only for the agent we're encoding
                if current_player == player_id:
                    # Get reward after taking action
                    state.apply_action(action)

                    # Reward is from the perspective of player_id
                    if state.is_terminal():
                        reward = state.returns()[player_id]
                    else:
                        # For non-terminal states, reward is 0 (game-dependent)
                        reward = 0.0

                    transitions.append((info_state, action, reward))
                else:
                    state.apply_action(action)

            # Break if we have enough transitions
            if len(transitions) >= num_transitions:
                break

    return transitions[:num_transitions]


def normalize_transitions(
    transitions: List[Tuple[torch.Tensor, int, float]],
    state_min: Optional[torch.Tensor] = None,
    state_max: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[torch.Tensor, int, float]], torch.Tensor, torch.Tensor]:
    """
    Normalize state features to [-1, 1] range and rewards to [0, 1].

    Args:
        transitions: List of (state, action, reward) tuples
        state_min: Precomputed min values for normalization (optional)
        state_max: Precomputed max values for normalization (optional)

    Returns:
        Normalized transitions and (state_min, state_max) for later use
    """
    if not transitions:
        return transitions, None, None

    states = torch.stack([t[0] for t in transitions])

    if state_min is None:
        state_min = states.min(dim=0)[0]
    if state_max is None:
        state_max = states.max(dim=0)[0]

    # Avoid division by zero
    state_range = state_max - state_min
    state_range = torch.where(state_range < 1e-8, torch.ones_like(state_range), state_range)

    normalized = []
    for state, action, reward in transitions:
        # Normalize state to [-1, 1]
        norm_state = 2 * (state - state_min) / state_range - 1
        # Normalize reward (assuming rewards are in a known range, e.g., [-1, 1])
        norm_reward = (reward + 1) / 2  # Map [-1, 1] to [0, 1]
        normalized.append((norm_state, action, norm_reward))

    return normalized, state_min, state_max


# ==============================================================================
# Positional Encoding
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ==============================================================================
# Trajectory Transformer Encoder
# ==============================================================================

class TrajectoryEncoder(nn.Module):
    """
    Transformer-based encoder that processes behavioral trajectories.

    Architecture:
    1. Input projection: (state_dim + 1 + 1) -> hidden_dim
       - state_dim: dimension of state features
       - +1 for action (embedded)
       - +1 for reward
    2. Positional encoding
    3. Transformer encoder layers
    4. CLS token pooling to get fixed-size embedding
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Action embedding layer
        self.action_embedding = nn.Embedding(action_dim, hidden_dim // 4)

        # Input projection: state + action_embedding + reward -> hidden_dim
        transition_dim = state_dim + (hidden_dim // 4) + 1
        self.input_projection = nn.Linear(transition_dim, hidden_dim)

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm for final output
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        transitions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process trajectory transitions to get policy embedding.

        Args:
            transitions: Tensor of shape (batch, seq_len, transition_dim)
                        where transition_dim = state_dim + 1 (action) + 1 (reward)
            lengths: Optional tensor of actual sequence lengths for padding mask

        Returns:
            embeddings: Tensor of shape (batch, hidden_dim)
        """
        batch_size, seq_len, _ = transitions.shape

        # Split transitions into components
        states = transitions[:, :, :self.state_dim]  # (batch, seq_len, state_dim)
        actions = transitions[:, :, self.state_dim].long()  # (batch, seq_len)
        rewards = transitions[:, :, self.state_dim + 1:self.state_dim + 2]  # (batch, seq_len, 1)

        # Embed actions
        action_embeds = self.action_embedding(actions)  # (batch, seq_len, embed_dim)

        # Concatenate state, action embedding, and reward
        combined = torch.cat([states, action_embeds, rewards], dim=-1)  # (batch, seq_len, transition_dim)

        # Project to hidden dimension
        x = self.input_projection(combined)  # (batch, seq_len, hidden_dim)

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # (1, batch, hidden_dim)
        x = x.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=0)  # (seq_len + 1, batch, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create padding mask if lengths are provided
        src_key_padding_mask = None
        if lengths is not None:
            # Mask shape: (batch, seq_len + 1) where True means ignore
            max_len = seq_len + 1
            src_key_padding_mask = torch.arange(max_len, device=lengths.device).expand(
                batch_size, max_len
            ) > lengths.unsqueeze(1)

        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Use CLS token output as the embedding
        embedding = x[0]  # (batch, hidden_dim)
        embedding = self.layer_norm(embedding)

        return embedding


# ==============================================================================
# Contrastive Learning
# ==============================================================================

class ContrastiveLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.

    Given a batch of policies, we collect two different trajectories from each policy.
    Positive pairs: two trajectories from the same policy
    Negative pairs: trajectories from different policies
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of embeddings.

        Args:
            embeddings_1: (batch_size, embedding_dim) - first trajectory per policy
            embeddings_2: (batch_size, embedding_dim) - second trajectory per policy

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings_1.shape[0]

        # Normalize embeddings
        embeddings_1 = F.normalize(embeddings_1, p=2, dim=1)
        embeddings_2 = F.normalize(embeddings_2, p=2, dim=1)

        # Concatenate embeddings
        embeddings = torch.cat([embeddings_1, embeddings_2], dim=0)  # (2 * batch_size, dim)

        # Compute similarity matrix
        similarity_matrix = torch.mm(embeddings, embeddings.t())  # (2N, 2N)

        # Create mask for positive pairs
        # Positive pairs are at positions (i, i+N) and (i+N, i)
        mask = torch.zeros_like(similarity_matrix, dtype=torch.bool)
        for i in range(batch_size):
            mask[i, i + batch_size] = True
            mask[i + batch_size, i] = True

        # Mask out self-similarity (diagonal)
        mask.fill_diagonal_(False)

        # Compute logits
        similarity_matrix = similarity_matrix / self.temperature

        # For numerical stability, subtract max
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits)

        # Mask out diagonal
        exp_logits = exp_logits * (~torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device))

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()

        return loss


# ==============================================================================
# Dataset for Contrastive Learning
# ==============================================================================

class TrajectoryContrastiveDataset(Dataset):
    """
    Dataset that generates pairs of trajectories from the same policy for contrastive learning.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        policies: List[PPOAgent],
        num_transitions: int = 200,
        normalize: bool = True,
        device: str = "cpu",
        trajectories_per_policy: int = 10,
    ):
        self.game = game
        self.policies = policies
        self.num_transitions = num_transitions
        self.normalize = normalize
        self.device = device
        self.trajectories_per_policy = trajectories_per_policy

        self.state_dim = game.information_state_tensor_shape()[0]
        self.action_dim = game.num_distinct_actions()

        # Collect normalization statistics if needed
        self.state_min = None
        self.state_max = None
        if normalize:
            self._compute_normalization_stats()

        # Pre-collect and cache all trajectories
        print(f"\nPre-collecting {trajectories_per_policy} trajectories per policy for {len(policies)} policies...")
        self.cached_trajectories = []
        self._precollect_trajectories()

    def _compute_normalization_stats(self):
        """Compute min/max statistics for state normalization."""
        print("Computing normalization statistics...")
        all_states = []
        for i, policy_agent in enumerate(tqdm(self.policies[:10], desc="Sampling for stats")):
            transitions = collect_trajectory_transitions(
                self.game, policy_agent, player_id=0, num_transitions=50,
                opponent_pool=self.policies  # Use diverse opponents
            )
            states = torch.stack([t[0] for t in transitions])
            all_states.append(states)

        all_states = torch.cat(all_states, dim=0)
        self.state_min = all_states.min(dim=0)[0]
        self.state_max = all_states.max(dim=0)[0]

    def _precollect_trajectories(self):
        """Pre-collect all trajectories and cache them."""
        for policy_idx, policy_agent in enumerate(tqdm(self.policies, desc="Collecting trajectories")):
            policy_trajs = []
            for _ in range(self.trajectories_per_policy):
                transitions = collect_trajectory_transitions(
                    self.game, policy_agent, player_id=0, num_transitions=self.num_transitions,
                    opponent_pool=self.policies  # Use diverse opponents for each trajectory
                )

                # Normalize if requested
                if self.normalize:
                    transitions, _, _ = normalize_transitions(transitions, self.state_min, self.state_max)

                # Convert to tensor
                traj_tensor = self._transitions_to_tensor(transitions)
                policy_trajs.append(traj_tensor)

            self.cached_trajectories.append(policy_trajs)

    def __len__(self) -> int:
        return len(self.policies)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two different cached trajectory samples from the same policy.

        Returns:
            (trajectory_1, trajectory_2): Each of shape (num_transitions, state_dim + 2)
        """
        # Randomly select two different trajectories from the cached set
        policy_trajs = self.cached_trajectories[idx]
        indices = torch.randperm(len(policy_trajs))[:2]
        traj_1 = policy_trajs[indices[0]]
        traj_2 = policy_trajs[indices[1]]

        return traj_1, traj_2

    def _transitions_to_tensor(self, transitions: List[Tuple[torch.Tensor, int, float]]) -> torch.Tensor:
        """Convert list of transitions to a single tensor."""
        tensors = []
        for state, action, reward in transitions:
            # Concatenate state, action (as scalar), reward
            transition = torch.cat([
                state,
                torch.tensor([action], dtype=torch.float32),
                torch.tensor([reward], dtype=torch.float32),
            ])
            tensors.append(transition)
        return torch.stack(tensors)


# ==============================================================================
# Training
# ==============================================================================

@dataclass
class TrajectoryEncoderConfig:
    """Configuration for trajectory encoder training."""
    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # Training
    num_transitions: int = 200
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_split: float = 0.2  # Validation split (0.0 = no validation)
    lr_scheduler: str = "cosine"  # Learning rate scheduler: "cosine", "step", or "none"

    # Contrastive learning
    temperature: float = 0.07

    # Data
    normalize: bool = True

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class TrajectoryEncoderTrainer:
    """Trainer for trajectory encoder with contrastive learning."""

    def __init__(
        self,
        config: TrajectoryEncoderConfig,
        game: pyspiel.Game,
        policies: List[PPOAgent],
    ):
        self.config = config
        self.game = game
        self.policies = policies

        # Train/val split
        if config.val_split > 0:
            split_idx = int(len(policies) * (1 - config.val_split))
            train_policies = policies[:split_idx]
            val_policies = policies[split_idx:]
            print(f"Train/val split: {len(train_policies)} train, {len(val_policies)} val policies")
        else:
            train_policies = policies
            val_policies = []
            print(f"No validation split: using all {len(train_policies)} policies for training")

        # Initialize model
        state_dim = game.information_state_tensor_shape()[0]
        action_dim = game.num_distinct_actions()

        self.model = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
        ).to(config.device)

        # Initialize loss and optimizer
        self.criterion = ContrastiveLoss(temperature=config.temperature)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Initialize learning rate scheduler
        if config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=1e-6
            )
            print(f"Using cosine annealing LR scheduler (eta_min=1e-6)")
        elif config.lr_scheduler == "step":
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.epochs // 3,
                gamma=0.1
            )
            print(f"Using step LR scheduler (step every {config.epochs // 3} epochs)")
        else:
            self.scheduler = None
            print(f"No LR scheduler")

        # Initialize train dataset and dataloader
        self.train_dataset = TrajectoryContrastiveDataset(
            game=game,
            policies=train_policies,
            num_transitions=config.num_transitions,
            normalize=config.normalize,
            device=config.device,
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Initialize val dataset and dataloader
        if val_policies:
            self.val_dataset = TrajectoryContrastiveDataset(
                game=game,
                policies=val_policies,
                num_transitions=config.num_transitions,
                normalize=config.normalize,
                device=config.device,
            )
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None

    def train(self) -> Tuple[TrajectoryEncoder, dict]:
        """Train the trajectory encoder and return model and loss history."""
        torch.manual_seed(self.config.seed)

        train_history = []
        val_history = []
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        best_model_state = None
        best_epoch = 0

        print(f"\nTraining trajectory encoder for {self.config.epochs} epochs...")
        print(f"Train dataset size: {len(self.train_dataset)} policies")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.config.device}")
        if self.val_dataloader:
            print(f"Val dataset size: {len(self.val_dataset)} policies")

        progress_bar = tqdm(range(1, self.config.epochs + 1), desc="Training")

        for epoch in progress_bar:
            # Training
            train_loss = self._train_epoch()
            train_history.append(train_loss)

            # Validation
            if self.val_dataloader:
                val_loss = self._validate_epoch()
                val_history.append(val_loss)

                # Track best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch

                # Update progress bar
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.lr
                progress_bar.set_postfix({
                    'train': f'{train_loss:.4f}',
                    'val': f'{val_loss:.4f}',
                    'lr': f'{current_lr:.6f}'
                })

                # Print every 10 epochs or if new best
                if epoch % 10 == 0 or val_loss == best_val_loss:
                    print(f"Epoch {epoch}/{self.config.epochs}: "
                          f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                          f"LR={current_lr:.6f}" +
                          (" [NEW BEST]" if val_loss == best_val_loss else ""))
            else:
                # No validation - track best based on training loss
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch

                progress_bar.set_postfix(loss=f"{train_loss:.4f}")
                print(f"Epoch {epoch}/{self.config.epochs}: Loss = {train_loss:.4f}")

            # Step learning rate scheduler
            if self.scheduler:
                self.scheduler.step()

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if self.val_dataloader:
                print(f"\nLoaded best model from epoch {best_epoch} "
                      f"(val_loss={best_val_loss:.4f}, train_loss={best_train_loss:.4f})")
            else:
                print(f"\nLoaded best model from epoch {best_epoch} (loss={best_train_loss:.4f})")

        history = {
            'train': train_history,
            'val': val_history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss if self.val_dataloader else None,
            'best_train_loss': best_train_loss,
        }

        return self.model, history

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for traj_1, traj_2 in self.train_dataloader:
            traj_1 = traj_1.to(self.config.device)
            traj_2 = traj_2.to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            embedding_1 = self.model(traj_1)
            embedding_2 = self.model(traj_2)

            # Compute loss
            loss = self.criterion(embedding_1, embedding_2)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for traj_1, traj_2 in self.val_dataloader:
                traj_1 = traj_1.to(self.config.device)
                traj_2 = traj_2.to(self.config.device)

                # Forward pass
                embedding_1 = self.model(traj_1)
                embedding_2 = self.model(traj_2)

                # Compute loss
                loss = self.criterion(embedding_1, embedding_2)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)


# ==============================================================================
# Encoder Adapter
# ==============================================================================

class TrajectoryEncoderAdapter:
    """
    Adapter to match the existing encoder interface used in downstream tasks.

    This wraps a trained TrajectoryEncoder and provides the get_encoder() method
    that returns a function mapping PPOAgent -> embedding.
    """

    def __init__(
        self,
        model: TrajectoryEncoder,
        game: pyspiel.Game,
        config: TrajectoryEncoderConfig,
    ):
        self.model = model
        self.game = game
        self.config = config

        # Store normalization statistics from training
        self.state_min = None
        self.state_max = None

    def set_normalization_stats(self, state_min: torch.Tensor, state_max: torch.Tensor):
        """Set normalization statistics (should be called after training)."""
        self.state_min = state_min
        self.state_max = state_max

    def get_encoder(self, device: str = "cpu") -> Callable[[PPOAgent], torch.Tensor]:
        """
        Get encoder function that maps PPOAgent to embedding.

        Args:
            device: Device to run encoder on

        Returns:
            encoder_fn: Function PPOAgent -> torch.Tensor (embedding)
        """
        self.model.eval()
        self.model.to(device)

        def encoder_fn(policy: PPOAgent) -> torch.Tensor:
            """Encode a policy into a behavioral embedding."""
            with torch.no_grad():
                # Collect trajectory
                transitions = collect_trajectory_transitions(
                    self.game,
                    policy,
                    player_id=0,
                    num_transitions=self.config.num_transitions,
                )

                # Normalize if needed
                if self.config.normalize and self.state_min is not None:
                    transitions, _, _ = normalize_transitions(
                        transitions, self.state_min, self.state_max
                    )

                # Convert to tensor
                traj_tensors = []
                for state, action, reward in transitions:
                    transition = torch.cat([
                        state,
                        torch.tensor([action], dtype=torch.float32),
                        torch.tensor([reward], dtype=torch.float32),
                    ])
                    traj_tensors.append(transition)

                trajectory = torch.stack(traj_tensors).unsqueeze(0).to(device)  # (1, seq_len, dim)

                # Get embedding
                embedding = self.model(trajectory)
                return embedding.squeeze(0).cpu()

        return encoder_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train trajectory transformer encoder with contrastive learning."
    )

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--num-transitions", type=int, default=200, help="Transitions per trajectory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive learning temperature")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split (0.0 = no validation)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step", "none"], help="Learning rate scheduler")

    # Data
    parser.add_argument("--use-psro", action="store_true", help="Use PSRO agents instead of random agents")
    parser.add_argument("--num-agents", type=int, default=50, help="Number of PPO agents to train on (only used if not using PSRO)")
    parser.add_argument("--ppo-hidden-size", type=int, default=256, help="Hidden size for PPO agents")
    parser.add_argument("--normalize", action="store_true", help="Normalize states and rewards")
    parser.add_argument("--shuffle-psro", action="store_true", help="Shuffle PSRO agents before using them")

    # System
    parser.add_argument("--game", type=str, default="kuhn_poker", help="OpenSpiel game name")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output", type=str, default="checkpoints/trajectory_encoder.pt", help="Output path")

    return parser.parse_args()


if __name__ == "__main__":
    from utils import make_diverse_random_kuhn_poker_layer_init
    from psro import load_ppo_agents_from_psro

    args = parse_args()

    # Set device
    device = args.device or get_device_string()
    print(f"Using device: {device}")

    # Load game
    game = pyspiel.load_game(args.game)
    info_state_shape = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    print(f"Game: {args.game}")
    print(f"State dimension: {info_state_shape[0]}")
    print(f"Action dimension: {num_actions}")

    # Create or load PPO agents
    if args.use_psro:
        print(f"\nLoading PSRO agents (hidden_size={args.ppo_hidden_size})...")
        ppo_agents = load_ppo_agents_from_psro(hidden_size=args.ppo_hidden_size, shuffle=args.shuffle_psro)
        print(f"Loaded {len(ppo_agents)} PSRO agents")

        # Optionally limit to num_agents if specified and less than available
        if args.num_agents > 0 and args.num_agents < len(ppo_agents):
            print(f"Using first {args.num_agents} PSRO agents")
            ppo_agents = ppo_agents[:args.num_agents]
    else:
        print(f"\nCreating {args.num_agents} diverse random PPO agents...")
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)
        ppo_agents = [
            PPOAgent(num_actions, info_state_shape, "cpu", layer_init, args.ppo_hidden_size)
            for _ in range(args.num_agents)
        ]

    # Create config
    config = TrajectoryEncoderConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        num_transitions=args.num_transitions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        val_split=args.val_split,
        lr_scheduler=args.lr_scheduler,
        normalize=args.normalize,
        device=device,
        seed=args.seed,
    )

    # Train
    trainer = TrajectoryEncoderTrainer(config, game, ppo_agents)
    model, history = trainer.train()

    # Save model
    print(f"\nSaving model to {args.output}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'state_min': trainer.train_dataset.state_min,
        'state_max': trainer.train_dataset.state_max,
    }, args.output)

    print("\nTraining complete!")
    if history['val']:
        print(f"Best model: Epoch {history['best_epoch']} - "
              f"Train loss: {history['best_train_loss']:.4f}, "
              f"Val loss: {history['best_val_loss']:.4f}")
    else:
        print(f"Best model: Epoch {history['best_epoch']} - "
              f"Train loss: {history['best_train_loss']:.4f}")
