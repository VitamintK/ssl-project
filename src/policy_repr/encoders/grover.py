"""
Grover et al. (2018) — "Learning Policy Representations in Multiagent Systems"

Faithful implementation of the hybrid generative-discriminative model:
  - MLP encoder f_theta that averages (obs, action) pair embeddings across an episode
  - Conditional policy network pi_{phi,theta} for imitation
  - Hybrid loss: behavioral cloning + squared softmax triplet identification

Includes training infrastructure (config, dataset, trainer, adapter) to integrate
with the compare_encoders.py evaluation pipeline.
"""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pyspiel
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent

# Allow running this file directly (adds the src/ dir so `policy_repr` is importable)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_repr.encoders.trajectory import collect_trajectory_transitions, normalize_transitions


logger = logging.getLogger(__name__)

class GroverEncoder(nn.Module):
    """
    Representation function f_theta that maps an episode to a continuous embedding.
    As per the paper, it uses an MLP to process individual (observation, action) 
    pairs and averages them across the episode to form the final embedding.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 100, embed_dim: int = 100):
        super().__init__()
        # The MLP processes a single (obs, action) pair
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, episode_transitions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            episode_transitions: Tensor of shape (batch_size, seq_len, obs_dim + act_dim)
        Returns:
            Episode embeddings of shape (batch_size, embed_dim)
        """
        # Pass all transitions through the MLP
        intermediate_embeddings = self.mlp(episode_transitions) # (batch, seq_len, embed_dim)
        
        # Average across the sequence length (episode) to get the final embedding
        episode_embedding = intermediate_embeddings.mean(dim=1) # (batch, embed_dim)
        return episode_embedding


class ConditionalPolicy(nn.Module):
    """
    The conditional policy network \pi_{\phi, \theta} that predicts an agent's 
    actions given its current observation and its policy embedding.
    """
    def __init__(self, obs_dim: int, embed_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        # The input is the observation concatenated with the embedding
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim) 
        )

    def forward(self, obs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            embedding: Tensor of shape (batch_size, embed_dim). Will be expanded to match seq_len.
        Returns:
            Action logits (for discrete) or values (for continuous).
        """
        if obs.dim() == 3: # (batch, seq_len, obs_dim)
            seq_len = obs.size(1)
            # Expand embedding to match sequence length
            embedding = embedding.unsqueeze(1).expand(-1, seq_len, -1)
            
        x = torch.cat([obs, embedding], dim=-1)
        return self.mlp(x)


class GroverHybridLoss(nn.Module):
    """
    Implements the hybrid generative-discriminative objective (Eq. 3).
    """
    def __init__(self, lambda_weight: float = 0.1, is_continuous_action: bool = False):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.is_continuous_action = is_continuous_action

    def compute_imitation_loss(self, predicted_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Behavioral cloning loss (Negative Log Likelihood)."""
        if self.is_continuous_action:
            # For continuous control (e.g., RoboSumo), use MSE
            return F.mse_loss(predicted_actions, target_actions)
        else:
            # For discrete actions, use Cross Entropy
            # reshape for cross entropy: (batch * seq_len, act_dim)
            flat_preds = predicted_actions.reshape(-1, predicted_actions.size(-1))
            flat_targets = target_actions.reshape(-1).long()
            return F.cross_entropy(flat_preds, flat_targets)

    def compute_identification_loss(self, p_e: torch.Tensor, n_e: torch.Tensor, r_e: torch.Tensor) -> torch.Tensor:
        """
        Squared softmax triplet loss as defined in Eq. 2 of the paper.
        d = (1 + exp(||r_e - n_e||_2 - ||r_e - p_e||_2))^-2
        """
        # L2 distances
        dist_pos = torch.norm(r_e - p_e, p=2, dim=1)
        dist_neg = torch.norm(r_e - n_e, p=2, dim=1)
        
        # Apply the specific squared softmax formulation from the paper
        exp_term = torch.exp(dist_neg - dist_pos)
        id_loss = torch.pow(1.0 + exp_term, -2)
        
        return id_loss.mean()

    def forward(
        self, 
        encoder: GroverEncoder, 
        policy_net: ConditionalPolicy,
        e_pos: torch.Tensor, # e_+: positive episode (Agent I)
        e_ref: torch.Tensor, # e_*: reference episode (Agent I)
        e_neg: torch.Tensor, # e_-: negative episode (Agent J)
        obs_dim: int
    ) -> Tuple[torch.Tensor, dict]:
        
        # 1. Get embeddings
        p_e = encoder(e_pos)
        r_e = encoder(e_ref)
        n_e = encoder(e_neg)
        
        # 2. Discriminative Loss (Agent Identification)
        id_loss = self.compute_identification_loss(p_e, n_e, r_e)
        
        # 3. Generative Loss (Imitation)
        # Condition the policy on the reference embedding (r_e) 
        # to predict the actions in the positive episode (e_pos)
        pos_obs = e_pos[..., :obs_dim]
        pos_actions_target = e_pos[..., obs_dim:]
        
        predicted_actions = policy_net(pos_obs, r_e)
        im_loss = self.compute_imitation_loss(predicted_actions, pos_actions_target)
        
        # 4. Total Hybrid Loss
        total_loss = im_loss + (self.lambda_weight * id_loss)
        
        metrics = {
            "total_loss": total_loss.item(),
            "imitation_loss": im_loss.item(),
            "identification_loss": id_loss.item()
        }
        
        return total_loss, metrics


# ==============================================================================
# Training infrastructure
# ==============================================================================

@dataclass
class GroverConfig:
    """Configuration for the Grover encoder."""
    hidden_dim: int = 100
    embed_dim: int = 128
    policy_hidden_dim: int = 64
    lambda_weight: float = 0.1

    # Training
    num_transitions: int = 200
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_split: float = 0.2
    lr_scheduler: str = "cosine"

    # Data
    normalize: bool = True
    trajectories_per_policy: int = 50

    # System
    device: str = "cpu"
    seed: int = 0


def _collect_and_format_trajectory(
    game, policy_agent, player_id, num_transitions, opponent_pool,
    state_min, state_max, normalize, num_actions,
):
    """Collect a trajectory and format as (obs, one-hot action) pairs.

    Returns:
        obs_action: Tensor of shape (seq_len, obs_dim + act_dim)
        actions: Tensor of shape (seq_len,) — raw action indices for imitation loss
    """
    transitions = collect_trajectory_transitions(
        game, policy_agent, player_id=player_id,
        num_transitions=num_transitions,
        opponent_pool=opponent_pool,
    )
    if normalize and state_min is not None:
        transitions, _, _ = normalize_transitions(transitions, state_min, state_max)

    obs_list, action_list = [], []
    for state, action, _reward in transitions:
        one_hot = torch.zeros(num_actions)
        one_hot[action] = 1.0
        obs_action = torch.cat([state, one_hot])
        obs_list.append(obs_action)
        action_list.append(action)

    return torch.stack(obs_list), torch.tensor(action_list, dtype=torch.long)


class GroverTripletDataset(Dataset):
    """
    Dataset that yields (positive, reference, negative) episode triplets.

    Positive and reference episodes come from the same agent;
    the negative episode comes from a different agent.
    """

    def __init__(
        self, game, policies, num_transitions, normalize,
        trajectories_per_policy, num_actions, state_min=None, state_max=None,
    ):
        self.game = game
        self.num_actions = num_actions
        obs_dim = game.information_state_tensor_shape()[0]

        # Compute normalization stats if needed
        self.state_min = state_min
        self.state_max = state_max
        if normalize and state_min is None:
            print("Computing normalization statistics...")
            all_states = []
            for p in policies[:10]:
                trans = collect_trajectory_transitions(
                    game, p, player_id=0, num_transitions=50,
                    opponent_pool=policies,
                )
                all_states.append(torch.stack([t[0] for t in trans]))
            all_states = torch.cat(all_states, dim=0)
            self.state_min = all_states.min(dim=0)[0]
            self.state_max = all_states.max(dim=0)[0]

        # Pre-collect trajectories per agent: list of list of (obs_action, actions)
        print(f"Pre-collecting {trajectories_per_policy} trajectories for {len(policies)} policies...")
        self.agent_trajectories = []  # agent_trajectories[agent_id] = list of (obs_action_tensor, action_tensor)
        for policy in tqdm(policies, desc="Collecting trajectories"):
            trajs = []
            for _ in range(trajectories_per_policy):
                obs_action, actions = _collect_and_format_trajectory(
                    game, policy, player_id=0,
                    num_transitions=num_transitions,
                    opponent_pool=policies,
                    state_min=self.state_min, state_max=self.state_max,
                    normalize=normalize, num_actions=num_actions,
                )
                trajs.append((obs_action, actions))
            self.agent_trajectories.append(trajs)

        self.num_agents = len(policies)
        # Each sample is a triplet; we create num_agents * trajectories_per_policy samples
        self.length = self.num_agents * trajectories_per_policy

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        agent_id = idx % self.num_agents
        traj_idx = idx // self.num_agents

        # Reference and positive are two different trajectories from the same agent
        ref_idx = traj_idx
        pos_idx = (traj_idx + 1) % len(self.agent_trajectories[agent_id])

        # Negative is from a different agent
        neg_agent = agent_id
        while neg_agent == agent_id:
            neg_agent = random.randint(0, self.num_agents - 1)
        neg_idx = random.randint(0, len(self.agent_trajectories[neg_agent]) - 1)

        ref_obs_action, _ = self.agent_trajectories[agent_id][ref_idx]
        pos_obs_action, pos_actions = self.agent_trajectories[agent_id][pos_idx]
        neg_obs_action, _ = self.agent_trajectories[neg_agent][neg_idx]

        return ref_obs_action, pos_obs_action, neg_obs_action, pos_actions


class GroverTrainer:
    """Trainer for the Grover hybrid encoder."""

    def __init__(self, config: GroverConfig, game: pyspiel.Game, policies: List[PPOAgent]):
        self.config = config
        self.game = game

        obs_dim = game.information_state_tensor_shape()[0]
        num_actions = game.num_distinct_actions()

        # Train/val split
        split_idx = int(len(policies) * (1 - config.val_split))
        train_policies = policies[:split_idx]
        val_policies = policies[split_idx:]
        print(f"Train/val split: {len(train_policies)} / {len(val_policies)} policies")

        # Models
        self.encoder = GroverEncoder(
            obs_dim=obs_dim, act_dim=num_actions,
            hidden_dim=config.hidden_dim, embed_dim=config.embed_dim,
        ).to(config.device)

        self.policy_net = ConditionalPolicy(
            obs_dim=obs_dim, embed_dim=config.embed_dim,
            act_dim=num_actions, hidden_dim=config.policy_hidden_dim,
        ).to(config.device)

        self.loss_fn = GroverHybridLoss(
            lambda_weight=config.lambda_weight, is_continuous_action=False,
        )

        params = list(self.encoder.parameters()) + list(self.policy_net.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

        if config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)
        else:
            self.scheduler = None

        # Datasets
        self.train_dataset = GroverTripletDataset(
            game, train_policies,
            num_transitions=config.num_transitions,
            normalize=config.normalize,
            trajectories_per_policy=config.trajectories_per_policy,
            num_actions=num_actions,
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
        )

        if val_policies:
            self.val_dataset = GroverTripletDataset(
                game, val_policies,
                num_transitions=config.num_transitions,
                normalize=config.normalize,
                trajectories_per_policy=config.trajectories_per_policy,
                num_actions=num_actions,
                state_min=self.train_dataset.state_min,
                state_max=self.train_dataset.state_max,
            )
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=config.batch_size, shuffle=False,
            )
        else:
            self.val_dataset = None
            self.val_loader = None

        self.obs_dim = obs_dim

    def _run_epoch(self, loader, train: bool):
        self.encoder.train(train)
        self.policy_net.train(train)
        total_loss = 0.0
        n = 0
        for ref, pos, neg, pos_actions in loader:
            ref = ref.to(self.config.device)
            pos = pos.to(self.config.device)
            neg = neg.to(self.config.device)
            pos_actions = pos_actions.to(self.config.device)

            if train:
                self.optimizer.zero_grad()

            # Encode
            r_e = self.encoder(ref)
            p_e = self.encoder(pos)
            n_e = self.encoder(neg)

            # Identification loss
            id_loss = self.loss_fn.compute_identification_loss(p_e, n_e, r_e)

            # Imitation loss: predict actions in positive episode conditioned on reference embedding
            pos_obs = pos[..., :self.obs_dim]
            predicted_actions = self.policy_net(pos_obs, r_e)
            im_loss = self.loss_fn.compute_imitation_loss(predicted_actions, pos_actions)

            loss = im_loss + self.config.lambda_weight * id_loss

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def train(self) -> Tuple[GroverEncoder, dict]:
        torch.manual_seed(self.config.seed)

        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        history = {"train": [], "val": []}

        print(f"\nTraining Grover encoder for {self.config.epochs} epochs...")
        progress = tqdm(range(1, self.config.epochs + 1), desc="Training")

        for epoch in progress:
            train_loss = self._run_epoch(self.train_loader, train=True)
            history["train"].append(train_loss)

            if self.val_loader:
                with torch.no_grad():
                    val_loss = self._run_epoch(self.val_loader, train=False)
                history["val"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.encoder.state_dict().items()}
                    best_epoch = epoch

                progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
            else:
                progress.set_postfix(train=f"{train_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()

        if best_state:
            self.encoder.load_state_dict(best_state)
            print(f"Loaded best encoder from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        return self.encoder, history


class GroverAdapter:
    """Adapter matching the TrajectoryEncoderAdapter interface for downstream eval."""

    def __init__(
        self, model: GroverEncoder, game: pyspiel.Game,
        config: GroverConfig, policies: Optional[List[PPOAgent]] = None,
    ):
        self.model = model
        self.game = game
        self.config = config
        self.policies = policies
        self.state_min = None
        self.state_max = None
        self.num_actions = game.num_distinct_actions()
        self.obs_dim = game.information_state_tensor_shape()[0]

    def set_normalization_stats(self, state_min, state_max):
        self.state_min = state_min
        self.state_max = state_max

    def get_encoder(self, device: str = "cpu") -> Callable[[PPOAgent], torch.Tensor]:
        self.model.eval()
        self.model.to(device)

        def encoder_fn(policy: PPOAgent) -> torch.Tensor:
            with torch.no_grad():
                obs_action, _ = _collect_and_format_trajectory(
                    self.game, policy, player_id=0,
                    num_transitions=self.config.num_transitions,
                    opponent_pool=self.policies,
                    state_min=self.state_min, state_max=self.state_max,
                    normalize=self.config.normalize,
                    num_actions=self.num_actions,
                )
                trajectory = obs_action.unsqueeze(0).to(device)  # (1, seq_len, obs_dim + act_dim)
                embedding = self.model(trajectory)
                return embedding.squeeze(0).cpu()

        return encoder_fn


class GroverModelWrapper(nn.Module):
    """
    Wrapper that accepts the same input format as TrajectoryEncoder
    (batch, seq_len, state_dim + 2) and converts to Grover's expected format.

    This allows embed_trajectory() in agent_identification.py to work unchanged.
    """

    def __init__(self, encoder: GroverEncoder, obs_dim: int, num_actions: int):
        super().__init__()
        self.encoder = encoder
        self.obs_dim = obs_dim
        self.num_actions = num_actions

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (batch, seq_len, obs_dim + 1 + 1) — state, action_scalar, reward
        Returns:
            embedding: (batch, embed_dim)
        """
        states = trajectory[..., :self.obs_dim]
        actions = trajectory[..., self.obs_dim].long()
        # One-hot encode actions
        one_hot = F.one_hot(actions, num_classes=self.num_actions).float()
        # Concatenate obs + one-hot action (drop reward — Grover doesn't use it)
        obs_action = torch.cat([states, one_hot], dim=-1)
        return self.encoder(obs_action)


def load_grover_from_checkpoint(checkpoint_path: str, game: pyspiel.Game, policies=None, device: str = "cpu"):
    """Load a pre-trained Grover encoder from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    obs_dim = checkpoint["obs_dim"]
    num_actions = checkpoint["num_actions"]

    encoder = GroverEncoder(
        obs_dim=obs_dim, act_dim=num_actions,
        hidden_dim=config.hidden_dim, embed_dim=config.embed_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()

    adapter = GroverAdapter(encoder, game, config, policies=policies)
    state_min = checkpoint["state_min"]
    state_max = checkpoint["state_max"]
    if state_min is not None:
        state_min = state_min.cpu()
    if state_max is not None:
        state_max = state_max.cpu()
    adapter.set_normalization_stats(state_min, state_max)

    logger.info(f"Loaded Grover checkpoint from {checkpoint_path}")
    logger.info(f"  obs_dim={obs_dim}, num_actions={num_actions}, embed_dim={config.embed_dim}")

    return encoder, adapter, config