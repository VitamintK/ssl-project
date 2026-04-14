"""
Grover et al. (2018)-style VAE Trajectory Encoder baseline.

Reimplements the core idea from "Learning Policy Representations in Multiagent
Systems": encode agent trajectories with a GRU, learn a latent embedding via
a variational autoencoder (VAE), and decode to reconstruct the trajectory.

The key difference from our contrastive trajectory encoder is the training
objective: VAE reconstruction + KL divergence vs. NT-Xent contrastive loss.

This implementation reuses the same trajectory collection and normalization
utilities from trajectory_encoder.py for a fair comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pyspiel
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent


@dataclass
class GroverVAEConfig:
    """Configuration for the Grover-style VAE encoder."""
    hidden_dim: int = 128
    latent_dim: int = 128
    num_gru_layers: int = 2
    dropout: float = 0.1

    # Training
    num_transitions: int = 200
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 1e-5
    val_split: float = 0.2
    lr_scheduler: str = "cosine"
    kl_weight: float = 0.1  # Beta for KL term in ELBO

    # Data
    normalize: bool = True
    trajectories_per_policy: int = 50

    # System
    device: str = "cpu"
    seed: int = 0


class GroverTrajectoryVAE(nn.Module):
    """
    VAE-based trajectory encoder following Grover et al. (2018).

    Architecture:
        Encoder: transition projection -> GRU -> mu/logvar heads -> z
        Decoder: z -> GRU -> transition reconstruction
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Input dimension: state + action_embedding + reward
        self.action_embedding = nn.Embedding(action_dim, hidden_dim // 4)
        transition_dim = state_dim + (hidden_dim // 4) + 1
        self.input_projection = nn.Linear(transition_dim, hidden_dim)

        # Encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # VAE heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim * num_gru_layers)
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )
        # Reconstruct continuous state + reward
        self.output_state = nn.Linear(hidden_dim, state_dim)
        self.output_reward = nn.Linear(hidden_dim, 1)
        # Reconstruct discrete action
        self.output_action = nn.Linear(hidden_dim, action_dim)

        self.num_gru_layers = num_gru_layers

    def encode(self, transitions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a trajectory into latent space.

        Args:
            transitions: (batch, seq_len, state_dim + 2) — raw transitions

        Returns:
            z, mu, logvar
        """
        batch_size, seq_len, _ = transitions.shape

        # Split and project
        states = transitions[:, :, :self.state_dim]
        actions = transitions[:, :, self.state_dim].long()
        rewards = transitions[:, :, self.state_dim + 1:self.state_dim + 2]

        action_emb = self.action_embedding(actions)
        combined = torch.cat([states, action_emb, rewards], dim=-1)
        x = self.input_projection(combined)

        # GRU encoding — use final hidden state
        _, h_n = self.encoder_gru(x)  # h_n: (num_layers, batch, hidden)
        # Use last layer's hidden state
        h_final = h_n[-1]  # (batch, hidden)

        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)

        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def decode(self, z: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode latent vector into trajectory reconstruction.

        Returns:
            recon_states: (batch, seq_len, state_dim)
            recon_actions: (batch, seq_len, action_dim) — logits
            recon_rewards: (batch, seq_len, 1)
        """
        batch_size = z.shape[0]

        # Initialize decoder hidden state from z
        h_0 = self.latent_to_hidden(z)  # (batch, hidden * num_layers)
        h_0 = h_0.view(batch_size, self.num_gru_layers, self.hidden_dim)
        h_0 = h_0.permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden)

        # Create decoder input (learned start token, repeated)
        decoder_input = torch.zeros(batch_size, seq_len, self.hidden_dim, device=z.device)

        # Decode
        output, _ = self.decoder_gru(decoder_input, h_0)

        recon_states = self.output_state(output)
        recon_actions = self.output_action(output)
        recon_rewards = self.output_reward(output)

        return recon_states, recon_actions, recon_rewards

    def forward(self, transitions: torch.Tensor):
        """Full forward pass: encode -> sample -> decode."""
        seq_len = transitions.shape[1]
        z, mu, logvar = self.encode(transitions)
        recon_states, recon_actions, recon_rewards = self.decode(z, seq_len)
        return recon_states, recon_actions, recon_rewards, mu, logvar

    def get_embedding(self, transitions: torch.Tensor) -> torch.Tensor:
        """Get the latent embedding (mu) for downstream use."""
        _, mu, _ = self.encode(transitions)
        return mu


def vae_loss(
    recon_states, recon_actions, recon_rewards,
    target_transitions, mu, logvar,
    state_dim: int, kl_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute VAE ELBO loss.

    Reconstruction: MSE for states/rewards, cross-entropy for actions.
    Regularization: KL divergence from standard normal.
    """
    target_states = target_transitions[:, :, :state_dim]
    target_actions = target_transitions[:, :, state_dim].long()
    target_rewards = target_transitions[:, :, state_dim + 1:state_dim + 2]

    # Reconstruction losses
    state_loss = F.mse_loss(recon_states, target_states)
    action_loss = F.cross_entropy(
        recon_actions.reshape(-1, recon_actions.shape[-1]),
        target_actions.reshape(-1),
    )
    reward_loss = F.mse_loss(recon_rewards, target_rewards)
    recon_loss = state_loss + action_loss + reward_loss

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_weight * kl_loss

    metrics = {
        "total": total_loss.item(),
        "recon": recon_loss.item(),
        "kl": kl_loss.item(),
        "state": state_loss.item(),
        "action": action_loss.item(),
        "reward": reward_loss.item(),
    }
    return total_loss, metrics


class TrajectoryReconstructionDataset(Dataset):
    """
    Dataset for VAE training. Returns single trajectories (not pairs).
    Reuses trajectory collection from trajectory_encoder.py.
    """

    def __init__(
        self,
        game: pyspiel.Game,
        policies: List[PPOAgent],
        num_transitions: int = 200,
        normalize: bool = True,
        trajectories_per_policy: int = 50,
    ):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from trajectory_encoder import collect_trajectory_transitions, normalize_transitions

        self.game = game
        self.state_dim = game.information_state_tensor_shape()[0]

        # Compute normalization stats
        self.state_min = None
        self.state_max = None
        if normalize:
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

        # Pre-collect all trajectories
        print(f"Pre-collecting {trajectories_per_policy} trajectories for {len(policies)} policies...")
        self.trajectories = []
        for policy in tqdm(policies, desc="Collecting trajectories"):
            for _ in range(trajectories_per_policy):
                trans = collect_trajectory_transitions(
                    game, policy, player_id=0, num_transitions=num_transitions,
                    opponent_pool=policies,
                )
                if normalize and self.state_min is not None:
                    trans, _, _ = normalize_transitions(trans, self.state_min, self.state_max)

                tensors = []
                for state, action, reward in trans:
                    t = torch.cat([
                        state,
                        torch.tensor([action], dtype=torch.float32),
                        torch.tensor([reward], dtype=torch.float32),
                    ])
                    tensors.append(t)
                self.trajectories.append(torch.stack(tensors))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


class GroverVAETrainer:
    """Trainer for the Grover-style VAE trajectory encoder."""

    def __init__(self, config: GroverVAEConfig, game: pyspiel.Game, policies: List[PPOAgent]):
        self.config = config
        self.game = game

        state_dim = game.information_state_tensor_shape()[0]
        action_dim = game.num_distinct_actions()

        # Train/val split
        split_idx = int(len(policies) * (1 - config.val_split))
        train_policies = policies[:split_idx]
        val_policies = policies[split_idx:]
        print(f"Train/val split: {len(train_policies)} / {len(val_policies)} policies")

        # Model
        self.model = GroverTrajectoryVAE(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_gru_layers=config.num_gru_layers,
            dropout=config.dropout,
        ).to(config.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

        if config.lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=1e-6)
        else:
            self.scheduler = None

        # Datasets
        self.train_dataset = TrajectoryReconstructionDataset(
            game, train_policies,
            num_transitions=config.num_transitions,
            normalize=config.normalize,
            trajectories_per_policy=config.trajectories_per_policy,
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True,
        )

        if val_policies:
            self.val_dataset = TrajectoryReconstructionDataset(
                game, val_policies,
                num_transitions=config.num_transitions,
                normalize=config.normalize,
                trajectories_per_policy=config.trajectories_per_policy,
            )
            # Copy normalization stats from train
            self.val_dataset.state_min = self.train_dataset.state_min
            self.val_dataset.state_max = self.train_dataset.state_max
            self.val_loader = DataLoader(
                self.val_dataset, batch_size=config.batch_size, shuffle=False,
            )
        else:
            self.val_dataset = None
            self.val_loader = None

        self.state_dim = state_dim

    def train(self) -> Tuple[GroverTrajectoryVAE, dict]:
        torch.manual_seed(self.config.seed)

        best_val_loss = float("inf")
        best_model_state = None
        best_epoch = 0
        history = {"train": [], "val": []}

        print(f"\nTraining Grover VAE for {self.config.epochs} epochs...")
        progress = tqdm(range(1, self.config.epochs + 1), desc="Training")

        for epoch in progress:
            # Train
            self.model.train()
            train_loss = 0.0
            n = 0
            for batch in self.train_loader:
                batch = batch.to(self.config.device)
                self.optimizer.zero_grad()
                rs, ra, rr, mu, logvar = self.model(batch)
                loss, _ = vae_loss(rs, ra, rr, batch, mu, logvar,
                                   self.state_dim, self.config.kl_weight)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                n += 1
            train_loss /= max(n, 1)
            history["train"].append(train_loss)

            # Validate
            if self.val_loader:
                self.model.eval()
                val_loss = 0.0
                n = 0
                with torch.no_grad():
                    for batch in self.val_loader:
                        batch = batch.to(self.config.device)
                        rs, ra, rr, mu, logvar = self.model(batch)
                        loss, _ = vae_loss(rs, ra, rr, batch, mu, logvar,
                                           self.state_dim, self.config.kl_weight)
                        val_loss += loss.item()
                        n += 1
                val_loss /= max(n, 1)
                history["val"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    best_epoch = epoch

                progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")
            else:
                progress.set_postfix(train=f"{train_loss:.4f}")

            if self.scheduler:
                self.scheduler.step()

        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

        history["best_epoch"] = best_epoch
        history["best_val_loss"] = best_val_loss
        return self.model, history

    def save_checkpoint(self, path: str):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "state_min": self.train_dataset.state_min,
            "state_max": self.train_dataset.state_max,
        }, path)
        print(f"Saved checkpoint to {path}")


class GroverVAEAdapter:
    """Adapter matching the TrajectoryEncoderAdapter interface."""

    def __init__(self, model: GroverTrajectoryVAE, game: pyspiel.Game,
                 config: GroverVAEConfig, policies: Optional[List[PPOAgent]] = None):
        self.model = model
        self.game = game
        self.config = config
        self.policies = policies
        self.state_min = None
        self.state_max = None

    def set_normalization_stats(self, state_min, state_max):
        self.state_min = state_min
        self.state_max = state_max

    def get_encoder(self, device: str = "cpu") -> Callable[[PPOAgent], torch.Tensor]:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from trajectory_encoder import collect_trajectory_transitions, normalize_transitions

        self.model.eval()
        self.model.to(device)

        def encoder_fn(policy: PPOAgent) -> torch.Tensor:
            with torch.no_grad():
                transitions = collect_trajectory_transitions(
                    self.game, policy, player_id=0,
                    num_transitions=self.config.num_transitions,
                    opponent_pool=self.policies,
                )
                if self.config.normalize and self.state_min is not None:
                    transitions, _, _ = normalize_transitions(
                        transitions, self.state_min, self.state_max,
                    )
                tensors = []
                for state, action, reward in transitions:
                    t = torch.cat([
                        state,
                        torch.tensor([action], dtype=torch.float32),
                        torch.tensor([reward], dtype=torch.float32),
                    ])
                    tensors.append(t)
                trajectory = torch.stack(tensors).unsqueeze(0).to(device)
                embedding = self.model.get_embedding(trajectory)
                return embedding.squeeze(0).cpu()

        return encoder_fn
