"""Minimal functional autoencoder that matches PPO action probabilities per state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Callable

import torch
from torch import nn
from torch.nn.utils.stateless import functional_call
from torch.utils.data import DataLoader, Dataset

import pyspiel
from open_spiel.python.algorithms import get_all_states

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string
from weight_autoencoder import Autoencoder, ppo_agent_to_vector


def _state_tensor(state: pyspiel.State, player_id: int) -> torch.Tensor:
    return torch.tensor(
        state.information_state_tensor(player_id), dtype=torch.float32
    )


def _legal_mask(state: pyspiel.State, player_id: int, num_actions: int) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.bool)
    mask[state.legal_actions(player_id)] = True
    return mask


def _actor_param_specs(actor: nn.Module) -> list[tuple[str, torch.Size, int]]:
    specs: list[tuple[str, torch.Size, int]] = []
    for name, param in actor.named_parameters():
        specs.append((name, param.shape, param.numel()))
    return specs


def _vector_to_param_dict(
    vector: torch.Tensor, specs: list[tuple[str, torch.Size, int]]
) -> dict[str, torch.Tensor]:
    params: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, numel in specs:
        params[name] = vector[..., offset : offset + numel].view(shape)
        offset += numel
    return params


class PolicyBehaviorDataset(Dataset):
    """All (agent, state) pairs flattened into tensors for supervised learning."""

    def __init__(
        self,
        agents: Iterable[PPOAgent],
        decision_states: list[pyspiel.State],
        num_actions: int,
        device: torch.device | str,
    ):
        weight_vectors: list[torch.Tensor] = []
        state_vectors: list[torch.Tensor] = []
        action_probs: list[torch.Tensor] = []
        legal_masks: list[torch.Tensor] = []

        for agent in agents:
            weight_vec = ppo_agent_to_vector(agent).cpu()
            for state in decision_states:
                player = state.current_player()
                if player < 0:
                    continue
                mask = _legal_mask(state, player, num_actions)
                if not mask.any():
                    continue

                obs = _state_tensor(state, player)
                with torch.no_grad():
                    _, _, _, _, probs = agent.get_action_and_value(
                        obs.unsqueeze(0).to(device),
                        mask.unsqueeze(0).to(device),
                    )
                prob_vec = probs.squeeze(0).cpu()

                weight_vectors.append(weight_vec)
                state_vectors.append(obs)
                action_probs.append(prob_vec)
                legal_masks.append(mask)

        self.weights = torch.stack(weight_vectors)
        self.states = torch.stack(state_vectors)
        self.probs = torch.stack(action_probs)
        self.legal_masks = torch.stack(legal_masks)

    def __len__(self) -> int:
        return self.weights.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "weights": self.weights[idx],
            "states": self.states[idx],
            "probs": self.probs[idx],
            "legal_masks": self.legal_masks[idx],
        }


class FunctionalAutoencoder(Autoencoder):
    """Autoencoder over policy weights whose reconstructions are trained via action KL."""

    def __init__(
        self,
        weight_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        latent_dim: int = 128,
    ):
        super().__init__(weight_dim, hidden_dims, latent_dim)
        self.weight_dim = weight_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.weight_encoder = self.encoder


@dataclass
class TrainingConfig:
    num_agents: int = 8
    ppo_hidden_size: int = 256
    epochs: int = 5
    batch_size: int = 16
    lr: float = 3e-4
    device: str = get_device_string()


def masked_kl_divergence(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    safe_targets = targets * mask.float()
    safe_targets = safe_targets / safe_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_safe_targets = torch.log(safe_targets.clamp_min(1e-8))

    loss = (safe_targets * (log_safe_targets - log_probs)).sum(dim=-1)
    return loss.mean()


def reconstructed_logits(
    recon_weights: torch.Tensor,
    observations: torch.Tensor,
    actor: nn.Module,
    specs: list[tuple[str, torch.Size, int]],
) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    for weight_vec, obs in zip(recon_weights, observations):
        params = _vector_to_param_dict(weight_vec, specs)
        logits.append(functional_call(actor, params, (obs.unsqueeze(0),)))
    return torch.cat(logits, dim=0)


def build_agents(
    num_agents: int,
    num_actions: int,
    info_state_shape: tuple[int, ...],
    hidden_size: int,
) -> list[PPOAgent]:
    layer_init = make_diverse_random_kuhn_poker_layer_init(pyspiel.load_game("kuhn_poker"))
    agents = [
        PPOAgent(num_actions, info_state_shape, "cpu", layer_init, hidden_size)
        for _ in range(num_agents)
    ]
    return agents


def collect_decision_states(game: pyspiel.Game) -> list[pyspiel.State]:
    all_states = get_all_states.get_all_states(game)
    return [
        state
        for state in all_states.values()
        if not state.is_terminal() and not state.is_chance_node()
    ]


def train_functional_autoencoder(
    cfg: TrainingConfig,
    *,
    game: pyspiel.Game | None = None,
    agents: list[PPOAgent] | None = None,
    decision_states: list[pyspiel.State] | None = None,
) -> tuple[FunctionalAutoencoder, list[float]]:
    """Train a functional autoencoder and return the model plus epoch losses."""
    game = game or pyspiel.load_game("kuhn_poker")
    info_state_shape = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    agents = agents or build_agents(cfg.num_agents, num_actions, info_state_shape, cfg.ppo_hidden_size)
    decision_states = decision_states or collect_decision_states(game)
    dataset = PolicyBehaviorDataset(agents, decision_states, num_actions, "cpu")

    weight_dim = dataset.weights.size(1)
    model = FunctionalAutoencoder(weight_dim).to(cfg.device)
    actor_template = PPOAgent(
        num_actions, info_state_shape, cfg.device, hidden_size=cfg.ppo_hidden_size
    ).actor.to(cfg.device)
    param_specs = _actor_param_specs(actor_template)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    epoch_losses: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            weights = batch["weights"].to(cfg.device)
            states = batch["states"].to(cfg.device)
            probs = batch["probs"].to(cfg.device)
            legal_masks = batch["legal_masks"].to(cfg.device)

            optimizer.zero_grad()
            recon = model(weights)
            logits = reconstructed_logits(recon, states, actor_template, param_specs)
            loss = masked_kl_divergence(logits, probs, legal_masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * weights.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}: KL {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

    return model, epoch_losses


def save_functional_autoencoder(
    model: FunctionalAutoencoder,
    path: str | Path,
    training_cfg: TrainingConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist model weights, architectural hyperparameters, and optional metadata."""
    if not isinstance(model, FunctionalAutoencoder):
        raise TypeError("model must be an instance of FunctionalAutoencoder.")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = {
        "weight_dim": model.weight_dim,
        "hidden_dims": model.hidden_dims,
        "latent_dim": model.latent_dim,
    }

    checkpoint: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_config": model_config,
    }

    if training_cfg is not None:
        checkpoint["training_config"] = asdict(training_cfg)
    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, checkpoint_path)


def load_functional_autoencoder(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[FunctionalAutoencoder, TrainingConfig | None, dict[str, Any] | None]:
    """Load a functional autoencoder checkpoint and return model plus optional metadata."""
    map_location = device or "cpu"
    checkpoint = torch.load(Path(path), map_location=map_location)

    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError("Checkpoint missing model_config for FunctionalAutoencoder.")

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint missing state_dict.")

    model = FunctionalAutoencoder(**model_config)
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()

    training_cfg_dict = checkpoint.get("training_config")
    training_cfg = TrainingConfig(**training_cfg_dict) if training_cfg_dict else None

    metadata = checkpoint.get("metadata")

    return model, training_cfg, metadata


class FunctionalEncoderAdapter:
    """Expose the functional autoencoder's weight encoder via a simple API."""

    def __init__(
        self,
        model: FunctionalAutoencoder,
        policy_to_vector: Callable[[PPOAgent], torch.Tensor] = ppo_agent_to_vector,
    ) -> None:
        self.model = model
        self.policy_to_vector = policy_to_vector

    def get_encoder(self, device: str = "cpu") -> Callable[[PPOAgent], torch.Tensor]:
        self.model.eval()
        weight_encoder = self.model.weight_encoder.to(device)

        def encoder_fn(policy: PPOAgent) -> torch.Tensor:
            with torch.no_grad():
                vector = self.policy_to_vector(policy)
                if not isinstance(vector, torch.Tensor):
                    vector = torch.tensor(vector)
                vector = vector.float().to(device)
                if vector.ndim == 1:
                    vector = vector.unsqueeze(0)
                embedding = weight_encoder(vector)
                return embedding.squeeze(0).detach().cpu()

        return encoder_fn


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train functional autoencoder.")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--ppo-hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    return TrainingConfig(
        num_agents=args.num_agents,
        ppo_hidden_size=args.ppo_hidden_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )


if __name__ == "__main__":
    config = parse_args()
    torch.manual_seed(0)
    train_functional_autoencoder(config)
