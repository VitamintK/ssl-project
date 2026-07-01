"""Minimal functional autoencoder that matches PPO action probabilities per state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Callable

import torch
from torch import nn
from torch.nn.utils.stateless import functional_call
from torch.utils.data import DataLoader, Dataset, Subset

import pyspiel
from open_spiel.python.algorithms import get_all_states

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string
from weight_autoencoder import (
    Autoencoder,
    AutoencoderConfig,
    ppo_agent_to_vector,
    save_autoencoder,
    load_autoencoder,
)


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
        self.device = torch.device(device)
        self.agents = list(agents)
        if not self.agents:
            raise ValueError("PolicyBehaviorDataset requires at least one agent.")

        self.agent_weights = torch.stack(
            [ppo_agent_to_vector(agent).cpu() for agent in self.agents]
        )

        state_vectors: list[torch.Tensor] = []
        legal_masks: list[torch.Tensor] = []
        for state in decision_states:
            player = state.current_player()
            if player < 0:
                continue
            mask = _legal_mask(state, player, num_actions)
            if not mask.any():
                continue

            obs = _state_tensor(state, player)
            state_vectors.append(obs)
            legal_masks.append(mask)

        if not state_vectors:
            raise ValueError("PolicyBehaviorDataset requires at least one decision state.")

        self.states = torch.stack(state_vectors)
        self.legal_masks = torch.stack(legal_masks)
        self.num_states = self.states.size(0)

    def __len__(self) -> int:
        return len(self.agents) * self.num_states

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        agent_idx = idx // self.num_states
        state_idx = idx % self.num_states

        agent = self.agents[agent_idx]
        obs = self.states[state_idx]
        mask = self.legal_masks[state_idx]
        with torch.no_grad():
            _, _, _, _, probs = agent.get_action_and_value(
                obs.unsqueeze(0).to(self.device),
                mask.unsqueeze(0).to(self.device),
            )

        return {
            "weights": self.agent_weights[agent_idx],
            "states": obs,
            "probs": probs.squeeze(0).cpu(),
            "legal_masks": mask,
        }


def _fractional_subset(
    dataset: Dataset,
    fraction: float | None,
    seed: int | None = None,
) -> Dataset:
    """Return a randomly sampled subset of the dataset controlled by `fraction`."""
    if fraction is None or fraction >= 1.0:
        return dataset
    if not (0 < fraction <= 1):
        raise ValueError("dataset_fraction must be in the interval (0, 1].")

    subset_size = max(1, int(len(dataset) * fraction))
    if subset_size == len(dataset):
        return dataset

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


@dataclass
class TrainingConfig:
    num_agents: int = 8
    ppo_hidden_size: int = 256
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    @property
    def device(self) -> str:
        return self.autoencoder.device


def masked_kl_divergence(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    safe_targets = targets * mask.float()
    safe_targets = safe_targets / safe_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    log_safe_targets = torch.log(safe_targets.clamp_min(1e-8))
    log_probs = log_probs.masked_fill(~mask, 0.0)

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
    checkpoint_path: str | Path | None = None,
) -> tuple[Autoencoder, list[float]]:
    """Train a functional autoencoder and return the model plus epoch losses."""
    game = game or pyspiel.load_game("kuhn_poker")
    info_state_shape = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    agents = agents or build_agents(cfg.num_agents, num_actions, info_state_shape, cfg.ppo_hidden_size)
    decision_states = decision_states or collect_decision_states(game)
    print(f"Collected {len(decision_states)} decision states from the game.")
    dataset = PolicyBehaviorDataset(agents, decision_states, num_actions, "cpu")
    print(f"Constructed dataset with {len(dataset)} (agent, state) pairs.")
    weight_dim = dataset.agent_weights.size(1)
    ae_cfg = cfg.autoencoder
    model = Autoencoder(weight_dim, ae_cfg.hidden_dims, ae_cfg.bottleneck_dim).to(cfg.device)
    print("makde model")

    actor_template = PPOAgent(
        num_actions, info_state_shape, cfg.device, hidden_size=cfg.ppo_hidden_size
    ).actor.to(cfg.device)
    print
    param_specs = _actor_param_specs(actor_template)

    print("making data loader...")
    dataset_fraction = getattr(ae_cfg, "dataset_fraction", 1.0)
    train_dataset = _fractional_subset(dataset, dataset_fraction, getattr(ae_cfg, "seed", None))
    if len(train_dataset) != len(dataset):
        frac = len(train_dataset) / len(dataset)
        print(f"Subsampled training dataset to {len(train_dataset)} samples ({frac:.1%} of full dataset).")
    loader = DataLoader(train_dataset, batch_size=ae_cfg.batch_size, shuffle=True)
    print("starting training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay)
    epoch_losses: list[float] = []

    for epoch in range(1, ae_cfg.epochs + 1):
        total_loss = 0.0
        for batch_idx, batch in enumerate(loader):
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

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch}: KL {avg_loss:.4f}")
        epoch_losses.append(avg_loss)

    if checkpoint_path is not None:
        save_autoencoder(model, ae_cfg, checkpoint_path)
        print(f"Saved functional autoencoder checkpoint to {checkpoint_path}")

    return model, epoch_losses



class FunctionalEncoderAdapter:
    """Expose the autoencoder's weight encoder via a simple API."""

    def __init__(
        self,
        model: Autoencoder,
        policy_to_vector: Callable[[PPOAgent], torch.Tensor] = ppo_agent_to_vector,
    ) -> None:
        self.model = model
        self.policy_to_vector = policy_to_vector

    def get_encoder(self, device: str = "cpu") -> Callable[[PPOAgent], torch.Tensor]:
        self.model.eval()
        weight_encoder = self.model.encoder.to(device)

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


def _hidden_dims_arg(value: str) -> tuple[int, ...]:
    dims = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims must not be empty")
    return tuple(dims)

def parse_args() -> tuple[TrainingConfig, str]:
    parser = argparse.ArgumentParser(description="Train functional autoencoder.")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--ppo-hidden-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=_hidden_dims_arg, default=(512, 256))
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--dataset-fraction",
        type=float,
        default=1.0,
        help="Fraction of (agent, state) pairs to use for training (0 < f <= 1).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/functional_autoencoder_kuhn_poker.pt",
        help="Where to save the trained autoencoder checkpoint.",
    )
    args = parser.parse_args()

    device = args.device or get_device_string()
    print("Using device:", device)

    auto_cfg = AutoencoderConfig(
        hidden_dims=args.hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        dataset_fraction=args.dataset_fraction,
    )

    return TrainingConfig(
        num_agents=args.num_agents,
        ppo_hidden_size=args.ppo_hidden_size,
        autoencoder=auto_cfg,
    ), args.checkpoint_path


if __name__ == "__main__":
    config, checkpoint_path = parse_args()
    torch.manual_seed(0)
    train_functional_autoencoder(config, checkpoint_path=checkpoint_path)
