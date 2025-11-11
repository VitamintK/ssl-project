"""Prototype utilities for building contrastive behaviour datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

import pyspiel

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string


def _state_tensor(state: pyspiel.State, player_id: int) -> torch.Tensor:
    return torch.tensor(state.information_state_tensor(player_id), dtype=torch.float32)


def _legal_mask(state: pyspiel.State, player_id: int, num_actions: int) -> torch.Tensor:
    mask = torch.zeros(num_actions, dtype=torch.bool)
    mask[state.legal_actions(player_id)] = True
    return mask


def _pad_steps(
    steps: list[torch.Tensor],
    max_len: int,
    feature_dim: int,
) -> torch.Tensor:
    traj = torch.zeros((max_len, feature_dim), dtype=torch.float32)
    for idx, step in enumerate(steps[:max_len]):
        traj[idx] = step
    return traj


def _sample_chance_action(state: pyspiel.State, rng: random.Random) -> None:
    """Advance chance nodes by sampling from their outcome distribution."""
    outcomes = state.chance_outcomes()
    threshold = rng.random()
    cumulative = 0.0
    for action, prob in outcomes:
        cumulative += prob
        if threshold <= cumulative:
            state.apply_action(action)
            return
    state.apply_action(outcomes[-1][0])


def _trajectory_feature_dim(info_state_shape: Sequence[int], num_actions: int) -> int:
    info_dim = int(np.prod(info_state_shape, dtype=int))
    return info_dim + num_actions + 1


def _rollout_trajectory(
    game: pyspiel.Game,
    policy_a: PPOAgent,
    policy_b: PPOAgent,
    *,
    trajectory_length: int,
    seed: int,
    device: str,
    feature_dim: int,
    info_state_dim: int,
    num_actions: int,
) -> torch.Tensor:
    """Run a trajectory between two policies and return padded step features."""
    state = game.new_initial_state()
    policy_a.eval()
    policy_b.eval()
    steps: list[torch.Tensor] = []
    rng = random.Random(seed)
    policies = {0: policy_a, 1: policy_b}
    with torch.no_grad():
        while len(steps) < trajectory_length and not state.is_terminal():
            if state.is_chance_node():
                _sample_chance_action(state, rng)
                continue

            player_id = state.current_player()
            policy = policies.get(player_id)
            if policy is None:
                raise ValueError(
                    "Only two-player games are supported for behaviour rollouts."
                )

            obs = _state_tensor(state, player_id)
            mask = _legal_mask(state, player_id, num_actions)
            action, _, _, _, probs = policy.get_action_and_value(
                obs.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
            )
            action_id = int(action.squeeze(0).item())
            state.apply_action(action_id)

            step_vec = torch.zeros(feature_dim, dtype=torch.float32)
            step_vec[:info_state_dim] = obs
            step_vec[info_state_dim : info_state_dim + num_actions] = probs.squeeze(
                0
            ).cpu()
            step_vec[-1] = float(player_id)
            steps.append(step_vec)

    return _pad_steps(steps, trajectory_length, feature_dim)


def build_agents(
    num_agents: int,
    num_actions: int,
    info_state_shape: Sequence[int],
    hidden_size: int,
    device: str,
) -> list[PPOAgent]:
    layer_init = make_diverse_random_kuhn_poker_layer_init(pyspiel.load_game("kuhn_poker"))
    agents: list[PPOAgent] = []
    for _ in range(num_agents):
        agent = PPOAgent(num_actions, info_state_shape, device, layer_init, hidden_size)
        agent.to(device)
        agents.append(agent)
    return agents


@dataclass
class BehaviourDatasetConfig:
    """Configuration for constructing contrastive behaviour datasets."""

    num_policies: int = 100
    partner_triplets_per_policy: int = 1
    seeds_per_triplet: int = 4
    trajectory_length: int = 16
    ppo_hidden_size: int = 256
    device: str = get_device_string()
    game_name: str = "kuhn_poker"
    seed: int = 0


class ContrastiveTrajectoryDataset(Dataset):
    """Dataset of (A,B), (A,C) positives plus (B,C) negative per anchor policy."""

    def __init__(
        self,
        cfg: BehaviourDatasetConfig,
        *,
        game: pyspiel.Game | None = None,
        agents: list[PPOAgent] | None = None,
    ):
        self.cfg = cfg
        self.game = game or pyspiel.load_game(cfg.game_name)
        self.info_state_shape = self.game.information_state_tensor_shape()
        self.num_actions = self.game.num_distinct_actions()
        self.info_state_dim = int(np.prod(self.info_state_shape, dtype=int))
        self.feature_dim = _trajectory_feature_dim(
            self.info_state_shape, self.num_actions
        )

        total_required = cfg.num_policies
        if agents is None:
            agents = build_agents(
                total_required,
                self.num_actions,
                self.info_state_shape,
                cfg.ppo_hidden_size,
                cfg.device,
            )
        if len(agents) < total_required:
            raise ValueError("Not enough agents provided to match num_policies.")

        self.agents = agents[:total_required]
        self._build_triplets()

    def _build_triplets(self) -> None:
        cfg = self.cfg
        rng = random.Random(cfg.seed)
        traj_ab: list[torch.Tensor] = []
        traj_ac: list[torch.Tensor] = []
        traj_bc: list[torch.Tensor] = []
        anchor_ids: list[int] = []
        partner_ids: list[int] = []
        aux_partner_ids: list[int] = []
        seeds: list[int] = []

        for anchor_id, anchor in enumerate(self.agents):
            partners = [idx for idx in range(len(self.agents)) if idx != anchor_id]
            for _ in range(cfg.partner_triplets_per_policy):
                if len(partners) < 2:
                    raise ValueError("Need at least 3 policies to form triplets.")
                partner_id = rng.choice(partners)
                other_candidates = [idx for idx in partners if idx != partner_id]
                aux_id = rng.choice(other_candidates)

                partner = self.agents[partner_id]
                aux_partner = self.agents[aux_id]

                for seed_offset in range(cfg.seeds_per_triplet):
                    seed = rng.randrange(0, 2**31) + seed_offset
                    traj_ab.append(
                        _rollout_trajectory(
                            self.game,
                            anchor,
                            partner,
                            trajectory_length=cfg.trajectory_length,
                            seed=seed,
                            device=cfg.device,
                            feature_dim=self.feature_dim,
                            info_state_dim=self.info_state_dim,
                            num_actions=self.num_actions,
                        )
                    )
                    traj_ac.append(
                        _rollout_trajectory(
                            self.game,
                            anchor,
                            aux_partner,
                            trajectory_length=cfg.trajectory_length,
                            seed=seed + 1,
                            device=cfg.device,
                            feature_dim=self.feature_dim,
                            info_state_dim=self.info_state_dim,
                            num_actions=self.num_actions,
                        )
                    )
                    traj_bc.append(
                        _rollout_trajectory(
                            self.game,
                            partner,
                            aux_partner,
                            trajectory_length=cfg.trajectory_length,
                            seed=seed + 2,
                            device=cfg.device,
                            feature_dim=self.feature_dim,
                            info_state_dim=self.info_state_dim,
                            num_actions=self.num_actions,
                        )
                    )
                    anchor_ids.append(anchor_id)
                    partner_ids.append(partner_id)
                    aux_partner_ids.append(aux_id)
                    seeds.append(seed)

        self.traj_ab = torch.stack(traj_ab)
        self.traj_ac = torch.stack(traj_ac)
        self.traj_bc = torch.stack(traj_bc)
        self.anchor_ids = torch.tensor(anchor_ids, dtype=torch.long)
        self.partner_ids = torch.tensor(partner_ids, dtype=torch.long)
        self.aux_partner_ids = torch.tensor(aux_partner_ids, dtype=torch.long)
        self.seeds = torch.tensor(seeds, dtype=torch.long)

    def __len__(self) -> int:
        return self.traj_ab.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "traj_ab": self.traj_ab[idx],
            "traj_ac": self.traj_ac[idx],
            "traj_bc": self.traj_bc[idx],
            "anchor_id": self.anchor_ids[idx],
            "partner_id": self.partner_ids[idx],
            "aux_partner_id": self.aux_partner_ids[idx],
            "seed": self.seeds[idx],
        }


def build_behaviour_dataset(
    cfg: BehaviourDatasetConfig,
    *,
    game: pyspiel.Game | None = None,
    agents: list[PPOAgent] | None = None,
) -> ContrastiveTrajectoryDataset:
    """Convenience wrapper that instantiates a dataset from config."""
    return ContrastiveTrajectoryDataset(cfg, game=game, agents=agents)


if __name__ == "__main__":
    cfg = BehaviourDatasetConfig()
    dataset = build_behaviour_dataset(cfg)
    print(f"Built dataset with {len(dataset)} samples.")
    sample = dataset[0]
