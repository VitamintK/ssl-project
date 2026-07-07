"""
Pre-generate and save agent populations for reproducible evaluation.

Creates pretrain + eval agent populations for a given game and seed,
saving their state dicts so any evaluation script can load identical agents.

Usage:
    python generate_agents.py --game kuhn_poker --seed 42 --num-agents 500
    python generate_agents.py --game leduc_poker --seed 43 --num-agents 500
"""

import argparse
from pathlib import Path

import torch
import pyspiel

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from policy_repr.utils import make_diverse_random_kuhn_poker_layer_init
from policy_repr.downstream.heads import set_seed


def generate_and_save(game_name: str, seed: int, num_agents: int, output_dir: str = "agent_pools"):
    set_seed(seed)

    game = pyspiel.load_game(game_name)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    hidden_size = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    print(f"Creating {num_agents} pretrain agents...")
    pretrain_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, hidden_size)
        for _ in range(num_agents)
    ]

    print(f"Creating {num_agents} eval agents...")
    eval_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, hidden_size)
        for _ in range(num_agents)
    ]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    safe_name = game_name.replace("(", "_").replace(")", "").replace(",", "_").replace("=", "")
    save_path = out_path / f"{safe_name}_seed{seed}_n{num_agents}.pt"
    torch.save({
        "game": game_name,
        "seed": seed,
        "num_agents": num_agents,
        "hidden_size": hidden_size,
        "info_state_size": info_state_size,
        "num_actions": num_actions,
        "pretrain_state_dicts": [a.state_dict() for a in pretrain_agents],
        "eval_state_dicts": [a.state_dict() for a in eval_agents],
    }, save_path)
    print(f"Saved to {save_path}")


def load_agents(path: str, game=None):
    """Load pretrain and eval agents from a saved pool file."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    if game is None:
        game = pyspiel.load_game(checkpoint["game"])

    info_state_size = checkpoint["info_state_size"]
    num_actions = checkpoint["num_actions"]
    hidden_size = checkpoint["hidden_size"]
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    def _make_agents(state_dicts):
        agents = []
        for sd in state_dicts:
            agent = PPOAgent(num_actions, info_state_size, 'cpu', layer_init, hidden_size)
            agent.load_state_dict(sd)
            agents.append(agent)
        return agents

    pretrain = _make_agents(checkpoint["pretrain_state_dicts"])
    eval_agents = _make_agents(checkpoint["eval_state_dicts"])

    print(f"Loaded {len(pretrain)} pretrain + {len(eval_agents)} eval agents from {path}")
    return pretrain, eval_agents


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-agents", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="agent_pools")
    args = parser.parse_args()

    generate_and_save(args.game, args.seed, args.num_agents, args.output_dir)
