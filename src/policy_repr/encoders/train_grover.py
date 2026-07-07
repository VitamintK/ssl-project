"""
Train a Grover encoder and save the checkpoint.

Usage:
    python -m policy_repr.encoders.train_grover --game kuhn_poker --agent-pool agent_pools/kuhn_poker_seed42_n500.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import pyspiel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from policy_repr.utils import make_diverse_random_kuhn_poker_layer_init, get_device_string
from policy_repr.downstream.heads import set_seed
from policy_repr.datasets.generate_agents import load_agents
from policy_repr.encoders.grover import GroverConfig, GroverTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Grover encoder")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--agent-pool", type=str, default=None,
                        help="Path to saved agent pool (uses pretrain agents)")
    parser.add_argument("--num-agents", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = args.device or get_device_string()
    game = pyspiel.load_game(args.game)

    print(f"Game: {args.game}")
    print(f"Device: {device}")

    # Load or create agents
    if args.agent_pool:
        print(f"Loading pretrain agents from {args.agent_pool}...")
        ppo_agents, _ = load_agents(args.agent_pool, game)
    else:
        info_state_size = game.information_state_tensor_shape()
        num_actions = game.num_distinct_actions()
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)
        print(f"Creating {args.num_agents} random agents...")
        ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, 256)
            for _ in range(args.num_agents)
        ]

    print(f"Training on {len(ppo_agents)} agents")

    config = GroverConfig(
        hidden_dim=100, embed_dim=128, policy_hidden_dim=64,
        lambda_weight=0.1,
        num_transitions=200, batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, weight_decay=1e-5, val_split=0.2, lr_scheduler="cosine",
        normalize=True, trajectories_per_policy=50,
        device=device, seed=args.seed,
    )

    trainer = GroverTrainer(config, game, ppo_agents)
    encoder, history = trainer.train()

    # Save checkpoint
    obs_dim = game.information_state_tensor_shape()[0]
    num_actions = game.num_distinct_actions()

    output_path = args.output
    if output_path is None:
        safe_name = args.game.replace("(", "_").replace(")", "").replace(",", "_").replace("=", "")
        output_path = f"checkpoints/grover_{safe_name}.pt"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": encoder.state_dict(),
        "config": config,
        "obs_dim": obs_dim,
        "num_actions": num_actions,
        "state_min": trainer.train_dataset.state_min,
        "state_max": trainer.train_dataset.state_max,
    }, output_path)
    print(f"Saved checkpoint to {output_path}")
