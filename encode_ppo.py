import argparse
from typing import Callable

import numpy as np
import torch
from torch import nn

import pyspiel

from iig_rl_benchmark.algorithms.ppo import ppo

from weight_autoencoder import (
    AutoencoderConfig,
    VectorDataset,
    encoder_fn,
    save_autoencoder,
    train_autoencoder,
)


Initializer = Callable[[nn.Module, float, float], nn.Module]


def make_diverse_random_initializer(num_actions: int) -> Initializer:
    """Create the layer init function used for diverse PPO agents."""

    def diverse_random_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
        if not isinstance(layer, nn.Linear):
            raise TypeError("Expected linear layer during initialization.")
        torch.nn.init.orthogonal_(layer.weight, 2.2)
        if layer.out_features == num_actions:
            torch.nn.init.uniform_(layer.bias, -1, 1)
        else:
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    return diverse_random_init


def build_random_ppo_agent(
    game: pyspiel.Game,
    device: str,
    init_fn: Initializer,
) -> ppo.PPOAgent:
    """Instantiate a randomly initialized PPOAgent for the given game."""
    num_actions = game.num_distinct_actions()
    observation_shape = game.information_state_tensor_shape()
    return ppo.PPOAgent(num_actions, observation_shape, device, init_fn)


def encode_random_agents(
    num_agents: int = 100,
    seed: int | None = None,
    device: str = "cpu",
    game_name: str = "kuhn_poker",
) -> torch.Tensor:
    """Create `num_agents` random PPO agents and encode their actor weights."""
    if num_agents <= 0:
        raise ValueError("num_agents must be positive.")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    game = pyspiel.load_game(game_name)
    num_actions = game.num_distinct_actions()
    init_fn = make_diverse_random_initializer(num_actions)

    encoded_vectors: list[torch.Tensor] = []
    for _ in range(num_agents):
        agent = build_random_ppo_agent(game, device, init_fn)
        encoded_vectors.append(encoder_fn(agent))

    vectors = torch.stack(list(encoded_vectors))
    return vectors


def _parse_args() -> argparse.Namespace:
    def hidden_dims_arg(value: str) -> tuple[int, ...]:
        dims = [int(v.strip()) for v in value.split(",") if v.strip()]
        if not dims:
            raise argparse.ArgumentTypeError("hidden dims must not be empty")
        return tuple(dims)

    parser = argparse.ArgumentParser(description="Encode random PPO agents for weight autoencoding.")
    parser.add_argument("--num-agents", type=int, default=100, help="Number of random PPO agents to encode.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for agent instantiation.")
    parser.add_argument("--game", type=str, default="kuhn_poker", help="OpenSpiel game name.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the stacked weight vectors as a .pt file.",
    )
    parser.add_argument(
        "--autoencoder-output",
        type=str,
        default=None,
        help="Optional checkpoint path for the trained autoencoder.",
    )
    parser.add_argument("--hidden-dims", type=hidden_dims_arg, default=(64, 64), help="Comma separated hidden dims.")
    parser.add_argument("--bottleneck-dim", type=int, default=64, help="Autoencoder latent dimension size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for the autoencoder.")
    parser.add_argument("--batch-size", type=int, default=16, help="Autoencoder training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for autoencoder training.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split proportion.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    vectors = encode_random_agents(
        num_agents=args.num_agents,
        seed=args.seed,
        device=args.device,
        game_name=args.game,
    )
    if args.output is not None:
        torch.save(vectors, args.output)
        print(f"Saved PPO weight vectors to {args.output} with shape {tuple(vectors.shape)}")
    else:
        print(f"Generated PPO weight vectors with shape {tuple(vectors.shape)}")

    dataset = VectorDataset(vectors)
    cfg = AutoencoderConfig(
        input_dim=vectors.shape[1],
        hidden_dims=args.hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed if args.seed is not None else 0,
        device=args.device,
    )
    model, history = train_autoencoder(dataset, cfg)
    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    print(f"Trained autoencoder. Final train loss: {final_train:.6f}, val loss: {final_val:.6f}")

    if args.autoencoder_output is not None:
        save_autoencoder(model, args.autoencoder_output, cfg)
        print(f"Saved autoencoder checkpoint to {args.autoencoder_output}")


if __name__ == "__main__":
    main()
