import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union


import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


import argparse
from typing import Callable


import pyspiel

from iig_rl_benchmark.algorithms.ppo import ppo


import utils
Initializer = Callable[[nn.Module, float, float], nn.Module]

class VectorDataset(Dataset):
    """Dataset wrapper for weight vectors derived from PPO agents or raw tensors."""

    def __init__(
        self,
        data: Union[torch.Tensor, Sequence[Any]],
        transform: Callable[[Any], torch.Tensor] | None = None,
    ):
        if isinstance(data, torch.Tensor):
            vectors = data
        else:
            if not isinstance(data, Sequence):
                raise TypeError("data must be a tensor or a sequence of PPO agents.")
            if len(data) == 0:
                raise ValueError("data sequence must not be empty.")
            transform = transform or encoder_fn
            vectors = torch.stack([transform(agent) for agent in data])

        if vectors.ndim != 2:
            raise ValueError("vectors must be (num_samples, dim)")
        self.vectors = vectors.float()

    def __len__(self) -> int:
        return self.vectors.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]


def encoder_fn(ppo_agent: Any) -> torch.Tensor:
    """Flatten all actor parameters from a PPO agent for autoencoder input."""
    actor = getattr(ppo_agent, "actor", None)
    if actor is None:
        raise AttributeError("ppo_agent must expose an 'actor' attribute.")
    if not isinstance(actor, nn.Module):
        raise TypeError("ppo_agent.actor must be an nn.Module.")

    params = [param.detach() for param in actor.parameters()]
    if not params:
        raise ValueError("ppo_agent.actor has no parameters to extract.")

    # Use Torch helper to produce a single contiguous 1D tensor of all weights.
    return nn.utils.parameters_to_vector(params).detach()


class WeightAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], bottleneck_dim: int):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        encoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            encoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        encoder_layers.append(nn.Linear(dims[-1], bottleneck_dim))

        decoder_dims = [bottleneck_dim, *reversed(hidden_dims)]
        decoder_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(decoder_dims[:-1], decoder_dims[1:]):
            decoder_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class AutoencoderConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (512, 256)
    bottleneck_dim: int = 64
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    weight_decay: float = 0.0
    val_split: float = 0.1
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_autoencoder(
    agents: Sequence[ppo.PPOAgent],
    cfg: AutoencoderConfig,
    callbacks: Iterable[Callable[..., None]] | None = None,
) -> tuple[WeightAutoencoder, dict]:
    """Train a weight autoencoder and return both the model and its loss history.

    Raises:
        TypeError: If `vectors` is neither a Tensor nor a Dataset.
        ValueError: If the validation split is invalid or the dataset is too small.
    """

    vectors = VectorDataset(agents)  #takes list of tensor and convert to dataset

    torch.manual_seed(cfg.seed)

    train_len, val_len = _split_lengths(len(vectors), cfg.val_split)
    train_ds, val_ds = random_split(vectors, [train_len, val_len])

    pin_memory = cfg.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )

    model = WeightAutoencoder(cfg.input_dim, cfg.hidden_dims, cfg.bottleneck_dim).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    callback_list = list(callbacks or [])

    progress_bar = tqdm(range(1, cfg.epochs + 1), desc="Training Autoencoder")
    for epoch in progress_bar:
        train_loss = _run_epoch(model, train_loader, criterion, cfg.device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, cfg.device, None)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        progress_bar.set_postfix(
            train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}"
        )

        for cb in callback_list:
            cb(epoch=epoch, model=model, train_loss=train_loss, val_loss=val_loss)

    return model, history


def save_autoencoder(
    model: WeightAutoencoder,
    path: str | os.PathLike[str],
    cfg: AutoencoderConfig | None = None,
) -> Path:
    """Persist the autoencoder weights and optionally its config, returning the saved path."""
    checkpoint: dict[str, Any] = {"state_dict": model.state_dict()}
    if cfg is not None:
        checkpoint["config"] = asdict(cfg)
    target_path = Path(path)

    # Treat missing suffix or trailing separator as a directory intent.
    directory_intent = (
        (target_path.exists() and target_path.is_dir())
        or str(path).endswith(os.sep)
        or (target_path.suffix == "" and not target_path.parent.exists())
    )
    if directory_intent:
        target_path.mkdir(parents=True, exist_ok=True)
        target_path = target_path / "autoencoder.pt"
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, target_path)
    return target_path


def load_autoencoder(
    path: str | os.PathLike[str],
    cfg: AutoencoderConfig | None = None,
    device: str | torch.device | None = None,
) -> tuple[WeightAutoencoder, AutoencoderConfig]:
    """Restore the autoencoder and return the model along with the resolved config."""
    checkpoint = torch.load(path, map_location=device or "cpu")

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError("Checkpoint is missing 'state_dict'.")

    saved_cfg = checkpoint.get("config")
    if cfg is None:
        if saved_cfg is None:
            raise ValueError("AutoencoderConfig is required to instantiate the model.")
        cfg = AutoencoderConfig(**saved_cfg)

    model = WeightAutoencoder(cfg.input_dim, cfg.hidden_dims, cfg.bottleneck_dim)
    model.load_state_dict(state_dict)
    target_device = device or cfg.device
    model.to(target_device)
    return model, cfg


def _split_lengths(total_len: int, val_split: float) -> tuple[int, int]:

    val_len  = math.floor(total_len * val_split)
    train_len = total_len - val_len

    if train_len <= 0:
        raise ValueError("Not enough training samples; reduce val_split or increase dataset size.")

    return train_len, val_len


def _run_epoch(
    model: WeightAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: optim.Optimizer | None,
) -> float:
    if optimizer is None:
        model.eval()
        context = torch.no_grad()
    else:
        model.train()
        context = torch.enable_grad()

    total_loss = 0.0
    total_items = 0

    with context:
        for batch in loader:
            batch = batch.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            recon = model(batch)
            loss = criterion(recon, batch)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch.size(0)
            total_items += batch.size(0)

    return total_loss / total_items




def build_ppo_agent(
    game: pyspiel.Game,
    device: str,
    init_fn: Initializer,
) -> ppo.PPOAgent:
    """Instantiate a randomly initialized PPOAgent for the given game."""
    num_actions = game.num_distinct_actions()
    observation_shape = game.information_state_tensor_shape()
    return ppo.PPOAgent(num_actions, observation_shape, device, init_fn)


def generate_random_agents(
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
    utils.game = game
    num_actions = game.num_distinct_actions()
    init_fn = utils.diverse_random_kuhn_poker_layer_init

    encoded_vectors: list[torch.Tensor] = []
    for _ in range(num_agents):
        agent = build_ppo_agent(game, device, init_fn)
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
        default='checkpoints/',
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

if __name__ == "__main__":
    args = _parse_args()


    agents = generate_random_agents(
        num_agents=args.num_agents,
        seed=args.seed,
        device=args.device,
        game_name=args.game,
    )

    cfg = AutoencoderConfig(
        input_dim=agents.shape[1],
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
    model, history = train_autoencoder(agents, cfg)
    
    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    print(f"Trained autoencoder. Final train loss: {final_train:.6f}, val loss: {final_val:.6f}")

    if args.autoencoder_output is not None:
        checkpoint_path = save_autoencoder(model, args.autoencoder_output, cfg)
        print(f"Saved autoencoder checkpoint to {checkpoint_path}