import argparse
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from iig_rl_benchmark.algorithms.ppo import ppo

from pathlib import Path
from dataclasses import asdict

class VectorDataset(Dataset):
    """Simple dataset wrapper for 2D weight tensors."""

    def __init__(self, vectors: torch.Tensor):
        if vectors.ndim != 2:
            raise ValueError("vectors must be (num_samples, dim)")
        self.vectors = vectors.float()

    def __len__(self) -> int:
        return self.vectors.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.vectors[idx]


class Autoencoder(nn.Module):
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
    hidden_dims: tuple[int, ...] = (512, 256)
    bottleneck_dim: int = 64
    lr: float = 3e-4
    batch_size: int = 64
    epochs: int = 15
    weight_decay: float = 0.0
    val_split: float = 0.1
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class WeightAutoencoder:
    def __init__(self, cfg: AutoencoderConfig, policies: list[any], policy_to_vector: Callable):
        self.cfg = cfg
        self.policies = policies
        self.policy_to_vector = policy_to_vector

    def train(self, callbacks: Iterable[Callable[..., None]] | None = None):
        """Train a weight autoencoder and return both the model and its loss history.
        Raises:
            TypeError: If `vectors` is neither a Tensor nor a Dataset.
            ValueError: If the validation split is invalid or the dataset is too small.
        """
        cfg = self.cfg
        vectors = [self.policy_to_vector(policy) for policy in self.policies]
        vectors = torch.stack(vectors)
        input_dim = vectors.shape[1]
        vectors = VectorDataset(vectors)
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

        self.autoencoder = Autoencoder(input_dim, cfg.hidden_dims, cfg.bottleneck_dim).to(cfg.device)
        optimizer = optim.AdamW(self.autoencoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        criterion = nn.MSELoss()

        history = {"train_loss": [], "val_loss": []}
        callback_list = list(callbacks or [])

        progress_bar = tqdm(range(1, cfg.epochs + 1), desc="Training Autoencoder")
        for epoch in progress_bar:
            train_loss = self._run_epoch(self.autoencoder, train_loader, criterion, cfg.device, optimizer)
            val_loss = self._run_epoch(self.autoencoder, val_loader, criterion, cfg.device, None)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            progress_bar.set_postfix(
                train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}"
            )
            for cb in callback_list:
                cb(epoch=epoch, model=self.autoencoder, train_loss=train_loss, val_loss=val_loss)

        return self.autoencoder, history
    
    def _run_epoch(
        self,
        model: nn.Module,
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



    def get_encoder(self, device: str = "cpu") -> callable:
        """
        Get the encoder part of a trained autoencoder as a callable function.
        Args:
            device: Device to run the encoder on
        Returns:
            encoder_fn: Function that takes a weight vector and returns its bottleneck embedding
        """
        self.autoencoder.eval()
        self.autoencoder.to(device)
        def encoder_fn(policy: any) -> torch.Tensor:
            """Encode weights into bottleneck representation."""
            with torch.no_grad():
                vector = self.policy_to_vector(policy)
                if not isinstance(vector, torch.Tensor):
                    vector = torch.tensor(vector)
                vector = vector.float().to(device)
                if vector.ndim == 1:
                    vector = vector.unsqueeze(0)
                embedding = self.autoencoder.encoder(vector)
                return embedding.squeeze(0)
        return encoder_fn

def ppo_agent_to_vector(ppo_agent: ppo.PPOAgent) -> torch.Tensor:
    parameters = [param.detach() for param in ppo_agent.actor.parameters()]
    parameters = nn.utils.parameters_to_vector(parameters)
    return parameters

def _split_lengths(total_len: int, val_split: float) -> tuple[int, int]:

    val_len  = math.floor(total_len * val_split)
    train_len = total_len - val_len

    if train_len <= 0:
        raise ValueError("Not enough training samples; reduce val_split or increase dataset size.")

    return train_len, val_len

def save_autoencoder(
    model: WeightAutoencoder,
    cfg: AutoencoderConfig,
    path: str | Path,
) -> None:
    """Persist the autoencoder weights along with the config metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict(), "config": asdict(cfg)} 
    torch.save(checkpoint, path)


def load_autoencoder(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[WeightAutoencoder, AutoencoderConfig]:
    """Load a saved autoencoder checkpoint."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    cfg_dict = checkpoint.get("config")
    if cfg_dict is None:
        raise ValueError("Checkpoint is missing autoencoder config metadata.")

    cfg = AutoencoderConfig(**cfg_dict)

    model = WeightAutoencoder(cfg.input_dim, cfg.hidden_dims, cfg.bottleneck_dim)
    model.load_state_dict(checkpoint["state_dict"])
    if device is not None:
        model = model.to(device)
        cfg.device = str(device)
    return model, cfg

def _parse_args() -> argparse.Namespace:
    def hidden_dims_arg(value: str) -> tuple[int, ...]:
        dims = [int(v.strip()) for v in value.split(",") if v.strip()]
        if not dims:
            raise argparse.ArgumentTypeError("hidden dims must not be empty")
        return tuple(dims)

    parser = argparse.ArgumentParser(description="Encode random PPO agents for weight autoencoding.")
    parser.add_argument("--num-agents", type=int, default=1000, help="Number of random PPO agents to encode.")
    parser.add_argument("--ppo-agent-hidden-size", type=int, default=256, help="Hidden size for the PPO agent.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", type=str, default=None, help="Torch device for agent instantiation.")
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
    parser.add_argument("--hidden-dims", type=hidden_dims_arg, default=(1024, 512), help="Comma separated hidden dims.")
    parser.add_argument("--bottleneck-dim", type=int, default=64, help="Autoencoder latent dimension size.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for the autoencoder.")
    parser.add_argument("--batch-size", type=int, default=16, help="Autoencoder training batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for autoencoder training.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for optimizer.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split proportion.")
    return parser.parse_args()

if __name__ == "__main__":
    import pyspiel
    from utils import make_diverse_random_kuhn_poker_layer_init, get_device_string
    from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent

    args = _parse_args()

    if (device := args.device) is None:
        device = get_device_string()
    print(f"Using device: {device}")
    
    game = pyspiel.load_game(args.game)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    # Train autoencoder on agent weights
    ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, args.ppo_agent_hidden_size) for i in range(args.num_agents)]
    print("\nTraining autoencoder on agent weights...")
    ae_config = AutoencoderConfig(
        hidden_dims=args.hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        seed=args.seed if args.seed is not None else 0,
        device=device,
    )
    weight_autoencoder = WeightAutoencoder(ae_config, ppo_agents, ppo_agent_to_vector)
    _, ae_history = weight_autoencoder.train()
    final_train = ae_history["train_loss"][-1]
    final_val = ae_history["val_loss"][-1]
    print(f"Trained autoencoder. Final train loss: {final_train:.6f}, val loss: {final_val:.6f}")