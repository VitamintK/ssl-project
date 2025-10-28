import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


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
    vectors: Union[torch.Tensor, Dataset],
    cfg: AutoencoderConfig,
    callbacks: Iterable[Callable[..., None]] | None = None,
) -> tuple[WeightAutoencoder, dict]:
    """Train a weight autoencoder and return both the model and its loss history.

    Raises:
        TypeError: If `vectors` is neither a Tensor nor a Dataset.
        ValueError: If the validation split is invalid or the dataset is too small.
    """
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


if __name__ == "__main__":
    print("Training Weight Autoencoder on synthetic data...")
    dim, samples = 1024, 10_000
    synthetic_vectors = torch.randn(samples, dim)

    cfg = AutoencoderConfig(input_dim=dim, bottleneck_dim=32, hidden_dims=(256, 128), epochs=10)
    model, hist = train_autoencoder(synthetic_vectors, cfg)

    print(hist["train_loss"][-1], hist["val_loss"][-1])
