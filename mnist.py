from __future__ import annotations

import argparse
import copy
import dataclasses
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms


# Allow running this script directly without installing as a package.
THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

DEFAULT_CHECKPOINT_PATH = THIS_DIR / "checkpoints" / "mnist.pt"

from weight_autoencoder import AutoencoderConfig, train_autoencoder  # type: ignore  # noqa: E402


@dataclass
class ParameterMetadata:
    name: str
    shape: torch.Size
    numel: int
    dtype: torch.dtype


def module_to_vector(module: nn.Module) -> tuple[torch.Tensor, List[ParameterMetadata]]:
    """Flatten all named parameters of the module into a single vector.

    Returns:
        vector: 1D tensor containing all parameters concatenated.
        metadata: shape/dtype info for rebuilding the parameters.
    """
    flat_params: list[torch.Tensor] = []
    metadata: list[ParameterMetadata] = []

    for name, param in module.named_parameters():
        metadata.append(
            ParameterMetadata(
                name=name,
                shape=param.data.shape,
                numel=param.data.numel(),
                dtype=param.data.dtype,
            )
        )
        flat_params.append(param.detach().reshape(-1).cpu())

    if not flat_params:
        raise ValueError("module_to_vector received a module with no parameters.")

    vector = torch.cat(flat_params, dim=0)
    return vector, metadata


def vector_to_module(
    vector: torch.Tensor,
    module: nn.Module,
    metadata: Sequence[ParameterMetadata],
) -> None:
    """Load flattened parameters back into the module using stored metadata."""
    params_dict = dict(module.named_parameters())
    offset = 0

    with torch.no_grad():
        for meta in metadata:
 
            numel = meta.numel
            slice_ = vector[offset : offset + numel]

            param = params_dict[meta.name]
            reshaped = slice_.to(param.device, dtype=meta.dtype).view(meta.shape)
            param.copy_(reshaped)
            offset += numel



class MNISTMLP(nn.Module):
    def __init__(self, hidden_sizes: Iterable[int] = (256, 128)):
        super().__init__()
        input_dim = 28 * 28
        sizes: List[int] = [input_dim, *hidden_sizes, 10]
        layers: list[nn.Module] = []

        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 10:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


@dataclass
class MNISTConfig:
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 5
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_ds = datasets.MNIST(root=str(THIS_DIR / "data"), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(THIS_DIR / "data"), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total


def train_classifier(cfg: MNISTConfig) -> tuple[MNISTMLP, dict, DataLoader]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_loader, test_loader = get_dataloaders(cfg.batch_size)

    model = MNISTMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_acc": []}

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)

        print(f"Epoch {epoch+1}/{cfg.epochs}: loss={train_loss:.4f}, test_acc={test_acc:.4f}")

    return model, history, test_loader


def save_classifier_checkpoint(
    model: MNISTMLP,
    checkpoint_path: pathlib.Path,
    cfg: MNISTConfig,
    history: dict,
) -> None:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "mnist_config": dataclasses.asdict(cfg),
        "history": history,
    }
    torch.save(payload, str(checkpoint_path))


def load_classifier_checkpoint(
    checkpoint_path: pathlib.Path,
    device: torch.device,
) -> tuple[MNISTMLP, dict]:
    checkpoint_path = checkpoint_path.resolve()

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model = MNISTMLP().to(device)
    model.load_state_dict(state_dict)
    return model, checkpoint


class NoisyVectorDataset(Dataset):
    def __init__(self, base_vector: torch.Tensor, samples: int, noise_std: float, seed: int | None = None, include_original: bool = True):

        self.base = base_vector.detach().cpu().float()
        self.samples = samples
        self.noise_std = noise_std
        self.include_original = include_original
        self.seed = 0 if seed is None else seed

    def __len__(self) -> int:
        return self.samples

    def _noise_for_index(self, idx: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        noise = torch.randn(self.base.shape, generator=generator, dtype=self.base.dtype)
        return noise * self.noise_std

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.include_original and idx == 0:
            return self.base.clone()
        return self.base + self._noise_for_index(idx)



def build_autoencoder_dataset(
    weight_vector: torch.Tensor,
    samples: int = 4096,
    noise_std: float = 0.01,
    *,
    include_original: bool = True,
    seed: int | None = None,
) -> Dataset:
    """Return a dataset that samples noisy variants of `weight_vector` on demand.

    Args:
        weight_vector: Flattened parameter vector used as the center of the distribution.
        samples: Total number of items the dataset should expose (includes the original
            vector when `include_original` is True).
        noise_std: Standard deviation of the isotropic Gaussian noise applied to the base.
        include_original: Whether to ensure the first sample equals the pristine weights.
        seed: Optional seed for deterministic noise sampling.
    """

    base_vector = weight_vector.detach().cpu().float()
    return NoisyVectorDataset(
        base_vector,
        samples=samples,
        noise_std=noise_std,
        seed=seed,
        include_original=include_original,
    )


def mnist_weight_autoencoder_demo(
    mnist_cfg: MNISTConfig,
    ae_cfg: AutoencoderConfig,
    ae_samples: int = 2048,
    ae_noise_std: float = 0.01,
    *,
    test_encoder: bool = False,
    checkpoint_path: pathlib.Path | None = None,
) -> None:
    """Train MNIST classifier, autoencode its weights, and report accuracy delta."""
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
    device = torch.device(mnist_cfg.device)
    if test_encoder:
        model, _ = load_classifier_checkpoint(checkpoint_path, device)
        print(f"Loaded MNIST checkpoint from {checkpoint_path}")
        _, test_loader = get_dataloaders(mnist_cfg.batch_size)
        base_acc = evaluate(model, test_loader, device)
    else:
        model, history, test_loader = train_classifier(mnist_cfg)
        save_classifier_checkpoint(model, checkpoint_path, mnist_cfg, history)
        print(f"Saved MNIST checkpoint to {checkpoint_path}")
        base_acc = history["test_acc"][-1]

    print(f"Baseline accuracy: {base_acc:.4f}")

    weight_vector, metadata = module_to_vector(model)
    if ae_cfg.input_dim != weight_vector.numel():
        ae_cfg = dataclasses.replace(ae_cfg, input_dim=weight_vector.numel())
    print('training autoencoder with input_dim=', ae_cfg.input_dim)

    ae_dataset = build_autoencoder_dataset(
        weight_vector,
        samples=ae_samples,
        noise_std=ae_noise_std,
        seed=mnist_cfg.seed,
    )

    def log_autoencoder_epoch(**kwargs: object) -> None:
        epoch = kwargs.get("epoch")
        train_loss = kwargs.get("train_loss")
        val_loss = kwargs.get("val_loss")
        print(f"AE Epoch {epoch+1}/{ae_cfg.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    autoencoder, ae_history = train_autoencoder(ae_dataset, ae_cfg, callbacks=[log_autoencoder_epoch])
    print("Autoencoder training complete.")

    with torch.no_grad():
        reconstructed_vector = autoencoder(weight_vector.unsqueeze(0).to(ae_cfg.device)).squeeze(0).cpu()
    reconstruction_mse = F.mse_loss(reconstructed_vector, weight_vector).item()

    print(f"Autoencoder reconstruction MSE: {reconstruction_mse:.6f}")
    recon_model = copy.deepcopy(model).to(device)
    vector_to_module(reconstructed_vector, recon_model, metadata)
    recon_acc = evaluate(recon_model, test_loader, device)

    print("--- Results ---")
    print(f"Baseline accuracy:      {base_acc:.4f}")
    print(f"Reconstructed accuracy: {recon_acc:.4f}")
    print(f"Accuracy delta:         {base_acc - recon_acc:+.4f}")
    print(f"Weight MSE:             {reconstruction_mse:.6f}")
    print(f"Autoencoder final losses: train={ae_history['train_loss'][-1]:.6f}, val={ae_history['val_loss'][-1]:.6f}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST classifier and autoencode its weights.")
    parser.add_argument(
        "--test-encoder",
        action="store_true",
        help="Skip MNIST training and load the classifier weights from a checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to MNIST checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    mnist_cfg = MNISTConfig(epochs=5)
    ae_cfg = AutoencoderConfig(
        input_dim=0,
        hidden_dims=(256, 256),
        bottleneck_dim=128,
        epochs=5,
        batch_size=256,
        lr=1e-3,
    )

    mnist_weight_autoencoder_demo(
        mnist_cfg,
        ae_cfg,
        ae_noise_std=2,
        ae_samples=4096,
        test_encoder=args.test_encoder,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
