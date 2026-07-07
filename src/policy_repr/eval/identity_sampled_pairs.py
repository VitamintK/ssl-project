"""
Evaluate the identity encoder on Task B using sampled random pairs.

This avoids materializing or computing the full N x N payoff matrix.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pyspiel
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from policy_repr.datasets.generate_agents import load_agents
from policy_repr.downstream.heads import set_seed
from policy_repr.utils import PPOAgentPolicy, get_expected_payoffs
from policy_repr.encoders.weight_autoencoder import ppo_agent_to_vector


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


class SampledPairDataset(Dataset):
    def __init__(self, embeddings, pairs, payoffs):
        self.embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.payoffs = torch.as_tensor(payoffs, dtype=torch.float32)

    def __len__(self):
        return len(self.payoffs)

    def __getitem__(self, idx):
        p1, p2 = self.pairs[idx]
        return torch.cat((self.embeddings[p1], self.embeddings[p2])), self.payoffs[idx]


def sample_pairs(rng, p1_pool, p2_pool, num_pairs):
    p1 = rng.choice(np.asarray(list(p1_pool), dtype=np.int64), size=num_pairs, replace=True)
    p2 = rng.choice(np.asarray(list(p2_pool), dtype=np.int64), size=num_pairs, replace=True)
    return np.stack([p1, p2], axis=1)


def compute_payoffs(game, agents, pairs, desc):
    payoffs = []
    policy_cache = {}

    def policy(agent_idx, player_id):
        key = (int(agent_idx), int(player_id))
        if key not in policy_cache:
            policy_cache[key] = PPOAgentPolicy(game, agents[int(agent_idx)], int(player_id), False)
        return policy_cache[key]

    for p1_idx, p2_idx in tqdm(pairs, desc=desc):
        payoffs.append(get_expected_payoffs(game, policy(p1_idx, 0), policy(p2_idx, 1)))
    return np.asarray(payoffs, dtype=np.float32)


def train_linear_probe(embeddings, train_pairs, train_payoffs, val_pairs, val_payoffs, args):
    embed_dim = np.asarray(embeddings).shape[1] * 2
    model = torch.nn.Linear(embed_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(
        SampledPairDataset(embeddings, train_pairs, train_payoffs),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SampledPairDataset(embeddings, val_pairs, val_payoffs),
        batch_size=args.batch_size,
        shuffle=False,
    )

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_items = 0
        for batch_x, batch_y in train_loader:
            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
            train_items += len(batch_y)

        model.eval()
        val_loss = 0.0
        val_items = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x).squeeze(-1)
                loss = criterion(pred, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_items += len(batch_y)

        train_mse = train_loss / train_items
        val_mse = val_loss / val_items
        best_val = min(best_val, val_mse)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{args.epochs}: train={train_mse:.6f}, val={val_mse:.6f}")

    baseline_mean = float(np.mean(train_payoffs))
    baseline_mse = float(np.mean((val_payoffs - baseline_mean) ** 2))
    improvement = (1.0 - best_val / baseline_mse) * 100.0
    return best_val, baseline_mse, improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--agent-pool", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-pairs", type=int, default=10000)
    parser.add_argument("--num-val-pairs", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cache-dir", default="sampled_payoffs")
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    game = pyspiel.load_game(args.game)
    _, eval_agents = load_agents(args.agent_pool, game)
    n_agents = len(eval_agents)

    logger.info(f"Computing identity embeddings for {n_agents} agents...")
    embeddings = [ppo_agent_to_vector(agent).detach().cpu().numpy() for agent in eval_agents]
    logger.info(f"Embedding dim: {embeddings[0].shape[0]}")

    n_val = max(1, int(n_agents * 0.2))
    perm = rng.permutation(n_agents)
    val_set = set(perm[:n_val])
    train_set = set(perm[n_val:])
    train_pairs = sample_pairs(rng, train_set, train_set, args.num_train_pairs)
    val_pairs = sample_pairs(rng, val_set, val_set, args.num_val_pairs)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_game = args.game.replace("(", "_").replace(")", "").replace(",", "_").replace("=", "")
    cache_path = cache_dir / (
        f"identity_task_b_{safe_game}_seed{args.seed}_"
        f"train{args.num_train_pairs}_val{args.num_val_pairs}.npz"
    )

    if cache_path.exists():
        logger.info(f"Loading sampled payoffs from {cache_path}")
        data = np.load(cache_path)
        train_pairs = data["train_pairs"]
        val_pairs = data["val_pairs"]
        train_payoffs = data["train_payoffs"]
        val_payoffs = data["val_payoffs"]
    else:
        logger.info(f"Computing {len(train_pairs)} train sampled pair payoffs...")
        train_payoffs = compute_payoffs(game, eval_agents, train_pairs, "Train pairs")
        logger.info(f"Computing {len(val_pairs)} val sampled pair payoffs...")
        val_payoffs = compute_payoffs(game, eval_agents, val_pairs, "Val pairs")
        np.savez(
            cache_path,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            train_payoffs=train_payoffs,
            val_payoffs=val_payoffs,
        )
        logger.info(f"Saved sampled payoffs to {cache_path}")

    mse, baseline, improvement = train_linear_probe(
        embeddings, train_pairs, train_payoffs, val_pairs, val_payoffs, args
    )
    logger.info("=" * 80)
    logger.info(f"Task B sampled identity results: {args.game} seed={args.seed}")
    logger.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"Baseline: {baseline:.6f}")
    logger.info(f"Improvement: {improvement:.2f}%")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
