"""
Evaluate any encoder on Tasks A and B using precomputed payoff matrices.

Supports: identity, trajectory, grover, tabular encoders.
Uses precomputed payoff matrices to avoid redundant computation.

Usage:
    python eval_with_payoffs.py --game kuhn_poker --encoder trajectory \
        --checkpoint checkpoints/trajectory_encoder_kuhn_random_improved_500.pt \
        --agent-pool agent_pools/kuhn_poker_seed42_n500.pt \
        --payoffs payoff_matrices/kuhn_poker_seed42_payoffs.npz \
        --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import pyspiel
from torch.utils.data import DataLoader, Dataset, TensorDataset
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import get_all_states

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from policy_repr.utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from policy_repr.downstream.heads import PayoffPredictor, set_seed
from policy_repr.datasets.generate_agents import load_agents
from policy_repr.encoders.trajectory import TrajectoryEncoderConfig
from policy_repr.encoders.weight_autoencoder import ppo_agent_to_vector

# Inject TrajectoryEncoderConfig into __main__ for unpickling checkpoints saved from __main__
sys.modules['__main__'].TrajectoryEncoderConfig = TrajectoryEncoderConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/eval_with_payoffs.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)


# ---- Embedding functions ----

def compute_tabular_embeddings(agents, game):
    """Compute tabular action-distribution embeddings."""
    all_states_dict = get_all_states.get_all_states(game)
    decision_states = [
        state for state in all_states_dict.values()
        if not state.is_terminal() and not state.is_chance_node()
    ]
    num_actions = game.num_distinct_actions()
    logger.info(f"  Decision states: {len(decision_states)}, embed dim: {len(decision_states) * num_actions}")

    embeddings = []
    for agent in agents:
        parts = []
        for state in decision_states:
            player_id = state.current_player()
            legal_actions = state.legal_actions(player_id)
            legal_action_mask = torch.zeros(num_actions)
            legal_action_mask[legal_actions] = 1
            info_state = torch.Tensor(state.information_state_tensor(player_id))
            with torch.no_grad():
                _, _, _, _, probs = agent.get_action_and_value(info_state, legal_action_mask)
            parts.append(probs.detach().cpu().numpy())
        embeddings.append(np.concatenate(parts))
    return embeddings


def compute_trajectory_embeddings(agents, game, checkpoint_path, pretrain_agents, device):
    """Compute trajectory encoder embeddings."""
    from policy_repr.encoders.trajectory import load_trajectory_encoder_from_checkpoint
    _, adapter, _ = load_trajectory_encoder_from_checkpoint(
        checkpoint_path, game, policies=pretrain_agents, device=device
    )
    encoder_fn = adapter.get_encoder(device=device)
    return [encoder_fn(agent).detach().cpu().numpy() for agent in agents]


def compute_grover_embeddings(agents, game, checkpoint_path, pretrain_agents, device):
    """Compute Grover encoder embeddings."""
    from policy_repr.encoders.grover import load_grover_from_checkpoint
    _, adapter, _ = load_grover_from_checkpoint(
        checkpoint_path, game, policies=pretrain_agents, device=device
    )
    encoder_fn = adapter.get_encoder(device=device)
    return [encoder_fn(agent).detach().cpu().numpy() for agent in agents]


def compute_identity_embeddings(agents):
    """Use raw PPO actor parameters as embeddings."""
    embeddings = []
    for agent in agents:
        embeddings.append(ppo_agent_to_vector(agent).detach().cpu().numpy())
    logger.info(f"  Embed dim: {embeddings[0].shape[0]}")
    return embeddings


# ---- Evaluation with precomputed payoffs ----

class PairEmbeddingDataset(Dataset):
    """Dataset that concatenates pair embeddings lazily to avoid O(N^2D) memory."""

    def __init__(self, embeddings, pair_indices, payoffs, indices):
        self.embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
        self.pair_indices = np.asarray(pair_indices, dtype=np.int64)
        self.payoffs = torch.as_tensor(payoffs, dtype=torch.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pair_idx = self.indices[idx]
        p1, p2 = self.pair_indices[pair_idx]
        x = torch.cat((self.embeddings[p1], self.embeddings[p2]))
        return x, self.payoffs[pair_idx]


def _evaluate_loader(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            total_loss += loss.item() * len(batch_y)
            total_items += len(batch_y)
    return total_loss / total_items


def eval_task(task_name, embeddings, payoffs, pair_indices, seed, batch_size=16):
    """Run linear probe evaluation using precomputed payoffs."""
    set_seed(seed)  # Ensure identical train/val split across all encoders

    logger.info(f"\n{'='*80}")
    logger.info(f"{task_name}")
    logger.info(f"{'='*80}")

    embeddings_arr = np.array(embeddings)
    n_agents = len(embeddings)

    if pair_indices is None:
        # Task A: single agent embeddings, no concatenation needed
        # pair_indices is just [(i, 0) for i in range(n_agents)]
        X = embeddings_arr
        y = payoffs
    else:
        X = None
        y = payoffs

    # Train/val split on P1 agents
    n_p1 = n_agents
    n_val = max(1, int(n_p1 * 0.2))
    p1_perm = np.random.permutation(n_p1)
    val_set = set(p1_perm[:n_val])
    train_set = set(p1_perm[n_val:])

    if pair_indices is None:
        train_idx = np.array([i for i in range(len(y)) if i in train_set])
        val_idx = np.array([i for i in range(len(y)) if i in val_set])
    else:
        # Also split P2
        n_val_p2 = max(1, int(n_p1 * 0.2))
        p2_perm = np.random.permutation(n_p1)
        val_p2_set = set(p2_perm[:n_val_p2])
        train_p2_set = set(p2_perm[n_val_p2:])
        train_idx = np.array([i for i, (p1, p2) in enumerate(pair_indices)
                              if p1 in train_set and p2 in train_p2_set])
        val_idx = np.array([i for i, (p1, p2) in enumerate(pair_indices)
                            if p1 in val_set and p2 in val_p2_set])

    logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Linear probe
    embed_dim = embeddings_arr.shape[1] if pair_indices is None else embeddings_arr.shape[1] * 2
    model = torch.nn.Linear(embed_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    if pair_indices is None:
        X_train = torch.FloatTensor(X[train_idx])
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor(X[val_idx])
        y_val = torch.FloatTensor(y[val_idx])
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    else:
        train_ds = PairEmbeddingDataset(embeddings_arr, pair_indices, y, train_idx)
        val_ds = PairEmbeddingDataset(embeddings_arr, pair_indices, y, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            pred = model(batch_x).squeeze(-1)
            loss = criterion(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_loss = _evaluate_loader(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: train={epoch_loss/n_batches:.6f}, val={val_loss:.6f}")

    # Baseline: mean predictor
    mean_payoff = float(np.mean(y[train_idx]))
    baseline_mse = float(np.mean((y[val_idx] - mean_payoff) ** 2))

    mse = best_val_loss
    imp = (1 - mse / baseline_mse) * 100

    logger.info(f"\n  Results:")
    logger.info(f"    MSE:      {mse:.6f}")
    logger.info(f"    Baseline: {baseline_mse:.6f}")
    logger.info(f"    Improvement: {imp:.2f}%")

    return mse, baseline_mse, imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate encoder on Tasks A+B with precomputed payoffs")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True,
                        choices=["identity", "trajectory", "grover", "tabular"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to encoder checkpoint (not needed for tabular)")
    parser.add_argument("--agent-pool", type=str, required=True,
                        help="Path to saved agent pool")
    parser.add_argument("--payoffs", type=str, required=True,
                        help="Path to precomputed payoffs (.npz)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = args.device or get_device_string()
    logger.info(f"Game: {args.game}")
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {device}")

    game = pyspiel.load_game(args.game)

    # Load agents
    logger.info(f"\nLoading agents from {args.agent_pool}...")
    pretrain_agents, eval_agents = load_agents(args.agent_pool, game)

    # Load precomputed payoffs
    logger.info(f"Loading payoffs from {args.payoffs}...")
    payoff_data = np.load(args.payoffs, allow_pickle=True)
    task_a_payoffs = payoff_data["task_a_payoffs"]
    task_b_payoffs = payoff_data["task_b_payoffs"]
    task_b_pairs = [tuple(p) for p in payoff_data["task_b_pair_indices"]]
    logger.info(f"  Task A: {len(task_a_payoffs)} payoffs")
    logger.info(f"  Task B: {len(task_b_payoffs)} payoffs")

    # Compute embeddings
    logger.info(f"\nComputing {args.encoder} embeddings...")
    if args.encoder == "identity":
        embeddings = compute_identity_embeddings(eval_agents)
    elif args.encoder == "tabular":
        embeddings = compute_tabular_embeddings(eval_agents, game)
    elif args.encoder == "trajectory":
        embeddings = compute_trajectory_embeddings(
            eval_agents, game, args.checkpoint, pretrain_agents, device)
    elif args.encoder == "grover":
        embeddings = compute_grover_embeddings(
            eval_agents, game, args.checkpoint, pretrain_agents, device)

    # Evaluate
    a_mse, a_base, a_imp = eval_task("Task A: Fixed opponent payoff prediction",
                                      embeddings, task_a_payoffs, None, args.seed, args.batch_size)
    b_mse, b_base, b_imp = eval_task("Task B: Agent matchup payoff prediction",
                                      embeddings, task_b_payoffs, task_b_pairs, args.seed, args.batch_size)

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Task A: MSE={a_mse:.6f}  Baseline={a_base:.6f}  Imp={a_imp:.2f}%")
    logger.info(f"Task B: MSE={b_mse:.6f}  Baseline={b_base:.6f}  Imp={b_imp:.2f}%")
    logger.info(f"{'='*80}")
