"""
Evaluate any encoder on Tasks A and B using precomputed payoff matrices.

Supports: trajectory, grover, tabular encoders.
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
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import get_all_states

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import PPOAgentPolicy, get_device_string, make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed
from generate_agents import load_agents
from trajectory_encoder import TrajectoryEncoderConfig

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
    from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint
    _, adapter, _ = load_trajectory_encoder_from_checkpoint(
        checkpoint_path, game, policies=pretrain_agents, device=device
    )
    encoder_fn = adapter.get_encoder(device=device)
    return [encoder_fn(agent).detach().cpu().numpy() for agent in agents]


def compute_grover_embeddings(agents, game, checkpoint_path, pretrain_agents, device):
    """Compute Grover encoder embeddings."""
    from test_grover_encoder import load_grover_from_checkpoint
    _, adapter, _ = load_grover_from_checkpoint(
        checkpoint_path, game, policies=pretrain_agents, device=device
    )
    encoder_fn = adapter.get_encoder(device=device)
    return [encoder_fn(agent).detach().cpu().numpy() for agent in agents]


# ---- Evaluation with precomputed payoffs ----

def eval_task(task_name, embeddings, payoffs, pair_indices, seed):
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
        # Task B: concatenate p1 and p2 embeddings
        X = np.array([
            np.concatenate([embeddings_arr[p1], embeddings_arr[p2]])
            for p1, p2 in pair_indices
        ])
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

    X_train = torch.FloatTensor(X[train_idx])
    y_train = torch.FloatTensor(y[train_idx])
    X_val = torch.FloatTensor(X[val_idx])
    y_val = torch.FloatTensor(y[val_idx])

    # Linear probe
    embed_dim = X.shape[1]
    model = torch.nn.Linear(embed_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0
        n_batches = 0
        for i in range(0, len(X_train), 16):
            batch_idx = perm[i:i+16]
            pred = model(X_train[batch_idx]).squeeze(-1)
            loss = criterion(pred, y_train[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze(-1)
            val_loss = criterion(val_pred, y_val).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: train={epoch_loss/n_batches:.6f}, val={val_loss:.6f}")

    # Baseline: mean predictor
    mean_payoff = y_train.mean().item()
    baseline_mse = ((y_val - mean_payoff) ** 2).mean().item()

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
                        choices=["trajectory", "grover", "tabular"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to encoder checkpoint (not needed for tabular)")
    parser.add_argument("--agent-pool", type=str, required=True,
                        help="Path to saved agent pool")
    parser.add_argument("--payoffs", type=str, required=True,
                        help="Path to precomputed payoffs (.npz)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
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
    if args.encoder == "tabular":
        embeddings = compute_tabular_embeddings(eval_agents, game)
    elif args.encoder == "trajectory":
        embeddings = compute_trajectory_embeddings(
            eval_agents, game, args.checkpoint, pretrain_agents, device)
    elif args.encoder == "grover":
        embeddings = compute_grover_embeddings(
            eval_agents, game, args.checkpoint, pretrain_agents, device)

    # Evaluate
    a_mse, a_base, a_imp = eval_task("Task A: Fixed opponent payoff prediction",
                                      embeddings, task_a_payoffs, None, args.seed)
    b_mse, b_base, b_imp = eval_task("Task B: Agent matchup payoff prediction",
                                      embeddings, task_b_payoffs, task_b_pairs, args.seed)

    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Task A: MSE={a_mse:.6f}  Baseline={a_base:.6f}  Imp={a_imp:.2f}%")
    logger.info(f"Task B: MSE={b_mse:.6f}  Baseline={b_base:.6f}  Imp={b_imp:.2f}%")
    logger.info(f"{'='*80}")
