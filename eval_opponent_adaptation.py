"""
Downstream application: Embedding-based opponent-adaptive strategy selection.

Given a novel opponent observed through a small number of trajectories,
use pre-trained embeddings to select the best counter-strategy from a
library of agents with known pairwise payoffs.

Usage:
    python eval_opponent_adaptation.py \
        --game "liars_dice(numdice=1,dice_sides=4)" \
        --encoder trajectory \
        --checkpoint checkpoints/trajectory_encoder_liars_dice_numdice1_dice_sides4_random_improved_500.pt \
        --agent-pool agent_pools/liars_dice_numdice1_dice_sides4_seed42_n500.pt \
        --payoffs payoff_matrices/liars_dice_numdice1_dice_sides4__seed42_payoffs.npz \
        --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pyspiel

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import PPOAgentPolicy, get_device_string
from downstream import set_seed
from generate_agents import load_agents
from trajectory_encoder import TrajectoryEncoderConfig

# Inject for unpickling
sys.modules['__main__'].TrajectoryEncoderConfig = TrajectoryEncoderConfig


def compute_embeddings(encoder_type, agents, game, checkpoint_path, pretrain_agents, device):
    """Compute embeddings for a list of agents."""
    if encoder_type == "trajectory":
        from test_trajectory_encoder import load_trajectory_encoder_from_checkpoint
        _, adapter, _ = load_trajectory_encoder_from_checkpoint(
            checkpoint_path, game, policies=pretrain_agents, device=device
        )
        encoder_fn = adapter.get_encoder(device=device)
        return np.array([encoder_fn(a).detach().cpu().numpy() for a in agents])
    elif encoder_type == "grover":
        from test_grover_encoder import load_grover_from_checkpoint
        _, adapter, _ = load_grover_from_checkpoint(
            checkpoint_path, game, policies=pretrain_agents, device=device
        )
        encoder_fn = adapter.get_encoder(device=device)
        return np.array([encoder_fn(a).detach().cpu().numpy() for a in agents])
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")


def run_opponent_adaptation(
    payoff_matrix, embeddings, n_test=50, n_probe_trajectories_list=None, seed=42,
    split="random"
):
    """
    Evaluate embedding-based opponent-adaptive strategy selection.

    Args:
        payoff_matrix: (N, N) matrix where entry [i,j] = payoff of agent i vs agent j
        embeddings: (N, D) embedding matrix for all N agents
        n_test: Number of held-out test opponents
        n_probe_trajectories_list: Not used here (embeddings are precomputed)
        seed: Random seed for train/test split
        split: "random", "hardest", or "easiest" — how to select test opponents
    """
    set_seed(seed)
    n_agents = len(embeddings)

    if split == "random":
        perm = np.random.permutation(n_agents)
        test_idx = perm[:n_test]
        library_idx = perm[n_test:]
    elif split in ("hardest", "easiest"):
        # Rank agents by how hard they are to beat (mean payoff when they play as P2)
        # Low mean payoff for P1 = hard opponent for P1
        mean_payoff_as_p2 = payoff_matrix.mean(axis=0)  # mean payoff P1 gets vs each P2
        sorted_idx = np.argsort(mean_payoff_as_p2)
        if split == "hardest":
            # Hardest opponents: P1 gets lowest payoff against them
            test_idx = sorted_idx[:n_test]
        else:
            # Easiest opponents: P1 gets highest payoff against them
            test_idx = sorted_idx[-n_test:]
        library_idx = np.array([i for i in range(n_agents) if i not in set(test_idx)])
    else:
        raise ValueError(f"Unknown split: {split}")

    test_embeddings = embeddings[test_idx]
    library_embeddings = embeddings[library_idx]

    # Precompute: for each library agent, what's their payoff against each test opponent?
    # payoff_matrix[i, j] = payoff of agent i (as P1) vs agent j (as P2)
    # We want: given test opponent j, find library agent i that maximizes payoff[i, j]
    # Library agents play as P1, test opponents play as P2
    library_vs_test = payoff_matrix[np.ix_(library_idx, test_idx)]  # (n_library, n_test)

    results = {}

    # --- Oracle: best possible response from library ---
    oracle_payoffs = library_vs_test.max(axis=0)  # Best library agent for each test opponent
    results["oracle"] = {
        "mean": oracle_payoffs.mean(),
        "std": oracle_payoffs.std(),
    }

    # --- Random baseline: pick a random library agent ---
    random_payoffs = []
    for _ in range(100):  # Average over 100 random draws
        random_choices = np.random.randint(0, len(library_idx), size=n_test)
        payoffs = np.array([library_vs_test[random_choices[j], j] for j in range(n_test)])
        random_payoffs.append(payoffs.mean())
    results["random"] = {
        "mean": np.mean(random_payoffs),
        "std": np.std(random_payoffs),
    }

    # --- Best-against-average: agent with highest mean payoff across library ---
    # Use only library-vs-library payoffs to avoid leaking test info
    library_vs_library = payoff_matrix[np.ix_(library_idx, library_idx)]
    best_avg_idx = library_vs_library.mean(axis=1).argmax()  # Best avg performer in library
    best_avg_payoffs = library_vs_test[best_avg_idx, :]
    results["best_avg"] = {
        "mean": best_avg_payoffs.mean(),
        "std": best_avg_payoffs.std(),
    }

    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    # --- Embedding-based nearest neighbor (NN → best counter) ---
    for metric_name, dist_fn in [("cosine", "cosine"), ("l2", "l2")]:
        if dist_fn == "cosine":
            sims = cosine_similarity(test_embeddings, library_embeddings)
            nn_in_library = sims.argmax(axis=1)
        else:
            dists = euclidean_distances(test_embeddings, library_embeddings)
            nn_in_library = dists.argmin(axis=1)

        nn_payoffs = []
        for j in range(n_test):
            proxy_idx = nn_in_library[j]
            best_response_in_library = library_vs_library[:, proxy_idx].argmax()
            nn_payoffs.append(library_vs_test[best_response_in_library, j])

        nn_payoffs = np.array(nn_payoffs)
        results[f"nn_{metric_name}"] = {
            "mean": nn_payoffs.mean(),
            "std": nn_payoffs.std(),
        }

    # --- k-NN weighted (proxy-based) ---
    for k in [3, 5]:
        sims = cosine_similarity(test_embeddings, library_embeddings)
        knn_payoffs = []
        for j in range(n_test):
            top_k = sims[j].argsort()[-k:][::-1]
            weights = sims[j, top_k]
            weights = weights / weights.sum()
            weighted_payoffs = np.zeros(len(library_idx))
            for rank, proxy_idx in enumerate(top_k):
                weighted_payoffs += weights[rank] * library_vs_library[:, proxy_idx]
            best_response = weighted_payoffs.argmax()
            knn_payoffs.append(library_vs_test[best_response, j])

        knn_payoffs = np.array(knn_payoffs)
        results[f"knn_{k}_cosine"] = {
            "mean": knn_payoffs.mean(),
            "std": knn_payoffs.std(),
        }

    # --- Similarity-weighted payoff prediction (kernel regression) ---
    # Instead of finding a proxy and countering it, directly predict payoffs
    # against the test opponent using similarity-weighted known payoffs.
    # predicted_payoff[i, test_j] = sum_k sim(test_j, lib_k) * payoff[i, lib_k] / sum_k sim(test_j, lib_k)
    sims = cosine_similarity(test_embeddings, library_embeddings)  # (n_test, n_library)
    # Clamp negative similarities to 0 for weighting
    sims_pos = np.maximum(sims, 0)

    for temp_name, temperature in [("kernel", 1.0), ("kernel_t5", 5.0), ("kernel_t10", 10.0)]:
        kernel_payoffs = []
        for j in range(n_test):
            # Softmax-style weighting with temperature
            w = sims_pos[j] ** temperature
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
            else:
                w = np.ones(len(library_idx)) / len(library_idx)
            # Predicted payoff of each library agent against test opponent j
            predicted = library_vs_library @ w  # (n_library,)
            best_response = predicted.argmax()
            kernel_payoffs.append(library_vs_test[best_response, j])

        kernel_payoffs = np.array(kernel_payoffs)
        results[temp_name] = {
            "mean": kernel_payoffs.mean(),
            "std": kernel_payoffs.std(),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding-based opponent adaptation")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True, choices=["trajectory", "grover"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--agent-pool", type=str, required=True)
    parser.add_argument("--payoffs", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--split", type=str, default="random",
                        choices=["random", "hardest", "easiest"],
                        help="How to select test opponents")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or get_device_string()
    print(f"Game: {args.game}")
    print(f"Encoder: {args.encoder}")
    print(f"Split: {args.split}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    game = pyspiel.load_game(args.game)

    # Load agents
    print(f"\nLoading agents from {args.agent_pool}...")
    pretrain_agents, eval_agents = load_agents(args.agent_pool, game)

    # Load precomputed payoffs
    print(f"Loading payoffs from {args.payoffs}...")
    payoff_data = np.load(args.payoffs, allow_pickle=True)
    task_b_payoffs = payoff_data["task_b_payoffs"]
    task_b_pairs = payoff_data["task_b_pair_indices"]
    n_agents = len(eval_agents)

    # Reconstruct full payoff matrix from flat array + pair indices
    payoff_matrix = np.zeros((n_agents, n_agents))
    for idx, (p1, p2) in enumerate(task_b_pairs):
        payoff_matrix[p1, p2] = task_b_payoffs[idx]

    print(f"Payoff matrix: {payoff_matrix.shape}")
    print(f"  Mean: {payoff_matrix.mean():.4f}, Std: {payoff_matrix.std():.4f}")

    # Compute embeddings
    print(f"\nComputing {args.encoder} embeddings...")
    embeddings = compute_embeddings(
        args.encoder, eval_agents, game, args.checkpoint, pretrain_agents, device
    )
    print(f"Embedding shape: {embeddings.shape}")

    # Run opponent adaptation experiment
    print(f"\n{'='*70}")
    print(f"Opponent-Adaptive Strategy Selection ({args.encoder})")
    print(f"{'='*70}")

    results = run_opponent_adaptation(
        payoff_matrix, embeddings, n_test=args.n_test, seed=args.seed,
        split=args.split
    )

    print(f"\n{'Method':<25} {'Mean Payoff':>15} {'Std':>10}")
    print("-" * 55)
    for method, vals in sorted(results.items()):
        print(f"{method:<25} {vals['mean']:>15.4f} {vals['std']:>10.4f}")

    # --- Paired statistical test (kernel vs best_avg) ---
    # Per-opponent payoff difference gives 50 paired observations
    from scipy import stats as sp_stats
    sims = cosine_similarity(test_embeddings, library_embeddings)
    sims_pos = np.maximum(sims, 0)
    best_temp_name, best_temp_payoffs = None, None
    for temp_name, temperature in [("kernel", 1.0), ("kernel_t5", 5.0), ("kernel_t10", 10.0)]:
        if results.get(temp_name, {}).get("mean", -999) == max(
            results.get("kernel", {}).get("mean", -999),
            results.get("kernel_t5", {}).get("mean", -999),
            results.get("kernel_t10", {}).get("mean", -999),
        ):
            best_temp_name = temp_name

    # Recompute per-opponent payoffs for best kernel and best_avg
    best_temperature = {"kernel": 1.0, "kernel_t5": 5.0, "kernel_t10": 10.0}[best_temp_name]
    kernel_per_opponent = []
    best_avg_per_opponent = library_vs_test[best_avg_idx, :]
    for j in range(n_test):
        w = sims_pos[j] ** best_temperature
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(len(library_idx)) / len(library_idx)
        predicted = library_vs_library @ w
        best_response = predicted.argmax()
        kernel_per_opponent.append(library_vs_test[best_response, j])
    kernel_per_opponent = np.array(kernel_per_opponent)

    diff = kernel_per_opponent - best_avg_per_opponent
    t_stat, p_value = sp_stats.ttest_rel(kernel_per_opponent, best_avg_per_opponent)
    wilcoxon_stat, wilcoxon_p = sp_stats.wilcoxon(diff[diff != 0]) if np.any(diff != 0) else (0, 1.0)
    results["paired_test"] = {
        "mean_diff": diff.mean(),
        "n_kernel_wins": int((diff > 0).sum()),
        "n_best_avg_wins": int((diff < 0).sum()),
        "n_ties": int((diff == 0).sum()),
        "ttest_p": p_value,
        "wilcoxon_p": wilcoxon_p,
    }

    print(f"\n{'='*70}")
    print("Summary:")
    oracle = results["oracle"]["mean"]
    random_val = results["random"]["mean"]
    best_nn = max(
        results.get("nn_cosine", {}).get("mean", -999),
        results.get("nn_l2", {}).get("mean", -999),
        results.get("knn_3_cosine", {}).get("mean", -999),
        results.get("knn_5_cosine", {}).get("mean", -999),
        results.get("kernel", {}).get("mean", -999),
        results.get("kernel_t5", {}).get("mean", -999),
        results.get("kernel_t10", {}).get("mean", -999),
    )
    print(f"  Oracle:     {oracle:.4f}")
    print(f"  Best embed: {best_nn:.4f}")
    print(f"  Random:     {random_val:.4f}")
    print(f"  Best-Avg:   {results['best_avg']['mean']:.4f}")
    if oracle != random_val:
        pct = (best_nn - random_val) / (oracle - random_val) * 100
        print(f"  Embedding captures {pct:.1f}% of oracle-random gap")
    pt = results["paired_test"]
    print(f"\n  Paired test (kernel vs best_avg, n={n_test} opponents):")
    print(f"    Kernel wins: {pt['n_kernel_wins']}, Best-Avg wins: {pt['n_best_avg_wins']}, Ties: {pt['n_ties']}")
    print(f"    Mean diff: {pt['mean_diff']:.4f}")
    print(f"    Paired t-test p={pt['ttest_p']:.4f}")
    print(f"    Wilcoxon p={pt['wilcoxon_p']:.4f}")
    print(f"{'='*70}")
