"""
Precompute and save ground-truth payoff matrices for Tasks A and B.

Since the payoff matrix depends only on the agents (not the encoder),
this can be computed once and reused across all encoder evaluations.

Usage:
    python precompute_payoffs.py --game kuhn_poker --agent-pool agent_pools/kuhn_poker_seed42_n500.pt --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pyspiel
from open_spiel.python import policy as policy_lib
from tqdm import tqdm

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from policy_repr.utils import PPOAgentPolicy, get_expected_payoffs
from policy_repr.downstream.heads import set_seed
from policy_repr.datasets.generate_agents import load_agents


def precompute_task_a(game, eval_agents):
    """Compute payoff of each agent vs uniform random opponent."""
    opponent_policy = policy_lib.UniformRandomPolicy(game)
    payoffs = []
    for agent in tqdm(eval_agents, desc="Task A payoffs"):
        p1_policy = PPOAgentPolicy(game, agent, 0, False)
        payoff = get_expected_payoffs(game, p1_policy, opponent_policy)
        payoffs.append(payoff)
    return np.array(payoffs)


def precompute_task_b(game, eval_agents):
    """Compute pairwise payoff matrix for all agent matchups."""
    payoffs = []
    pair_indices = []
    for p1_idx, p1_agent in enumerate(tqdm(eval_agents, desc="Task B P1")):
        p1_policy = PPOAgentPolicy(game, p1_agent, 0, False)
        for p2_idx, p2_agent in enumerate(eval_agents):
            p2_policy = PPOAgentPolicy(game, p2_agent, 1, False)
            payoff = get_expected_payoffs(game, p1_policy, p2_policy)
            payoffs.append(payoff)
            pair_indices.append((p1_idx, p2_idx))
    return np.array(payoffs), pair_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute payoff matrices for Tasks A and B")
    parser.add_argument("--game", type=str, required=True)
    parser.add_argument("--agent-pool", type=str, required=True,
                        help="Path to saved agent pool (from generate_agents.py)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="payoff_matrices")
    args = parser.parse_args()

    set_seed(args.seed)
    game = pyspiel.load_game(args.game)

    print(f"Loading agents from {args.agent_pool}...")
    pretrain_agents, eval_agents = load_agents(args.agent_pool, game)
    print(f"Loaded {len(eval_agents)} eval agents")

    # Task A
    print("\n=== Computing Task A payoffs ===")
    task_a_payoffs = precompute_task_a(game, eval_agents)
    print(f"Task A: {len(task_a_payoffs)} payoffs computed")
    print(f"  Mean: {task_a_payoffs.mean():.6f}, Std: {task_a_payoffs.std():.6f}")

    # Task B
    print("\n=== Computing Task B payoffs ===")
    task_b_payoffs, task_b_pairs = precompute_task_b(game, eval_agents)
    print(f"Task B: {len(task_b_payoffs)} payoffs computed")
    print(f"  Mean: {task_b_payoffs.mean():.6f}, Std: {task_b_payoffs.std():.6f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a safe filename for games with parentheses
    game_safe = args.game.replace("(", "_").replace(")", "_").replace(",", "_").replace("=", "")
    save_path = out_dir / f"{game_safe}_seed{args.seed}_payoffs.npz"
    np.savez(
        save_path,
        task_a_payoffs=task_a_payoffs,
        task_b_payoffs=task_b_payoffs,
        task_b_pair_indices=np.array(task_b_pairs),
        game=args.game,
        seed=args.seed,
        num_agents=len(eval_agents),
    )
    print(f"\nSaved to {save_path}")
