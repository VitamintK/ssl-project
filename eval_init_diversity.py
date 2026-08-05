"""
Evaluate diversity of random policy populations under different weight initializations,
and optionally compare against trained PSRO / NEUPL policy populations.

For each source, creates or loads N policies, measures exact expected payoffs against
a uniform random opponent, and reports mean/std of payoff and P(check | jack).

Initializations compared (always run):
  pytorch_default  - kaiming_uniform_(weight, a=sqrt(5)) + uniform bias (PyTorch default)
  our_method       - orthogonal_(weight, 2.2) + uniform(-1,1) bias on output, 0 elsewhere
  zero_bias        - same as our_method but constant 0 bias everywhere
  scaled_ortho     - orthogonal with sqrt(2) hidden, 0.01 last layer; same bias as our_method

Optional sources (via flags):
  --psro           - load trained PSRO policies (hs256)
  --neupl          - generate policies from trained NEUPL checkpoint (hs256)
"""

import math
import argparse
from dataclasses import dataclass
import random
from typing import Callable

import numpy as np
import torch
import pyspiel
from tqdm import tqdm

from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms.expected_game_score import policy_value

from downstream import set_seed
from psro import load_ppo_agents_from_psro, make_neupl_policies
from utils import PPOAgentPolicy, make_diverse_random_kuhn_poker_layer_init


# ---------------------------------------------------------------------------
# Layer-init factory functions
# ---------------------------------------------------------------------------

def make_pytorch_default_layer_init() -> Callable:
    """PyTorch's default nn.Linear init: kaiming_uniform weight, uniform bias."""
    def layer_init(layer, std=None, bias_const=None):
        torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(layer.bias, -bound, bound)
        return layer
    return layer_init


def make_zero_bias_layer_init(game) -> Callable:
    """Our method but with constant 0 bias everywhere."""
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, 2.2)
        torch.nn.init.constant_(layer.bias, 0.0)
        return layer
    return layer_init


def make_scaled_ortho_layer_init(game) -> Callable:
    """Orthogonal with sqrt(2) for hidden, 0.01 for last actor layer.
    Bias: uniform(-1,1) on output layer, 0 elsewhere (same as our_method)."""
    num_actions = game.num_distinct_actions()

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        if layer.out_features == num_actions:
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            torch.nn.init.uniform_(layer.bias, -1, 1)
        else:
            torch.nn.init.orthogonal_(layer.weight, math.sqrt(2))
            torch.nn.init.constant_(layer.bias, 0.0)
        return layer
    return layer_init


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def get_check_prob_with_jack(game, p0_policy) -> float:
    """Return p0's probability of checking when holding a jack (first decision)."""
    state = game.new_initial_state()
    state.apply_action(0)   # deal jack (card 0) to p0
    state.apply_action(1)   # deal queen (card 1) to p1; doesn't affect p0's info state
    # legal actions: 0 = check, 1 = bet
    return p0_policy.action_probabilities(state).get(0, 0.0)


def evaluate_policies(name: str, policies: list, game) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact payoff vs uniform random and P(check|jack) for each policy."""
    opponent = policy_lib.UniformRandomPolicy(game)
    payoffs, check_probs = [], []
    for p0_policy in tqdm(policies, desc=name):
        payoffs.append(policy_value(game.new_initial_state(), [p0_policy, opponent])[0])
        check_probs.append(get_check_prob_with_jack(game, p0_policy))
    return np.array(payoffs), np.array(check_probs)


# ---------------------------------------------------------------------------
# Policy source builders
# ---------------------------------------------------------------------------

@dataclass
class PolicySource:
    key: str
    display_name: str
    policies: list  # list of Policy objects (p0 only)


def build_random_init_sources(game, n_agents: int) -> list[PolicySource]:
    inits = [
        ("pytorch_default", "PyTorch Default", make_pytorch_default_layer_init()),
        ("our_method",      "Our Method",      make_diverse_random_kuhn_poker_layer_init(game)),
        ("zero_bias",       "Zero Bias",        make_zero_bias_layer_init(game)),
        ("scaled_ortho",    "Scaled Ortho",     make_scaled_ortho_layer_init(game)),
    ]
    sources = []
    for key, display_name, layer_init_fn in inits:
        policies = [
            PPOAgentPolicy(
                game,
                PPOAgent(game.num_distinct_actions(), game.information_state_tensor_shape(),
                         'cpu', layer_init_fn, 256),
                player_id=0,
                use_observation=False,
            )
            for _ in range(n_agents)
        ]
        sources.append(PolicySource(key=key, display_name=display_name, policies=policies))
    return sources


def build_psro_source(game, n_agents: int) -> PolicySource:
    agents = load_ppo_agents_from_psro(
        game_short_name=game.get_type().short_name,
        player_id=0,
        hidden_size=256,
    )
    policies = [PPOAgentPolicy(game, agent, player_id=0, use_observation=False) for agent in agents]
    policies = random.sample(policies, 500)
    return PolicySource(key="psro", display_name="PSRO", policies=policies)


def build_neupl_source(game, n_agents: int) -> PolicySource:
    policies_and_embeddings = make_neupl_policies(
        game_short_name=game.get_type().short_name,
        neupl_config=dict(hidden_size=256, policy_embedding_size=64),
        num_policies_to_make=n_agents,
    )
    # policies_and_embeddings[player_id] is a list of (embedding, policy) tuples
    policies = [p for _, p in policies_and_embeddings[0]]
    return PolicySource(key="neupl", display_name="NEUPL", policies=policies)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_text_table(rows: list[tuple]):
    """rows: list of (display_name, payoffs, check_probs)"""
    header = f"{'Init method':<20}  {'Mean payoff':>12}  {'Std payoff':>12}  {'Mean P(check|J)':>16}  {'Std P(check|J)':>15}"
    print(header)
    print("-" * len(header))
    for name, payoffs, check_probs in rows:
        print(f"{name:<20}  {payoffs.mean():>12.3f}  {payoffs.std():>12.3f}  "
              f"{check_probs.mean():>16.3f}  {check_probs.std():>15.3f}")


def print_latex_table(rows: list[tuple]):
    """rows: list of (display_name, payoffs, check_probs)"""
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Init & Payoff & $P(\text{check} \mid J)$ \\")
    print(r"\midrule")
    for name, payoffs, check_probs in rows:
        print(f"{name} & ${payoffs.mean():.3f} \\pm {payoffs.std():.3f}$ & "
              f"${check_probs.mean():.3f} \\pm {check_probs.std():.3f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="kuhn_poker",
                        help="OpenSpiel game name (default: kuhn_poker)")
    parser.add_argument("--source", choices=["random", "psro", "neupl"], default="random",
                        help="Policy source to evaluate (default: random)")
    parser.add_argument("--n-agents", type=int, default=1000,
                        help="Number of agents (random inits or NEUPL samples)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compact", action="store_true",
                        help="Limit output to pytorch_default and our_method only (random source only)")
    args = parser.parse_args()

    set_seed(args.seed)
    game = pyspiel.load_game(args.game)

    # Build policy sources
    if args.source == "random":
        sources = build_random_init_sources(game, args.n_agents)
        if args.compact:
            compact_keys = {"pytorch_default", "our_method"}
            sources = [s for s in sources if s.key in compact_keys]
    elif args.source == "psro":
        sources = [build_psro_source(game, args.n_agents)]
    elif args.source == "neupl":
        sources = [build_neupl_source(game, args.n_agents)]

    # Evaluate
    print(f"\nEvaluating (exact expected payoffs, {args.n_agents} random agents per init)\n")
    rows = []
    for source in sources:
        payoffs, check_probs = evaluate_policies(source.display_name, source.policies, game)
        rows.append((source.display_name, payoffs, check_probs))

    print()
    print_text_table(rows)
    print()
    print_latex_table(rows)
