"""Batched tabularization must reproduce the live exact payoff (Task F value-fn targets)."""
import numpy as np
import pytest
import torch
import pyspiel
from iig_rl_benchmark.algorithms.ppo.ppo import PPOConditionedOnPolicyRepresentationAgent
from utils import (PPONeuplAgentPolicy, tabularize_policy,
                   _get_expected_payoffs_exact, _batched_action_probs)
from open_spiel.python.algorithms.expected_game_score import policy_value


def _make_neupl_policy(game, agent, player_id, embedding):
    return PPONeuplAgentPolicy(game, agent, player_id, use_observation=False, embedding=embedding)


def _agent_and_policies(game, seed=0):
    torch.manual_seed(seed)
    obs = game.information_state_tensor_shape()
    agent = PPOConditionedOnPolicyRepresentationAgent(
        num_actions=game.num_distinct_actions(), observation_shape=obs, device="cpu",
        num_policies=4, policy_embedding_size=8, hidden_size=32)
    agent.eval()
    emb_dim = 8
    e0 = torch.randn(1, emb_dim)
    e1 = torch.randn(1, emb_dim)
    p0 = _make_neupl_policy(game, agent, 0, e0)
    p1 = _make_neupl_policy(game, agent, 1, e1)
    return agent, p0, p1


def test_batched_tabular_matches_live_exact_kuhn():
    game = pyspiel.load_game("kuhn_poker")
    _, p0, p1 = _agent_and_policies(game, seed=1)
    live = _get_expected_payoffs_exact(game, p0, p1)
    t0, t1 = tabularize_policy(game, p0, 0), tabularize_policy(game, p1, 1)
    batched = policy_value(game.new_initial_state(), [t0, t1])[0]
    assert abs(live - batched) < 1e-6, f"live={live} batched={batched}"


def test_batched_tabular_matches_live_exact_leduc():
    game = pyspiel.load_game("leduc_poker")
    _, p0, p1 = _agent_and_policies(game, seed=2)
    live = _get_expected_payoffs_exact(game, p0, p1)
    t0, t1 = tabularize_policy(game, p0, 0), tabularize_policy(game, p1, 1)
    batched = policy_value(game.new_initial_state(), [t0, t1])[0]
    assert abs(live - batched) < 1e-6, f"live={live} batched={batched}"


def test_batched_probs_match_per_state():
    # The batched forward must equal the per-state action_probabilities row for row.
    game = pyspiel.load_game("kuhn_poker")
    _, p0, _ = _agent_and_policies(game, seed=3)
    tab = _policy_states(game, player_id=0)
    probs = _batched_action_probs(p0, tab, 0, game)
    for i, s in enumerate(tab):
        per_state = p0.action_probabilities(s, 0)
        for a, p in per_state.items():
            assert abs(probs[i][a] - p) < 1e-6


def _policy_states(game, player_id):
    from open_spiel.python import policy as policy_lib
    return [s for s in policy_lib.TabularPolicy(game).states if s.current_player() == player_id]


# --- reach-decomposition grid payoffs must match per-pair policy_value ---
from utils import grid_payoffs_via_reach, _TERMINAL_CACHE


def _grid_reference(game, P0, P1):
    """Reference: the exact per-pair value via policy_value over tabular policies."""
    t0 = [tabularize_policy(game, a, 0) for a in P0]
    t1 = [tabularize_policy(game, b, 1) for b in P1]
    return np.array([[policy_value(game.new_initial_state(), [a, b])[0] for b in t1] for a in t0])


def _policies_grid(game, n=3, seed=0):
    torch.manual_seed(seed)
    agent = PPOConditionedOnPolicyRepresentationAgent(
        num_actions=game.num_distinct_actions(), observation_shape=game.information_state_tensor_shape(),
        device="cpu", num_policies=8, policy_embedding_size=8, hidden_size=32)
    agent.eval()
    P0 = [PPONeuplAgentPolicy(game, agent, 0, False, embedding=torch.randn(1, 8)) for _ in range(n)]
    P1 = [PPONeuplAgentPolicy(game, agent, 1, False, embedding=torch.randn(1, 8)) for _ in range(n)]
    return P0, P1


@pytest.mark.parametrize("gname,tol", [("kuhn_poker", 1e-9), ("leduc_poker", 1e-7)])
def test_reach_grid_matches_policy_value(gname, tol):
    _TERMINAL_CACHE.clear()
    game = pyspiel.load_game(gname)
    P0, P1 = _policies_grid(game, n=3, seed=5)
    ref = _grid_reference(game, P0, P1)
    tab_p1 = {i: tabularize_policy(game, P0[i], 0) for i in range(len(P0))}
    tab_p2 = {j: tabularize_policy(game, P1[j], 1) for j in range(len(P1))}
    pairs = [(i, j) for i in range(len(P0)) for j in range(len(P1))]
    got = np.array(grid_payoffs_via_reach(game, tab_p1, tab_p2, pairs)).reshape(len(P0), len(P1))
    assert np.abs(ref - got).max() < tol, f"max diff {np.abs(ref - got).max():.2e}"


def test_reach_grid_respects_pair_subset_and_order():
    _TERMINAL_CACHE.clear()
    game = pyspiel.load_game("kuhn_poker")
    P0, P1 = _policies_grid(game, n=4, seed=6)
    tab_p1 = {i: tabularize_policy(game, P0[i], 0) for i in range(4)}
    tab_p2 = {j: tabularize_policy(game, P1[j], 1) for j in range(4)}
    pairs = [(2, 1), (0, 3), (3, 3)]  # arbitrary subset + order
    got = grid_payoffs_via_reach(game, tab_p1, tab_p2, pairs)
    for k, (i, j) in enumerate(pairs):
        ref = policy_value(game.new_initial_state(), [tab_p1[i], tab_p2[j]])[0]
        assert abs(got[k] - ref) < 1e-9
