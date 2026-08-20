import numpy as np
import pyspiel
from config import ModelConfig
from downstream import PayoffPredictor
from utils import make_diverse_random_kuhn_poker_layer_init, PPOAgentPolicy
from iig_rl_benchmark.algorithms.ppo import ppo
from weight_autoencoder import ppo_agent_to_vector


def _agents(game, n):
    li = make_diverse_random_kuhn_poker_layer_init(game)
    return [ppo.PPOAgent(game.num_distinct_actions(),
                         game.information_state_tensor_shape(), "cpu", li, 256)
            for _ in range(n)]


def _predictor(game, n_p1, n_p2, num_pairs):
    p1_agents, p2_agents = _agents(game, n_p1), _agents(game, n_p2)
    return PayoffPredictor(
        game=game,
        p1_policies=[PPOAgentPolicy(game, a, 0, False) for a in p1_agents],
        p2_policies=[PPOAgentPolicy(game, a, 1, False) for a in p2_agents],
        p1_embeddings=[ppo_agent_to_vector(a).detach().numpy() for a in p1_agents],
        p2_embeddings=[ppo_agent_to_vector(a).detach().numpy() for a in p2_agents],
        model_config=ModelConfig(model_type="linear"),
        device="cpu",
        num_pairs=num_pairs,
    )


def test_num_pairs_none_uses_full_grid():
    game = pyspiel.load_game("kuhn_poker")
    pred = _predictor(game, 4, 3, num_pairs=None)
    pred.compute_ground_truth_payoffs()
    assert len(pred.pair_indices) == 4 * 3
    assert len(pred.ground_truth_payoffs) == 4 * 3


def test_num_pairs_samples_distinct_subset():
    game = pyspiel.load_game("kuhn_poker")
    K = 5
    pred = _predictor(game, 4, 4, num_pairs=K)
    pred.compute_ground_truth_payoffs()
    assert len(pred.pair_indices) == K
    assert len(pred.ground_truth_payoffs) == K
    # distinct pairs, all within range
    assert len(set(pred.pair_indices)) == K
    for p1_idx, p2_idx in pred.pair_indices:
        assert 0 <= p1_idx < 4 and 0 <= p2_idx < 4


def test_num_pairs_at_least_grid_uses_full_grid():
    game = pyspiel.load_game("kuhn_poker")
    pred = _predictor(game, 3, 3, num_pairs=1000)  # >= 9 -> full grid
    pred.compute_ground_truth_payoffs()
    assert len(pred.pair_indices) == 9
