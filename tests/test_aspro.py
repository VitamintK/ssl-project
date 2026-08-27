import numpy as np
import pytest
import pyspiel
from open_spiel.python.policy import UniformRandomPolicy
from psro import best_response_value, run_neupl_v2


def test_brv_of_uniform_p0_on_kuhn():
    game = pyspiel.load_game("kuhn_poker")
    u = UniformRandomPolicy(game)
    # Best response value for player 1 against a uniform player 0 (known ~0.4167).
    brv = best_response_value(game, u, fixed_player=0, br_player=1)
    assert game.min_utility() <= brv <= game.max_utility()
    assert abs(brv - 0.41666667) < 1e-6


def test_aspro_rejects_randall_loss():
    with pytest.raises(ValueError):
        run_neupl_v2(game_name="kuhn_poker", aspro=True, use_randall_loss=True)


def test_aspro_rejects_gt_payoffs():
    with pytest.raises(ValueError):
        run_neupl_v2(game_name="kuhn_poker", aspro=True, gt_payoffs=True)
