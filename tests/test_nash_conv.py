import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability
from downstream import compute_nash_conv


def test_matches_openspiel_on_uniform():
    game = pyspiel.load_game("kuhn_poker")
    uni = policy_lib.UniformRandomPolicy(game)
    got = compute_nash_conv(game, uni, uni)
    want = exploitability.nash_conv(game, policy_lib.UniformRandomPolicy(game))
    assert abs(got - want) < 1e-9
    assert got > 0.0  # uniform is exploitable
