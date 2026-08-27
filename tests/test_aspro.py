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


import json, os, glob


def _latest_experiment_dir():
    dirs = glob.glob(os.path.join("results", "*", "neupl", "*", "*", "kuhn_poker", "*"))
    return max(dirs, key=os.path.getmtime)


@pytest.mark.parametrize("exploited,bandit", [("p1", "hedge"), ("p2", "rm")])
def test_aspro_smoke_runs_and_logs_brv(exploited, bandit):
    run_neupl_v2(
        game_name="kuhn_poker", aspro=True, exploited_player=exploited, bandit=bandit,
        num_iterations=2, num_pols_sampled=2, total_episodes_per_policy=5, T=2,
        expl_check_episode_interval=1,  # force a BRV measurement every iteration
    )
    exp = _latest_experiment_dir()
    with open(os.path.join(exp, "stats.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records, "no stats recorded"
    assert "brv_per_policy" in records[-1]
    assert len(records[-1]["brv_per_policy"]) >= 1
    assert os.path.exists(os.path.join(exp, "brv.png"))
    assert os.path.exists(os.path.join(exp, "policy0_ckpt.pt"))
