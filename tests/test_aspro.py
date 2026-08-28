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
        run_neupl_v2(game_name="kuhn_poker", apsro=True, use_randall_loss=True)


def test_aspro_rejects_gt_payoffs():
    with pytest.raises(ValueError):
        run_neupl_v2(game_name="kuhn_poker", apsro=True, gt_payoffs=True)


import json, os, glob


def _latest_experiment_dir():
    dirs = glob.glob(os.path.join("results", "*", "neupl", "*", "*", "kuhn_poker", "*"))
    return max(dirs, key=os.path.getmtime)


@pytest.mark.parametrize("exploited,bandit", [("p1", "hedge"), ("p2", "rm")])
def test_aspro_smoke_runs_and_logs_brv(exploited, bandit):
    run_neupl_v2(
        game_name="kuhn_poker", apsro=True, apsro_exploited_player=exploited, apsro_bandit=bandit,
        num_iterations=2, num_pols_sampled=2, total_episodes_per_policy=5, T=2,
        expl_check_episode_interval=1,  # force a BRV measurement every iteration
        save_checkpoints=False,  # don't gunk up select_neupl_directory with test runs
    )
    exp = _latest_experiment_dir()
    with open(os.path.join(exp, "stats.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records, "no stats recorded"
    assert "brv_per_policy" in records[-1]
    assert len(records[-1]["brv_per_policy"]) >= 1
    assert os.path.exists(os.path.join(exp, "brv.png"))
    # save_checkpoints=False: no checkpoints/config.json written (keeps the picker clean).
    assert not os.path.exists(os.path.join(exp, "policy0_ckpt.pt"))
    assert not os.path.exists(os.path.join(exp, "config.json"))


def test_aspro_debug_runs_without_error():
    # debug=True exercises _debug_apsro (exact payoffs vs each exploiter + bandit % + EV).
    run_neupl_v2(
        game_name="kuhn_poker", apsro=True, apsro_exploited_player="p1", apsro_bandit="hedge",
        num_iterations=1, num_pols_sampled=2, total_episodes_per_policy=3, T=1,
        expl_check_episode_interval=1, debug=True, save_checkpoints=False,
    )
    exp = _latest_experiment_dir()
    assert os.path.exists(os.path.join(exp, "stats.jsonl"))
    assert os.path.exists(os.path.join(exp, "brv.png"))


def test_aspro_checkpoint_is_loadable():
    # This test must save a checkpoint to exercise loading; clean it up afterward so it
    # doesn't linger in select_neupl_directory.
    import shutil
    from psro import load_ppo_agents_from_neupl
    run_neupl_v2(
        game_name="kuhn_poker", apsro=True, apsro_exploited_player="p1", apsro_bandit="hedge",
        num_iterations=1, num_pols_sampled=2, total_episodes_per_policy=3, T=1,
        expl_check_episode_interval=1,  # save_checkpoints defaults to True
    )
    exp = _latest_experiment_dir()
    try:
        # config.json must record num_policies so the loader sizes the embedding table right.
        with open(os.path.join(exp, "config.json")) as f:
            cfg = json.load(f)
        assert cfg["num_policies"] >= 2
        # select_neupl_directory filters on this key; must be present so aspro dirs surface.
        assert cfg["use_randall_loss"] is False
        assert cfg["apsro"] is True
        # Round-trip through the standard NeuPL loader (hidden_size/embedding must match training).
        dir_name = os.path.basename(os.path.normpath(exp))
        a0, a1 = load_ppo_agents_from_neupl(
            "kuhn_poker", hidden_size=256, policy_embedding_size=64, dir_name=dir_name)
        assert a0.num_policies == cfg["num_policies"]
        assert a1 is not None
    finally:
        shutil.rmtree(exp, ignore_errors=True)
