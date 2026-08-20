# tests/test_task_f_wiring.py
from config import ExperimentInfo
from main import _run_experiment


def test_run_experiment_task_f_random_identity():
    spec = {
        "source": "random", "game_name": "kuhn_poker", "player_id": 0,
        "embedding_type": "identity", "N_random": 6, "task": "f",
        "predictor_type": "mlp", "device": "cpu", "policy_device": "cpu",
        "task_f_overrides": {"outer_steps": 4, "inner_steps": 2, "num_restarts": 1,
                             "nashconv_baseline_samples": 1, "model_num_epochs": 30},
        "experiment_info": ExperimentInfo("kuhn ppo random identity Task F",
                                          embedding_type="identity", task_id="F"),
    }
    exp_info, result = _run_experiment(spec)
    assert result["nashconv"] >= 0.0
