import numpy as np
import pyspiel
from config import TaskFConfig, ModelConfig
from utils import make_diverse_random_kuhn_poker_layer_init, PPOAgentPolicy
from iig_rl_benchmark.algorithms.ppo import ppo
from weight_autoencoder import ppo_agent_to_vector, vector_to_ppo_agent
from config import ExperimentInfo
from tasks import run_task_f


def _agents(game, n):
    li = make_diverse_random_kuhn_poker_layer_init(game)
    return [ppo.PPOAgent(game.num_distinct_actions(),
                         game.information_state_tensor_shape(), "cpu", li, 256)
            for _ in range(n)]


def test_run_task_f_smoke_identity_source():
    game = pyspiel.load_game("kuhn_poker")
    p1_agents, p2_agents = _agents(game, 6), _agents(game, 6)
    p1_pol = [PPOAgentPolicy(game, a, 0, False) for a in p1_agents]
    p2_pol = [PPOAgentPolicy(game, a, 1, False) for a in p2_agents]
    p1_emb = [ppo_agent_to_vector(a).detach().numpy() for a in p1_agents]
    p2_emb = [ppo_agent_to_vector(a).detach().numpy() for a in p2_agents]
    # identity decoders: embedding is the actor weight-vector
    p1_dec = lambda e, t=p1_agents[0]: PPOAgentPolicy(game, vector_to_ppo_agent(t, e), 0, False)
    p2_dec = lambda e, t=p2_agents[0]: PPOAgentPolicy(game, vector_to_ppo_agent(t, e), 1, False)
    cfg = TaskFConfig(
        model_config=ModelConfig(model_type="mlp", num_epochs=30, early_stopping_patience=10),
        outer_steps=5, inner_steps=2, num_restarts=1, nashconv_baseline_samples=2)
    info = ExperimentInfo("kuhn identity Task F", embedding_type="identity", task_id="F")
    out = run_task_f(game, p1_pol, p1_emb, p2_pol, p2_emb, p1_dec, p2_dec, cfg, info, "cpu")
    assert "nashconv" in out and out["nashconv"] >= 0.0
    assert "nashconv_baseline" in out
