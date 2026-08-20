import numpy as np
import pyspiel
from utils import make_diverse_random_kuhn_poker_layer_init
from iig_rl_benchmark.algorithms.ppo import ppo
from weight_autoencoder import (
    WeightAutoencoder, AutoencoderConfig, ppo_agent_to_vector, vector_to_ppo_agent,
)


def _kuhn_agent(game):
    li = make_diverse_random_kuhn_poker_layer_init(game)
    return ppo.PPOAgent(game.num_distinct_actions(),
                        game.information_state_tensor_shape(), "cpu", li, 256)


def test_vector_to_ppo_agent_roundtrip():
    game = pyspiel.load_game("kuhn_poker")
    agent = _kuhn_agent(game)
    vec = ppo_agent_to_vector(agent)
    rebuilt = vector_to_ppo_agent(agent, vec)
    assert np.allclose(ppo_agent_to_vector(rebuilt).detach().numpy(),
                       vec.detach().numpy(), atol=1e-6)


def test_weight_autoencoder_get_decoder_returns_policy():
    game = pyspiel.load_game("kuhn_poker")
    agents = [_kuhn_agent(game) for _ in range(40)]
    ae = WeightAutoencoder(
        AutoencoderConfig(hidden_dims=(64,), bottleneck_dim=16, epochs=2,
                          batch_size=8, device="cpu"),
        agents, ppo_agent_to_vector)
    ae.train()
    encode = ae.get_encoder(device="cpu")
    decode = ae.get_decoder(game, player_id=0, template_agent=agents[0], device="cpu")
    emb = encode(agents[0])
    policy = decode(emb)
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    probs = policy.action_probabilities(state)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert set(probs.keys()) == set(state.legal_actions())
