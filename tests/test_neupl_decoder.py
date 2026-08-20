import torch
import pyspiel
from psro import neupl_decoder
from utils import PPONeuplAgentPolicy


class _StubNeuplAgent(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self._p = torch.nn.Parameter(torch.zeros(1))

    def get_action(self, info_state, embedding=None, legal_actions_mask=None):
        n = self.num_actions
        probs = torch.ones(1, n) / n
        if legal_actions_mask is not None:
            probs = probs * legal_actions_mask
            probs = probs / probs.sum()
        return None, None, None, probs


def test_neupl_decoder_returns_valid_policy():
    game = pyspiel.load_game("kuhn_poker")
    agent = _StubNeuplAgent(game.num_distinct_actions())
    emb = torch.zeros(8)  # (D,)
    policy = neupl_decoder(game, agent, emb, player_id=0)
    assert isinstance(policy, PPONeuplAgentPolicy)
    assert policy._embedding.ndim == 2 and policy._embedding.shape[0] == 1
    state = game.new_initial_state()
    # Advance past chance nodes to reach a decision node
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    probs = policy.action_probabilities(state)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert set(probs.keys()) == set(state.legal_actions())
