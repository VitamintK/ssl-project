from iig_rl_benchmark.algorithms.ppo import ppo
import numpy as np
import torch
import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode

class PPOAgentPolicy(policy.Policy):
    def __init__(self, game, ppo_agent, player_id: int,
                    use_observation: bool):
        """
        Args:
            game: The game.
            agent: RL agent.
            player_id: ID of the player.
            use_observation: use observation if True, otherwise use infostate.
        """
        self._game = game
        self._player_id = player_id
        self._ppo_agent = ppo_agent
        self._use_observation = use_observation

    def action_probabilities(self, state: pyspiel.State, player_id=None):
        assert not state.is_simultaneous_node()
        assert not self._use_observation
        if player_id is None:
            player_id = state.current_player()
        else:
            assert player_id == state.current_player()
        player_id = int(player_id)
        legal_actions = state.legal_actions(player_id)
        legal_action_mask = torch.zeros(self._game.num_distinct_actions())
        legal_action_mask[legal_actions] = 1
        info_state = torch.Tensor(state.information_state_tensor(player_id))
        action, log_probs, entropy, value, probs = self._ppo_agent.get_action_and_value(info_state, legal_action_mask)
        probs = probs.detach().numpy()
        prob_dict = {a: probs[a] for a in legal_actions}
        return prob_dict

def get_expected_payoffs(game: pyspiel.Game, p0_ppo_agent: ppo.PPOAgent, p1_policy: policy.Policy) -> float:
    policies = [PPOAgentPolicy(game, p0_ppo_agent, 0, False), p1_policy]
    payoffs = []
    for i in range(100):
        payoff = sample_episode(game.new_initial_state(), policies)[0]
        payoffs.append(payoff)
    return np.mean(payoffs)

if __name__ == '__main__':
    game = pyspiel.load_game('kuhn_poker')
    num_actions = game.num_distinct_actions()
    observation_shape = game.information_state_tensor_shape()
    x = ppo.PPOAgent(num_actions, observation_shape, 'cpu')
    uniform_random_policy = policy.UniformRandomPolicy(game)
    print(get_expected_payoffs(game, x, uniform_random_policy))
