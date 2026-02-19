import random
from typing import Optional
from iig_rl_benchmark.algorithms.ppo import ppo
import numpy as np
import torch
import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode

def get_device_string():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

class PPOAgentPolicy(policy.Policy):
    def __init__(self,
                 game,
                 ppo_agent: ppo.PPOAgent,
                 player_id: int,
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

class PPONeuplAgentPolicy(policy.Policy):
    def __init__(self,
                 game,
                 ppo_agent: ppo.PPOConditionedOnPolicyRepresentationAgent,
                 player_id: int,
                 use_observation: bool,
                 policy_index: Optional[int] = None,
                 embedding: Optional[torch.Tensor] = None
                 ):
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
        assert (policy_index is None) != (embedding is None), "Exactly one of policy_index or embedding must be provided"
        assert policy_index is None
        self._policy_index = policy_index
        self._embedding = embedding
    
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
        action, log_probs, entropy, value, probs = self._ppo_agent.get_action_and_value(info_state, embedding=self._embedding, legal_actions_mask=legal_action_mask)
        # first dimension is batch dimension, so we squeeze.
        probs = probs.detach().numpy().squeeze(0)
        prob_dict = {a: probs[a] for a in legal_actions}
        return prob_dict

def get_expected_payoffs_agent(game: pyspiel.Game, p0_ppo_agent: ppo.PPOAgent, p1_policy: policy.Policy) -> float:
    policies = [PPOAgentPolicy(game, p0_ppo_agent, 0, False), p1_policy]
    payoffs = []
    for i in range(100):
        payoff = sample_episode(game.new_initial_state(), policies)[0]
        payoffs.append(payoff)
    return np.mean(payoffs)

def get_expected_payoffs(game: pyspiel.Game, p0_policy: policy.Policy, p1_policy: policy.Policy) -> float:
    policies = [p0_policy, p1_policy]
    payoffs = []
    for i in range(150):
        payoff = sample_episode(game.new_initial_state(), policies)[0]
        payoffs.append(payoff)
    return np.mean(payoffs)

def make_diverse_random_kuhn_poker_layer_init(game):
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, 2.2)
        if layer.out_features == game.num_distinct_actions():
            torch.nn.init.uniform(layer.bias, -1, 1)
        else:
            torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    return layer_init

class UniformRandomAgent(rl_agent.AbstractAgent):
  """An example agent class."""

  def __init__(self, player_id, num_actions, name="uniform_random_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick a random legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = random.choice(cur_legal_actions)
    probs = np.ones(self._num_actions) / self._num_actions

    return rl_agent.StepOutput(action=action, probs=probs)

if __name__ == '__main__':
    game = pyspiel.load_game('kuhn_poker')
    num_actions = game.num_distinct_actions()
    observation_shape = game.information_state_tensor_shape()
    # Here is how you can randomly initialize the weights of a PPO agent for Kuhn Poker:
    diverse_random_kuhn_poker_layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    x = ppo.PPOAgent(num_actions, observation_shape, 'cpu', diverse_random_kuhn_poker_layer_init)
    uniform_random_policy = policy.UniformRandomPolicy(game)
    print(get_expected_payoffs(game, x, uniform_random_policy))

