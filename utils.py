import random
from typing import Optional
from iig_rl_benchmark.algorithms.ppo import ppo
import numpy as np
import torch
import pyspiel
from open_spiel.python import policy
from open_spiel.python import rl_agent
from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode
from open_spiel.python.algorithms.expected_game_score import policy_value

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
        device = next(self._ppo_agent.parameters()).device
        legal_action_mask = torch.zeros(self._game.num_distinct_actions(), device=device)
        legal_action_mask[legal_actions] = 1
        info_state = torch.tensor(state.information_state_tensor(player_id), dtype=torch.float32, device=device)
        action, log_probs, entropy, value, probs = self._ppo_agent.get_action_and_value(info_state, legal_action_mask)
        probs = probs.detach().cpu().numpy()
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
        device = next(self._ppo_agent.parameters()).device
        legal_action_mask = torch.zeros(self._game.num_distinct_actions(), device=device)
        legal_action_mask[legal_actions] = 1
        info_state = torch.tensor(state.information_state_tensor(player_id), dtype=torch.float32, device=device)
        action, log_probs, entropy, probs = self._ppo_agent.get_action(info_state, embedding=self._embedding, legal_actions_mask=legal_action_mask)
        # first dimension is batch dimension, so we squeeze.
        probs = probs.detach().cpu().numpy().squeeze(0)
        prob_dict = {a: probs[a] for a in legal_actions}
        return prob_dict

def get_expected_payoffs_agent(game: pyspiel.Game, p0_ppo_agent: ppo.PPOAgent, p1_policy: policy.Policy) -> float:
    policies = [PPOAgentPolicy(game, p0_ppo_agent, 0, False), p1_policy]
    payoffs = []
    for i in range(100):
        payoff = sample_episode(game.new_initial_state(), policies)[0]
        payoffs.append(payoff)
    return np.mean(payoffs)

def _batched_action_probs(pol, states, player_id, game):
    """Action-probability rows for ``states`` in one network forward pass.

    Returns an (K, num_distinct_actions) array (row i = probabilities for states[i]) for the
    NN-backed policy wrappers, or None for any other policy type (caller falls back to
    per-state ``action_probabilities``). Exact same math as the per-state path -- the softmax
    is deterministic given weights -- just evaluated as one batch.
    """
    if not isinstance(pol, (PPOAgentPolicy, PPONeuplAgentPolicy)):
        return None
    agent = pol._ppo_agent
    device = next(agent.parameters()).device
    num_actions = game.num_distinct_actions()
    info = torch.tensor(np.array([s.information_state_tensor(player_id) for s in states]),
                        dtype=torch.float32, device=device)
    mask = torch.zeros((len(states), num_actions), device=device)
    for i, s in enumerate(states):
        mask[i, s.legal_actions(player_id)] = 1
    with torch.no_grad():
        if isinstance(pol, PPONeuplAgentPolicy):
            _, _, _, probs = agent.get_action(info, embedding=pol._embedding, legal_actions_mask=mask)
        else:
            _, probs = agent.get_action(info, legal_actions_mask=mask)
    return probs.detach().cpu().numpy()


def tabularize_policy(game: pyspiel.Game, pol: policy.Policy, player_id: int) -> policy.TabularPolicy:
    """Snapshot ``pol`` into an OpenSpiel TabularPolicy, filling only ``player_id``'s rows.

    Only the rows for states where ``player_id`` acts are set (the others keep the default
    uniform value and are never queried, since in a pair each tabular policy is only asked
    about its own player's states). NN-backed policies are evaluated in a single batched
    forward pass; other policies fall back to per-state ``action_probabilities``.
    """
    tab = policy.TabularPolicy(game)
    states = [s for s in tab.states if s.current_player() == player_id]
    if not states:
        return tab
    probs = _batched_action_probs(pol, states, player_id, game)
    for i, s in enumerate(states):
        row = tab.action_probability_array[tab.state_index(s)]
        if probs is not None:
            row[:] = probs[i]
        else:  # fallback: query the policy one state at a time
            row[:] = 0.0
            for a, p in pol.action_probabilities(s, player_id).items():
                row[a] = p
    return tab


_TERMINAL_CACHE = {}


def _enumerate_terminals(game, ref_tab):
    """Walk the tree once, independent of any policy.

    Returns ``(weight, dec0, dec1, num_cells)`` where, over all T terminal histories:
      - ``weight``: (T,) array of ``chance_reach(h) * return_to_player0(h)``,
      - ``dec0`` / ``dec1``: (T, Lmax) int arrays of *flat* ``state_index * num_actions +
        action`` indices for each player's decisions along the path, right-padded with a
        sentinel (``num_cells``) that gathers to 1.0 so it doesn't affect the reach product,
      - ``num_cells``: ``num_states * num_actions`` (the sentinel / flat-array length).

    Cached per game so repeated PayoffPredictor calls in a run reuse the walk.
    """
    key = str(game)
    if key in _TERMINAL_CACHE:
        return _TERMINAL_CACHE[key]
    A = game.num_distinct_actions()
    num_cells = len(ref_tab.states) * A
    weights, dec0, dec1 = [], [], []

    def rec(state, chance, d0, d1):
        if state.is_terminal():
            weights.append(chance * state.returns()[0])
            dec0.append(d0); dec1.append(d1)
            return
        if state.is_chance_node():
            for a, p in state.chance_outcomes():
                rec(state.child(a), chance * p, d0, d1)
            return
        cur = state.current_player()
        base = ref_tab.state_index(state) * A
        for a in state.legal_actions(cur):
            flat = base + a
            if cur == 0:
                rec(state.child(a), chance, d0 + [flat], d1)
            else:
                rec(state.child(a), chance, d0, d1 + [flat])

    rec(game.new_initial_state(), 1.0, [], [])

    def _pad(dec):
        L = max((len(d) for d in dec), default=0)
        out = np.full((len(dec), L), num_cells, dtype=np.int64)  # sentinel -> 1.0
        for i, d in enumerate(dec):
            out[i, :len(d)] = d
        return out

    result = (np.asarray(weights, dtype=np.float64), _pad(dec0), _pad(dec1), num_cells)
    _TERMINAL_CACHE[key] = result
    return result


def _reach_vectors(tab_by_index, indices, dec, num_cells):
    """Reach probability of each terminal (rows of ``dec``) under each listed policy.

    Returns an (len(indices), T) array: entry (k, h) = product of the policy's action
    probabilities along terminal h's decisions for that player.
    """
    R = np.empty((len(indices), dec.shape[0]), dtype=np.float64)
    for k, idx in enumerate(indices):
        flat = np.append(tab_by_index[idx].action_probability_array.reshape(-1), 1.0)  # sentinel=1
        R[k] = flat[dec].prod(axis=1)
    return R


def grid_payoffs_via_reach(game, tab_p1, tab_p2, pairs):
    """Exact player-0 payoff for each ``(p1_idx, p2_idx)`` in ``pairs`` (list order preserved).

    Uses the sequence-form decomposition V(i,j) = sum_h u0(h) c(h) r0^i(h) r1^j(h): one tree
    walk (cached) plus one reach vector per distinct policy, then a single matrix multiply for
    the whole grid -- no per-pair tree traversal. ``tab_p1``/``tab_p2`` map policy index to the
    TabularPolicy from ``tabularize_policy`` (players 0 and 1 respectively).
    """
    ref_tab = policy.TabularPolicy(game)
    weight, dec0, dec1, num_cells = _enumerate_terminals(game, ref_tab)
    p1_idx = sorted(tab_p1); p2_idx = sorted(tab_p2)
    R0 = _reach_vectors(tab_p1, p1_idx, dec0, num_cells)   # (n1, T)
    R1 = _reach_vectors(tab_p2, p2_idx, dec1, num_cells)   # (n2, T)
    M = (R0 * weight) @ R1.T                                # (n1, n2), exact P0 payoff grid
    row1 = {i: r for r, i in enumerate(p1_idx)}
    row2 = {j: r for r, j in enumerate(p2_idx)}
    return [float(M[row1[i], row2[j]]) for i, j in pairs]


def get_expected_payoffs(game: pyspiel.Game, p0_policy: policy.Policy, p1_policy: policy.Policy, exact=False) -> float:
    if exact:
        return _get_expected_payoffs_exact(game, p0_policy, p1_policy)
    else:
        return _get_expected_payoffs_sampled(game, p0_policy, p1_policy)


def _get_expected_payoffs_exact(game: pyspiel.Game, p0_policy: policy.Policy, p1_policy: policy.Policy) -> float:
    return policy_value(game.new_initial_state(), [p0_policy, p1_policy])[0]

def _get_expected_payoffs_sampled(game: pyspiel.Game, p0_policy: policy.Policy, p1_policy: policy.Policy) -> float:
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

# Policy as RL Agent ###################################################################
# I stg this must already be a class somewhere else, but I can't find it. ##############
class SyntheticState:
    def __init__(self, legal_actions, current_player, information_state_tensor=None, information_state_string=None):
        self._information_state_string = information_state_string
        self._information_state_tensor = information_state_tensor
        self._legal_actions = legal_actions
        self._current_player = current_player
    def legal_actions(self, pl):
        return self._legal_actions
    def information_state_string(self, pl=None):
        assert (pl is None) or (pl == self._current_player), "Information state string is only valid for the current player"
        return self._information_state_string
    def information_state_tensor(self, pl=None):
        assert (pl is None) or (pl == self._current_player), "Information state is only valid for the current player"
        return self._information_state_tensor
    def current_player(self):
        return self._current_player
    def is_simultaneous_node(self):
        return False

class PolicyAsAgent(rl_agent.AbstractAgent):
    """use a policy as an RL agent"""
    def __init__(self, player_id, num_actions, rng, policy):
        self._player_id = player_id
        self._rng = rng
        self._num_actions = num_actions
        self._policy = policy
    def action_probabilities(self, *args, **kwargs):
        return self._policy.action_probabilities(*args, **kwargs)
    def step(self, time_step, is_evaluation=False):
        if time_step.last():
            return
        synthetic_state = SyntheticState(
            # time_step.observations['information_state_string'][self._player_id],
            legal_actions=time_step.observations['legal_actions'][self._player_id],
            current_player=time_step.observations["current_player"],
            information_state_tensor=time_step.observations['info_state'][self._player_id],
            # information_state_string=time_step.observations['information_state_string'][self._player_id],
        )
        pol = self._policy.action_probabilities(synthetic_state, self._player_id)
        pol_list = list(zip(*pol.items()))
        action = self._rng.choice(pol_list[0], p=np.array(pol_list[1]))
        probs = np.zeros(self._num_actions)
        for k, v in pol.items():
            probs[k] = v
        return rl_agent.StepOutput(action=action, probs=probs)

########################################################################################

if __name__ == '__main__':
    game = pyspiel.load_game('kuhn_poker')
    num_actions = game.num_distinct_actions()
    observation_shape = game.information_state_tensor_shape()
    # Here is how you can randomly initialize the weights of a PPO agent for Kuhn Poker:
    diverse_random_kuhn_poker_layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    x = ppo.PPOAgent(num_actions, observation_shape, 'cpu', diverse_random_kuhn_poker_layer_init)
    uniform_random_policy = policy.UniformRandomPolicy(game)
    print(get_expected_payoffs(game, x, uniform_random_policy))

