from typing import Literal
import math
import pyspiel
from open_spiel.python.algorithms import get_all_states

import torch
from pathlib import Path

from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed
from weight_autoencoder import (
    AutoencoderConfig,
    WeightAutoencoder,
    ppo_agent_to_vector,
    save_autoencoder,
    load_autoencoder,
)


def action_probabilities(agent: PPOAgent, state: pyspiel.State) -> dict[int, float]:
    """Return the agent’s action distribution at a decision state."""
    player_id = state.current_player()
    if player_id < 0 or state.is_terminal():
        return {}

    obs = torch.tensor(
        state.information_state_tensor(player_id),
        dtype=torch.float32,
        device=agent.actor[0].weight.device,
    ).unsqueeze(0)

    legal_actions = state.legal_actions(player_id)
    legal_mask = torch.zeros((1, agent.num_actions), dtype=torch.bool, device=obs.device)
    legal_mask[0, legal_actions] = True

    with torch.no_grad():
        _, _, _, _, probs = agent.get_action_and_value(obs, legal_mask)

    probs = probs.squeeze(0)
    return {a: probs[a].item() for a in legal_actions}



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    NUM_AGENTS = 1
    PPO_AGENT_HIDDEN_SIZE = 256

    game = pyspiel.load_game("kuhn_poker")
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    # Train autoencoder on agent weights
    ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for i in range(NUM_AGENTS)]

    print("Getting all states in the game...")
    all_states = get_all_states.get_all_states(game)

    decision_states = [
    state for state in all_states.values()
    if not state.is_terminal() and not state.is_chance_node()
]

    count = 0
    for state in all_states:
        print(state)
        count += 1

    print()
    print("Total: {} states.".format(count))
    for agent in ppo_agents:
        for state in decision_states:
            probs = action_probabilities(agent, state)
            print("State:", state)
            print("Action probabilities:", probs)
