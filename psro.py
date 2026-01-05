import os
from datetime import datetime
from pathlib import Path
import random
import uuid
from omegaconf import OmegaConf
import pyspiel
import torch
import numpy as np
from iig_rl_benchmark.algorithms.psro import run_psro as iig_run_psro
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent, PPOConditionedOnPolicyRepresentationAgent
from utils import PPONeuplAgentPolicy, get_device_string
from typing import Union

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def run_psro(game_name: str = 'kuhn_poker'):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Fix for MacOS. Claude told me this is safe.
    game = pyspiel.load_game(game_name)
    config_path = 'configs/psro_liars_dice_ppo.yaml'
    # config_path = 'configs/psro_liars_dice_best_hparams_dqn.yaml'
    algorithm_config = OmegaConf.load(config_path)
    args = OmegaConf.load('configs/experiment.yaml')
    args.algorithm = algorithm_config
    args.game = game_name
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    random_str = uuid.uuid4().hex[:3]
    if 'hidden_size' in args.algorithm.inner_rl_agent:
        experiment_dir = os.path.join(
            args.save_dir, args.group_name, args.algorithm.algorithm_name, args.algorithm.inner_rl_agent.algorithm_name, f'hs{args.algorithm.inner_rl_agent.hidden_size}', args.game, f'{time_str}_{random_str}'
        )
    else:
        experiment_dir = os.path.join(
            args.save_dir, args.group_name, args.algorithm.algorithm_name, args.algorithm.inner_rl_agent.algorithm_name, args.game, f'{time_str}_{random_str}'
        )
    args.experiment_dir = experiment_dir
    runner = iig_run_psro.RunPSRO(args, game, is_neupl=False)
    runner.run()

def run_neupl(game_name: str = 'kuhn_poker'):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Fix for MacOS. Claude told me this is safe.
    game = pyspiel.load_game(game_name)
    config_path = 'configs/neupl.yaml'
    algorithm_config = OmegaConf.load(config_path)
    args = OmegaConf.load('configs/experiment.yaml')
    args.algorithm = algorithm_config
    args.algorithm.training_strategy_selector = 'exhaustive'
    args.algorithm.number_policies_selected = -1
    args.game = game_name
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    random_str = uuid.uuid4().hex[:3]
    experiment_dir = os.path.join(
        args.save_dir, args.group_name, 'neupl', args.algorithm.inner_rl_agent.algorithm_name, f'hs{args.algorithm.inner_rl_agent.hidden_size}', args.game, f'{time_str}_{random_str}'
    )
    args.experiment_dir = experiment_dir
    args.device = 'cpu' # get_device_string()
    runner = iig_run_psro.RunPSRO(args, game, is_neupl=True)
    runner.run_neupl() 



def load_ppo_agents_from_psro(
        game_short_name: str = 'kuhn_poker',
        player_id: Union[int, None] = None,
        hidden_size: int = 512,
        shuffle: bool = True,
):
    PATH = f"results/test/psro/ppo/hs{hidden_size}/{game_short_name}"
    base_dir = Path(PATH)
    game = pyspiel.load_game(game_short_name)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    ppo_agents = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir():
            glob = "policy*.pt" if player_id is None else f"policy{player_id}_ckpt*.pt"
            files = list(subdir.glob(glob))
            for file in files:
                policy = torch.load(file)
                agent = PPOAgent(num_actions, info_state_size, 'cpu', hidden_size=hidden_size)
                agent.actor.load_state_dict(policy)
                ppo_agents.append(agent)
    if shuffle:
        random.shuffle(ppo_agents)
    print(f"Loaded {len(ppo_agents)} PPO agents from {PATH}")
    return ppo_agents


def load_ppo_agents_from_single_psro_folder(
        game_short_name: str = 'kuhn_poker',
        player_id: Union[int, None] = None,
        hidden_size: int = 512,
        shuffle: bool = True,
        folder_selection: str = 'oldest',  # 'newest', 'oldest', or specific index
        max_agents: int = None,  # Maximum number of agents to load (None = all)
):
    """
    Load PPO agents from a single PSRO date folder.

    Args:
        game_short_name: Name of the game
        player_id: Which player's policies to load (None = both players)
        hidden_size: Hidden size of PPO agents
        shuffle: Whether to shuffle agents after loading
        folder_selection: Which folder to select - 'newest', 'oldest', or integer index
        max_agents: Maximum number of agents to return (None = all agents from folder)

    Returns:
        List of PPOAgent objects from the selected folder
    """
    PATH = f"results/test/psro/ppo/hs{hidden_size}/{game_short_name}"
    base_dir = Path(PATH)
    game = pyspiel.load_game(game_short_name)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Get all subdirectories
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

    if not subdirs:
        raise ValueError(f"No subdirectories found in {PATH}")

    # Select the folder based on folder_selection parameter
    if folder_selection == 'newest':
        selected_subdir = subdirs[-1]  # Last in sorted order (most recent date)
    elif folder_selection == 'oldest':
        selected_subdir = subdirs[0]   # First in sorted order (oldest date)
    elif isinstance(folder_selection, int):
        if folder_selection < 0 or folder_selection >= len(subdirs):
            raise ValueError(f"Folder index {folder_selection} out of range (0-{len(subdirs)-1})")
        selected_subdir = subdirs[folder_selection]
    else:
        raise ValueError(f"Invalid folder_selection: {folder_selection}. Use 'newest', 'oldest', or integer index")

    print(f"Selected folder: {selected_subdir.name}")

    # Load all agents from the selected folder
    ppo_agents = []
    glob_pattern = "policy*.pt" if player_id is None else f"policy{player_id}_ckpt*.pt"
    files = list(selected_subdir.glob(glob_pattern))

    for file in files:
        policy = torch.load(file)
        agent = PPOAgent(num_actions, info_state_size, 'cpu', hidden_size=hidden_size)
        agent.actor.load_state_dict(policy)
        ppo_agents.append(agent)

    print(f"Loaded {len(ppo_agents)} PPO agents from {selected_subdir}")

    # Shuffle if requested
    if shuffle:
        random.shuffle(ppo_agents)

    # Limit to max_agents if specified
    if max_agents is not None and max_agents < len(ppo_agents):
        ppo_agents = ppo_agents[:max_agents]
        print(f"Limited to {max_agents} agents")

    return ppo_agents


def load_ppo_agents_from_neupl(
        game_short_name: str = 'kuhn_poker',
        hidden_size: int = 512,
        policy_embedding_size: int = 64,
        shuffle: bool = True,
):
    """loads a single neupl checkpoint."""
    PATH = f"results/test/neupl/ppo/hs{hidden_size}/{game_short_name}"
    base_dir = Path(PATH)
    game = pyspiel.load_game(game_short_name)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    import glob
    # List the most recent 10 subdirectories in base_dir
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    # Sort by modification time, most recent first
    subdirs = sorted(subdirs, key=lambda d: d.stat().st_mtime, reverse=True)
    recent_subdirs = subdirs[:10]
    dir_info = []
    print("Most recent 10 directories and .pt file counts:")
    for idx, subdir in enumerate(recent_subdirs):
        pt_files = list(subdir.glob("*.pt"))
        dir_info.append((subdir, len(pt_files)))
        print(f"[{idx}]: {subdir.name} - {len(pt_files)} .pt files")

    selected_idx = None
    while selected_idx is None:
        try:
            user_input = input("Select the directory to load by entering its index (0-9): ").strip()
            i = int(user_input)
            if 0 <= i < len(dir_info):
                selected_idx = i
            else:
                print("Invalid index. Try again.")
        except Exception:
            print("Invalid input. Enter an integer between 0 and 9.")

    chosen_subdir = dir_info[selected_idx][0]
    print(f"Loading from directory: {chosen_subdir}")

    pt0 = max(chosen_subdir.glob("policy0_ckpt*.pt"))
    pt1 = max(chosen_subdir.glob("policy1_ckpt*.pt"))

    # pt0 = max(chosen_subdir.glob("policy0_ckpt24.pt"))
    # pt1 = max(chosen_subdir.glob("policy1_ckpt24.pt"))
    agent0 = PPOConditionedOnPolicyRepresentationAgent(
        num_actions, info_state_size, 'cpu',
        num_policies=100, policy_embedding_size=policy_embedding_size, hidden_size=hidden_size
    )
    agent1 = PPOConditionedOnPolicyRepresentationAgent(
        num_actions, info_state_size, 'cpu',
        num_policies=100, policy_embedding_size=policy_embedding_size, hidden_size=hidden_size
    )
    agent0.load(pt0)
    agent1.load(pt1)
    return agent0, agent1

def make_ppo_policies_from_neupl_agents(
    game_name: str,
    agents: list[PPOConditionedOnPolicyRepresentationAgent],
    original_num_policies: int = 100,
    num_policies_to_make: int = 1000,
):
    game = pyspiel.load_game(game_name)
    policies = []
    for player_id in range(2):
        player_policies = []
        for i in range(num_policies_to_make):
            policy_index_1 = torch.tensor(random.randint(0, original_num_policies - 1))
            policy_index_2 = torch.tensor(random.randint(0, original_num_policies - 1))
            embedding_1 = agents[player_id].embedding_prenorm(policy_index_1)
            embedding_2 = agents[player_id].embedding_prenorm(policy_index_2)
            mixture = random.random()
            embedding = (embedding_1 * mixture + embedding_2 * (1 - mixture)).unsqueeze(0)
            normed_embedding = agents[player_id].embedding_norm(embedding)
            player_policies.append((
                normed_embedding.squeeze(0),
                PPONeuplAgentPolicy(game, agents[player_id], player_id, use_observation=False, embedding=normed_embedding)
            ))
        policies.append(player_policies)
    return policies

def make_neupl_policies(
    game_short_name: str,
    hidden_size: int,
    policy_embedding_size: int,
    original_num_policies: int = 100,
    num_policies_to_make: int = 1000,
):
    agents = load_ppo_agents_from_neupl(game_short_name=game_short_name, hidden_size=hidden_size, policy_embedding_size=policy_embedding_size)
    return make_ppo_policies_from_neupl_agents(game_short_name, agents, original_num_policies=original_num_policies, num_policies_to_make=num_policies_to_make)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--neupl', action='store_true', help='Run neupl instead of standard psro')
    parser.add_argument('--game_name', type=str, default='kuhn_poker', help='Game to run')
    args = parser.parse_args()

    game_name = args.game_name
    if args.neupl:
        run_neupl(game_name)
    else:
        run_psro(game_name)