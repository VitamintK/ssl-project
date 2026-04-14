import os
from datetime import datetime
from pathlib import Path
import random
from typing import Literal, Optional, Union
import uuid
from omegaconf import OmegaConf
import pyspiel
import torch
import numpy as np
from iig_rl_benchmark.algorithms.psro import run_psro as iig_run_psro
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent, PPOConditionedOnPolicyRepresentationAgent
from utils import PPONeuplAgentPolicy, get_device_string

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

def run_neupl(game_name: str = 'kuhn_poker', use_randall_loss=False):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Fix for MacOS. Claude told me this is safe.
    game = pyspiel.load_game(game_name)
    config_path = 'configs/neupl.yaml'
    algorithm_config = OmegaConf.load(config_path)
    args = OmegaConf.load('configs/experiment.yaml')
    args.algorithm = algorithm_config
    args.algorithm.training_strategy_selector = 'exhaustive'
    args.algorithm.number_policies_selected = -1
    args.algorithm.use_randall_loss = use_randall_loss
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


def select_neupl_directory(
        game_short_name: str,
        use_randall_loss: bool,
        hidden_size: int = 512,
) -> str:
    """Prompt the user to pick a NEUPL checkpoint directory and return its name.

    Intended to be called in the main process before spawning workers, so
    that workers never need to call input().
    """
    PATH = f"results/test/neupl/ppo/hs{hidden_size}/{game_short_name}"
    base_dir = Path(PATH)
    subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()],
                     key=lambda d: d.stat().st_mtime, reverse=True)

    filtered = []
    for subdir in subdirs:
        config_path = subdir / "config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    cfg = json.load(f)
                if cfg.get("use_randall_loss") == use_randall_loss:
                    filtered.append(subdir)
            except Exception as e:
                print(f"Warning: Failed to parse {config_path}: {e}")
        # skip dirs without a config when filtering by use_randall_loss

    recent = filtered[:10]
    print(f"\nNEUPL directories for use_randall_loss={use_randall_loss}:")
    dir_info = []
    for idx, subdir in enumerate(recent):
        pt_files = list(subdir.glob("*.pt"))
        dir_info.append((subdir, len(pt_files)))
        print(f"  [{idx}]: {subdir.name} - {len(pt_files)} .pt files")

    selected_idx = None
    while selected_idx is None:
        try:
            i = int(input("Select directory index: ").strip())
            if 0 <= i < len(dir_info):
                selected_idx = i
            else:
                print(f"Enter a number between 0 and {len(dir_info) - 1}.")
        except Exception:
            print("Enter an integer.")

    return dir_info[selected_idx][0].name


def load_ppo_agents_from_neupl(
        game_short_name: str = 'kuhn_poker',
        use_randall_loss: bool = False,
        hidden_size: int = 512,
        policy_embedding_size: int = 64,
        # shuffle: bool = True,
        dir_name: Optional[str] = None,
):
    """loads a single neupl checkpoint."""
    PATH = f"results/test/neupl/ppo/hs{hidden_size}/{game_short_name}"
    base_dir = Path(PATH)
    game = pyspiel.load_game(game_short_name)
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    if dir_name is None:
        dir_name = select_neupl_directory(game_short_name, use_randall_loss, hidden_size)
        dir_name = base_dir / dir_name
    else:
        dir_name = base_dir / dir_name
    print(f"Loading from directory: {dir_name}")
    config_path = dir_name / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            num_policies = config.get("num_policies")
        except Exception as e:
            print(f"Warning: Failed to parse {config_path}: {e}")
            num_policies = 100
    else:
        num_policies = 100

    pt0 = max(dir_name.glob("policy0_ckpt*.pt"))
    pt1 = max(dir_name.glob("policy1_ckpt*.pt"))

    # pt0 = max(chosen_subdir.glob("policy0_ckpt24.pt"))
    # pt1 = max(chosen_subdir.glob("policy1_ckpt24.pt"))
    agent0 = PPOConditionedOnPolicyRepresentationAgent(
        num_actions, info_state_size, 'cpu',
        num_policies=num_policies, policy_embedding_size=policy_embedding_size, hidden_size=hidden_size
    )
    agent1 = PPOConditionedOnPolicyRepresentationAgent(
        num_actions, info_state_size, 'cpu',
        num_policies=num_policies, policy_embedding_size=policy_embedding_size, hidden_size=hidden_size
    )
    agent0.load(pt0)
    agent1.load(pt1)
    agent0.eval()
    agent1.eval()
    return agent0, agent1

def make_ppo_policies_from_neupl_agents(
    game_name: str,
    agents: list[PPOConditionedOnPolicyRepresentationAgent],
    original_num_policies: int = 100,
    num_policies_to_make: int = 1000,
    interpolate_prenorm: bool = True,
    sampling_mode: Literal["interpolate", "gaussian"] = "interpolate",
):
    """
    Generate policies from NEUPL agents using different sampling methods.

    Args:
        game_name: Name of the game
        agents: List of PPOConditionedOnPolicyRepresentationAgent for each player
        original_num_policies: Number of original policies in the NEUPL training
        num_policies_to_make: Number of new policies to generate
        interpolate_prenorm: If True, work in pre-norm space then normalize at the end
        sampling_mode: How to sample new embeddings:
            - "interpolate": Interpolate between two random existing policies
            - "gaussian": Sample from multivariate Gaussian fit to existing policies

    Returns:
        List of (embedding, policy) tuples for each player
    """
    game = pyspiel.load_game(game_name)
    policies = []

    for player_id in range(2):
        player_policies = []

        if sampling_mode == "gaussian":
            # Collect all existing embeddings
            existing_embeddings = []
            for policy_idx in range(1, original_num_policies):  # Skip 0 (uniform random)
                policy_index_tensor = torch.tensor(policy_idx)
                if interpolate_prenorm:
                    # Get pre-norm embeddings for Gaussian fitting
                    embedding = agents[player_id].embedding_prenorm(policy_index_tensor)
                    existing_embeddings.append(embedding.detach().cpu().numpy())
                else:
                    # Get post-norm embeddings for Gaussian fitting
                    embedding = agents[player_id].policy_representation_embedding(policy_index_tensor)
                    existing_embeddings.append(embedding.detach().cpu().numpy())

            existing_embeddings = np.array(existing_embeddings)

            # Fit multivariate Gaussian
            mean = np.mean(existing_embeddings, axis=0)
            cov = np.cov(existing_embeddings.T)

            # Sample from the Gaussian
            sampled_embeddings = np.random.multivariate_normal(mean, cov, num_policies_to_make)

            # Convert to policies
            for sampled_embedding in sampled_embeddings:
                embedding_tensor = torch.tensor(sampled_embedding, dtype=torch.float32)

                if interpolate_prenorm:
                    # Apply normalization to the sampled pre-norm embedding
                    normed_embedding = agents[player_id].embedding_norm(embedding_tensor.unsqueeze(0))
                else:
                    # Already in post-norm space
                    normed_embedding = embedding_tensor.unsqueeze(0)

                player_policies.append((
                    normed_embedding.squeeze(0),
                    PPONeuplAgentPolicy(game, agents[player_id], player_id, use_observation=False, embedding=normed_embedding)
                ))

        elif sampling_mode == "interpolate":
            # Original interpolation method
            for i in range(num_policies_to_make):
                policy_index_1 = torch.tensor(random.randint(1, original_num_policies - 1))
                policy_index_2 = torch.tensor(random.randint(1, original_num_policies - 1))
                mixture = random.random()

                if interpolate_prenorm:
                    embedding_1 = agents[player_id].embedding_prenorm(policy_index_1)
                    embedding_2 = agents[player_id].embedding_prenorm(policy_index_2)
                    embedding = (embedding_1 * mixture + embedding_2 * (1 - mixture)).unsqueeze(0)
                    normed_embedding = agents[player_id].embedding_norm(embedding)
                else:
                    embedding_1 = agents[player_id].policy_representation_embedding(policy_index_1)
                    embedding_2 = agents[player_id].policy_representation_embedding(policy_index_2)
                    normed_embedding = (embedding_1 * mixture + embedding_2 * (1 - mixture)).unsqueeze(0)

                player_policies.append((
                    normed_embedding.squeeze(0),
                    PPONeuplAgentPolicy(game, agents[player_id], player_id, use_observation=False, embedding=normed_embedding)
                ))
        else:
            raise ValueError(f"Invalid sampling_mode: {sampling_mode}. Must be 'interpolate' or 'gaussian'.")

        policies.append(player_policies)

    return policies

def make_neupl_policies(
    game_short_name: str,
    neupl_config: dict,
    original_num_policies: int = 100,
    num_policies_to_make: int = 1000,
    directory: Optional[str] = None,
    interpolate_prenorm: bool = True,
    sampling_mode: Literal["interpolate", "gaussian"] = "interpolate",
):
    agents = load_ppo_agents_from_neupl(game_short_name=game_short_name, **neupl_config, dir_name=directory)
    return make_ppo_policies_from_neupl_agents(
        game_short_name,
        agents,
        original_num_policies=original_num_policies,
        num_policies_to_make=num_policies_to_make,
        interpolate_prenorm=interpolate_prenorm,
        sampling_mode=sampling_mode,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--neupl', action='store_true', help='Run neupl instead of standard psro')
    parser.add_argument('--use_randall_loss', action='store_true', help='Use randall loss instead of standard loss')
    parser.add_argument('--game_name', type=str, default='kuhn_poker', help='Game to run')
    args = parser.parse_args()
    if args.use_randall_loss:
        assert args.neupl, "Use randall loss only with neupl"

    game_name = args.game_name
    if args.neupl:
        run_neupl(game_name, use_randall_loss=args.use_randall_loss)
    else:
        run_psro(game_name)