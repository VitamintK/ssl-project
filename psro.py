import os
from datetime import datetime
from pathlib import Path
import random
import uuid
from omegaconf import OmegaConf
import pyspiel
import torch
from iig_rl_benchmark.algorithms.psro import run_psro as iig_run_psro
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent

def run_psro(game_name: str = 'kuhn_poker'):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Fix for MacOS. Claude told me this is safe.
    game = pyspiel.load_game(game_name)
    num_actions = game.num_distinct_actions()
    observation_shape = game.information_state_tensor_shape()
    config_path = 'configs/psro_liars_dice_test.yaml'
    algorithm_config = OmegaConf.load(config_path)
    args = OmegaConf.load('configs/experiment.yaml')
    args.algorithm = algorithm_config
    args.game = game_name
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    random_str = uuid.uuid4().hex[:3]
    experiment_dir = os.path.join(
        args.save_dir, args.group_name, args.algorithm.algorithm_name, args.algorithm.inner_rl_agent.algorithm_name, f'hs{args.algorithm.inner_rl_agent.hidden_size}', args.game, f'{time_str}_{random_str}'
    )
    args.experiment_dir = experiment_dir
    runner = iig_run_psro.RunPSRO(args, game)
    runner.run()

def load_ppo_agents_from_psro(
        hidden_size: int = 512,
        shuffle: bool = True,
):
    PATH = f"results/test/psro/ppo/hs{hidden_size}/kuhn_poker"
    base_dir = Path(PATH)
    game = pyspiel.load_game('kuhn_poker')
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    ppo_agents = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.glob("*policy*.pt"))
            for file in files:
                policy = torch.load(file)
                agent = PPOAgent(num_actions, info_state_size, 'cpu', hidden_size=hidden_size)
                agent.actor.load_state_dict(policy)
                ppo_agents.append(agent)
    if shuffle:
        random.shuffle(ppo_agents)
    print(f"Loaded {len(ppo_agents)} PPO agents from {PATH}")
    return ppo_agents

if __name__ == '__main__':
    run_psro()
