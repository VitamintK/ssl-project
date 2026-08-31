import json
import logging
import os
import sys
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
    if game_name == 'leduc_poker':
        config_path = 'configs/neupl_leduc.yaml'
    else:
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



def best_response_value(game, fixed_policy, fixed_player, br_player):
    """Value ``br_player`` gets by exactly best-responding to ``fixed_policy``.

    ``fixed_policy`` plays ``fixed_player``; only its states matter. Requires
    ``fixed_policy.action_probabilities(state)`` (OpenSpiel Policy interface).
    """
    from open_spiel.python.algorithms import best_response as _best_response
    from utils import tabularize_policy

    # Only fixed_player's rows matter (br_player is an exact best response); tabularize_policy
    # fills them in one batched NN forward. Opponent rows stay default (ignored).
    profile = tabularize_policy(game, fixed_policy, fixed_player)
    responder = _best_response.BestResponsePolicy(game, br_player, profile)
    return float(responder.value(game.new_initial_state()))


def run_neupl_v2(game_name: str = 'kuhn_poker', use_randall_loss: bool = False, T: int = None,
                 debug: bool = False, gt_payoffs: bool = False, save_logs: bool = False,
                 apsro: bool = False, apsro_exploited_player: str = 'p1', apsro_bandit: str = 'hedge',
                 apsro_exploited_lr_scale: float = 1.0,
                 num_iterations: int = 680, num_pols_sampled: int = 8,
                 total_episodes_per_policy: int = 400, expl_check_episode_interval: int = None,
                 save_checkpoints: bool = True):
    """Custom NeuPL training loop.

    Replaces the iig_run_psro.RunPSRO-based loop with a hand-written one.
    Policy population is initialized identically to init_neupl_ppo_responder.
    Each outer iteration:
      1. Train Player 0's policies (indices 1..K) by sampling uniformly from
         [1..K] and playing against an opponent sampled from the restricted-game Nash.
      2. Update empirical payoffs for the active (K+1)×(K+1) subgame.
      3. Recompute Nash strategies.
      4. Repeat steps 1-3 for Player 1.
    K increments by 1 each outer iteration until it reaches N.
    """
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    if apsro and (use_randall_loss or gt_payoffs):
        raise ValueError("apsro is incompatible with use_randall_loss / gt_payoffs "
                         "(they require the payoff matrix, which apsro does not build)")
    if apsro_exploited_player not in ("p1", "p2"):
        raise ValueError(f"apsro_exploited_player must be 'p1' or 'p2', got {apsro_exploited_player!r}")

    import time
    import copy as _copy

    from open_spiel.python import rl_environment
    from open_spiel.python.rl_environment import TimeStep as RLTimeStep
    from open_spiel.python.algorithms import exploitability, lp_solver, policy_aggregator
    from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode as tabular_sample_episode
    from iig_rl_benchmark.algorithms.psro import rl_policy, rl_oracle

    game = pyspiel.load_game(game_name)
    # Kuhn APSRO uses its own hyperparameters (configs/neupl_apsro.yaml); everything else
    # (symmetric neupl_v2, and APSRO on other games) falls back to the shared neupl.yaml.
    if apsro and game_name == 'kuhn_poker':
        config_path = 'configs/neupl_apsro.yaml'
    elif game_name == 'leduc_poker':
        config_path = 'configs/neupl_leduc.yaml'
    else:
        config_path = 'configs/neupl.yaml'
    algorithm_config = OmegaConf.load(config_path)
    args = OmegaConf.load('configs/experiment.yaml')
    args.algorithm = algorithm_config
    args.game = game_name
    args.device = 'cpu'

    alg = args.algorithm
    N = alg.num_policies
    sims_per_entry = alg.sims_per_entry
    # number_training_episodes = alg.number_training_episodes
    # num_pols_sampled / total_episodes_per_policy / num_iterations are function params.
    # expl_check_episode_interval = 20000
    if expl_check_episode_interval is None:
        expl_check_episode_interval = args.algorithm.expl_check_episode_interval
    RANDALL_LR = 0.0003
    RANDALL_LOSS_EPOCHS = 1

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    random_str = uuid.uuid4().hex[:3]

    logger = logging.getLogger(f"neupl_v2.{time_str}.{random_str}")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if save_logs:
        log_dir = os.path.join("logs", "neupl")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{time_str}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Saving NeuPL logs to %s", log_path)
    experiment_dir = os.path.join(
        args.save_dir, args.group_name, 'neupl',
        alg.inner_rl_agent.algorithm_name,
        f'hs{alg.inner_rl_agent.hidden_size}', game_name, f'{time_str}_{random_str}',
    )
    os.makedirs(experiment_dir, exist_ok=True)

    env = rl_environment.Environment(game)
    num_actions = env.action_spec()["num_actions"]
    info_state_size = env.observation_spec()["info_state"][0]

    agent_class = rl_policy.PPOPolicy
    agent_kwargs = {
        "info_state_size": info_state_size,
        "num_actions": num_actions,
        "steps_per_batch": alg.inner_rl_agent.num_steps,
        "num_minibatches": alg.inner_rl_agent.num_minibatches,
        "update_epochs": alg.inner_rl_agent.update_epochs,
        "learning_rate": alg.inner_rl_agent.learning_rate,
        "gae": alg.inner_rl_agent.gae,
        "gamma": alg.inner_rl_agent.gamma,
        "gae_lambda": alg.inner_rl_agent.gae_lambda,
        "hidden_size": alg.inner_rl_agent.hidden_size,
        "normalize_advantages": alg.inner_rl_agent.norm_adv,
        "clip_coef": alg.inner_rl_agent.clip_coef,
        "clip_vloss": alg.inner_rl_agent.clip_vloss,
        "entropy_coef": alg.inner_rl_agent.ent_coef,
        "value_coef": alg.inner_rl_agent.vf_coef,
        "max_grad_norm": alg.inner_rl_agent.max_grad_norm,
        "target_kl": alg.inner_rl_agent.target_kl,
        "anneal_lr": alg.inner_rl_agent.anneal_lr,
        "use_wandb": False,
        "agent_fn": PPOConditionedOnPolicyRepresentationAgent,
        "oracle_type": "ppo",
        "neupl_ppo_kwargs": {
            "num_policies": N,
            "policy_embedding_size": 64,
        },
        "device": args.device,
        "use_joint_obs_for_critic": True,
        "optimizer_type": alg.optimizer_str,
    }

    # Initialize agents – all policies for a player share one underlying network.
    base_agents = [
        agent_class(env, player_id, neupl_ppo_policy_index=0, **agent_kwargs)
        for player_id in range(2)
    ]
    agents = [
        [
            agent_class(
                env, player_id,
                network=base_agents[player_id]._policy.agent.network,
                neupl_ppo_policy_index=i,
                **agent_kwargs,
            )
            for i in range(N)
        ]
        for player_id in range(2)
    ]
    if apsro:
        E = 0 if apsro_exploited_player == 'p1' else 1   # exploited (uniform anchor at index 0)
        X = 1 - E                                  # exploiter (index 0 stays trainable BR)
        agents[E][0] = rl_policy.UniformRandomAgentPolicy(env, E, num_actions=num_actions)
    else:
        agents[0][0] = rl_policy.UniformRandomAgentPolicy(env, 0, num_actions=num_actions)
        agents[1][0] = rl_policy.UniformRandomAgentPolicy(env, 1, num_actions=num_actions)
    for player_agents in agents:
        for agent in player_agents:
            agent.unfreeze()

    # Oracle is used for its sample_episode method (handles PPO post-step bookkeeping).
    oracle = rl_oracle.RLOracle(
        env, agent_class, agent_kwargs,
        # number_training_episodes=number_training_episodes,
        number_training_episodes=None,
        self_play_proportion=0.0,
        mutate=True,
        sigma=0.0,
    )

    # Payoff table: index 0 = uniform random, indices 1..N-1 = trained policies.
    meta_games = np.full((2, N, N), np.nan)

    # Randall loss: optimizes embeddings so that e_i · e_j ≈ payoff[i,j].
    if use_randall_loss:
        import torch
        randall_optimizers = [
            torch.optim.Adam(
                agents[pid][1]._policy.agent.network.policy_representation_embedding.parameters(),
                lr=RANDALL_LR,
            )
            for pid in range(2)
        ]

    class _State:
        episodes_played = 0
        episodes_training = 0
        episodes_payoff = 0
        episodes_at_last_expl_check = 0
        stats = []  # list of dicts, one per exploitability measurement
        last_exploitability = None
        payoff_matrix_update_rate = None

    state = _State()
    stats_path = os.path.join(experiment_dir, "stats.jsonl")

    def _gt_payoff_pair(i, j):
        """Return exact expected payoffs for policy pair (i, j) via full tree traversal."""
        from open_spiel.python.algorithms import expected_game_score
        returns = expected_game_score.policy_value(
            game.new_initial_state(), [agents[0][i], agents[1][j]]
        )
        return np.array(returns)

    def _estimate_payoffs(i, j, n_sims):
        """Return average payoffs over n_sims rollouts for policy pair (i,j)."""
        agents[0][i].freeze()
        agents[1][j].freeze()
        totals = np.zeros(2)
        for _ in range(n_sims):
            totals += tabular_sample_episode(
                game.new_initial_state(), [agents[0][i], agents[1][j]]
            )
            state.episodes_played += 1
            state.episodes_payoff += 1
        agents[0][i].unfreeze()
        agents[1][j].unfreeze()
        return totals / n_sims

    def _update_payoffs(k, n_sims=sims_per_entry):
        """Update the (k+1)×(k+1) payoff subgame.

        New entries are initialised from scratch; existing entries are
        updated with an EMA to track policy changes over training.
        """
        for i in range(k + 1):
            for j in range(k + 1):
                if np.isnan(meta_games[0][i, j]):
                    est = _estimate_payoffs(i, j, 5000)
                    meta_games[0][i, j] = est[0]
                    meta_games[1][i, j] = est[1]
                else:
                    ema_alpha = 1 - pow(state.payoff_matrix_update_rate, n_sims)
                    est = _estimate_payoffs(i, j, n_sims)
                    meta_games[0][i, j] = (1 - ema_alpha) * meta_games[0][i, j] + ema_alpha * est[0]
                    meta_games[1][i, j] = (1 - ema_alpha) * meta_games[1][i, j] + ema_alpha * est[1]

    def _update_randall_loss(k):
        """Optimize embeddings so that e_i · e_j ≈ payoffs[i,j] for i,j in 1..k."""
        if not use_randall_loss:
            return
        import torch
        n = k + 1
        payoffs = torch.tensor(meta_games[0][1:n, 1:n], dtype=torch.float32)
        for _ in range(RANDALL_LOSS_EPOCHS):
            inner_products = torch.zeros_like(payoffs)
            for i in range(1, n):
                for j in range(1, n):
                    ei = agents[0][i]._policy.agent.network.policy_representation_embedding(torch.tensor(i))
                    ej = agents[1][j]._policy.agent.network.policy_representation_embedding(torch.tensor(j))
                    inner_products[i - 1, j - 1] = ei @ ej
            loss = torch.norm(payoffs - inner_products)
            randall_optimizers[0].zero_grad()
            randall_optimizers[1].zero_grad()
            loss.backward()
            randall_optimizers[0].step()
            randall_optimizers[1].step()

    def _compute_nash(K, payoff_matrix=None):
        """Return Nash strategies [nash_p0, nash_p1] for the (K+1)×(K+1) subgame.

        payoff_matrix: optional (2, K+1, K+1) array to use instead of meta_games.
        """
        source = payoff_matrix if payoff_matrix is not None else meta_games
        subgame = [
            np.nan_to_num(source[p][:K + 1, :K + 1], nan=0.0).tolist()
            for p in range(2)
        ]
        nash_p0, nash_p1, _, _ = lp_solver.solve_zero_sum_matrix_game(
            pyspiel.create_matrix_game(*subgame)
        )
        nash_p0 = np.array(nash_p0).reshape(-1)
        nash_p1 = np.array(nash_p1).reshape(-1)
        # if nash_p0.min() < 0 or nash_p1.min() < 0:
        #     print(f"  [debug] raw LP output has negatives — p0 min={nash_p0.min():.2e}, p1 min={nash_p1.min():.2e}; clipping.")
        nash_p0 = np.clip(nash_p0, 0.0, None)
        nash_p1 = np.clip(nash_p1, 0.0, None)
        nash_p0 = nash_p0 / max(nash_p0.sum(), 1e-12)
        nash_p1 = nash_p1 / max(nash_p1.sum(), 1e-12)
        return [nash_p0, nash_p1]

    def _compute_subgame_nashes(k):
        """Return a dict mapping pi -> Nash of the pi×pi subgame (indices 0..pi-1).

        pi=1..k  : policy pi trains as best response to Nash({0..pi-1})
        pi=k+1   : full (k+1)-policy Nash, used for display / exploitability

        If gt_payoffs is set, compute a fresh ground-truth payoff matrix for the
        full (k+1)×(k+1) subgame and use slices of it for each subgame nash.
        """
        if gt_payoffs:
            n = k + 1
            gt = np.zeros((2, n, n))
            for i in range(n):
                for j in range(n):
                    ret = _gt_payoff_pair(i, j)
                    gt[0][i, j] = ret[0]
                    gt[1][i, j] = ret[1]
            return {pi: _compute_nash(pi - 1, payoff_matrix=gt) for pi in range(1, k + 2)}
        return {pi: _compute_nash(pi - 1) for pi in range(1, k + 2)}

    def _force_learn(training_pol, player_id):
        """Flush the PPO rollout buffer: train if >= 2 samples, otherwise just clear.

        Also snaps total_steps_done to a multiple of steps_per_batch so that the
        PPOWrapper auto-learn trigger (total_steps_done % steps_per_batch == 0) never
        fires with a nearly-empty buffer after the flush.
        """
        ppo = training_pol._policy.agent
        if ppo.cur_batch_idx >= 2:
            terminal_ts = oracle._env.get_time_step()
            obs_dict = _copy.copy(terminal_ts.observations)
            obs_dict["current_player"] = player_id
            fixed_ts = RLTimeStep(
                observations=obs_dict,
                rewards=terminal_ts.rewards,
                discounts=terminal_ts.discounts,
                step_type=terminal_ts.step_type,
            )
            ppo.learn([fixed_ts])
        else:
            ppo.cur_batch_idx = 0
        # Snap total_steps_done to a multiple of steps_per_batch so the auto-learn
        # trigger fires only after a full buffer accumulates, not after 1 step.
        ppo.total_steps_done = (ppo.total_steps_done // ppo.steps_per_batch) * ppo.steps_per_batch

    def _train_player(player, k, subgame_nashes):
        """Train player's policies 1..k.

        Policy pi best-responds to the Nash of the pi×pi subgame (indices 0..pi-1),
        matching the NeuPL convention of one restricted game per best-responder.
        Samples num_pols_sampled policies uniformly from [1..k].  For each,
        collects total_episodes_per_policy episodes then explicitly flushes
        the PPO rollout buffer via _force_learn.
        """
        opponent = 1 - player
        for _ in range(num_pols_sampled):
            pi = min(random.randint(1, k+1), N-1)
            training_pol = agents[player][pi]
            training_pol.unfreeze()
            opponent_nash = subgame_nashes[pi][opponent]
            logger.info('Training player %s policy %s against opponent %s with nash %s', player, pi, opponent, opponent_nash)
            for _ in range(total_episodes_per_policy):
                pj = int(np.random.choice(len(opponent_nash), p=opponent_nash))
                opponent_pol = agents[opponent][pj]
                opponent_pol.freeze()
                episode_agents = (
                    [training_pol, opponent_pol] if player == 0
                    else [opponent_pol, training_pol]
                )
                oracle.sample_episode(None, episode_agents, is_evaluation=False)
                opponent_pol.freeze()
                state.episodes_played += 1
                state.episodes_training += 1

            _force_learn(training_pol, player)

    def _fmt(x):
        return f"{x:.4f}" if not np.isnan(x) else "nan"

    def _print_payoffs(k):
        n = min(k + 1, 5)
        logger.info("  meta_games[0] (top-left corner):")
        for row in meta_games[0][:n]:
            logger.info("    %s", "\t".join(_fmt(x) for x in row[:n]))

    def _compute_exploitability(k, nash):
        active_policies = [agents[0][:k + 1], agents[1][:k + 1]]
        aggregator = policy_aggregator.PolicyAggregator(game)
        aggr_policies = aggregator.aggregate(range(2), active_policies, nash)
        expl, expl_per_player = exploitability.nash_conv(
            game, aggr_policies, return_only_nash_conv=False
        )
        return expl / 2, list(expl_per_player)

    def _record_exploitability(k, nash):
        expl, expl_per_player = _compute_exploitability(k, nash)
        expl_str = "\t".join(_fmt(v) for v in expl_per_player)
        logger.info("  Exploitability: %s  per player: %s  (at %s episodes)", _fmt(expl), expl_str, state.episodes_played)
        state.episodes_at_last_expl_check = state.episodes_played
        state.last_exploitability = expl

        record = {
            "episodes": state.episodes_played,
            "episodes_training": state.episodes_training,
            "episodes_payoff": state.episodes_payoff,
            "walltime": time.perf_counter() - run_start,
            "exploitability": expl,
            "exploitability_per_player": expl_per_player,
        }
        state.stats.append(record)
        with open(stats_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        _plot_stats(state.stats, experiment_dir, logger=logger)

    def _print_exploitability(k, nash):
        if state.episodes_played - state.episodes_at_last_expl_check < expl_check_episode_interval:
            return
        _record_exploitability(k, nash)

    def _debug_check(k, subgame_nashes):
        """Compare ground-truth payoffs against the payoff matrix, and print
        per-policy payoff vs aggregate opponent alongside the best-response payoff."""
        if not debug:
            return
        if state.episodes_played - state.episodes_at_last_expl_check < expl_check_episode_interval:
            return

        from open_spiel.python.algorithms import expected_game_score
        from open_spiel.python.policy import tabular_policy_from_callable, python_policy_to_pyspiel_policy

        def _to_pyspiel_policy(functional_policy):
            tabular = tabular_policy_from_callable(game, functional_policy)
            return python_policy_to_pyspiel_policy(tabular)

        n = k + 1
        logger.debug("  Ground-truth vs stored payoffs (k=%s):", k)

        # Collect all values first.
        emp = np.array([[meta_games[0][i, j] for j in range(n)] for i in range(n)])
        gt_mat = np.array([[_gt_payoff_pair(i, j)[0] for j in range(n)] for i in range(n)])
        diff = emp - gt_mat

        def _fmt3(x):
            return f"{x:.3f}" if not np.isnan(x) else "nan"

        # Format each cell as 3 stacked lines: empirical / gt / diff.
        cells = [[[_fmt3(emp[i,j]), _fmt3(gt_mat[i,j]), _fmt3(diff[i,j])]
                  for j in range(n)] for i in range(n)]
        col_w    = max(len(s) for row in cells for cell in row for s in cell)
        row_lbl_w = len(f"  i={n-1} ")          # e.g. "  i=9 "
        line_lbl_w = 3                            # "emp" / " gt" / "dif"
        sep_w    = row_lbl_w + line_lbl_w + 3    # + " | "
        sep      = " " * sep_w + ("--".join("-" * col_w for _ in range(n)))
        hdr      = " " * sep_w + "  ".join(f"j={j}".center(col_w) for j in range(n))
        logger.debug(hdr)
        logger.debug(sep)
        for i, row in enumerate(cells):
            for line_idx, line_label in enumerate(["emp", " gt", "dif"]):
                row_lbl = f"  i={i} ".ljust(row_lbl_w) if line_idx == 1 else " " * row_lbl_w
                prefix  = row_lbl + line_label + " | "
                logger.debug(prefix + "  ".join(cell[line_idx].rjust(col_w) for cell in row))
            logger.debug(sep)

        max_err = float(np.nanmax(np.abs(diff)))
        logger.debug("  max payoff err: %s", _fmt3(max_err))

        if use_randall_loss:
            import torch
            logger.debug("  Randall loss per cell (payoff - e_i·e_j) for i,j in 1..%s:", k)
            rl_cells = [[None] * k for _ in range(k)]
            for i in range(1, n):
                for j in range(1, n):
                    ei = agents[0][i]._policy.agent.network.policy_representation_embedding(torch.tensor(i))
                    ej = agents[1][j]._policy.agent.network.policy_representation_embedding(torch.tensor(j))
                    inner = (ei @ ej).item()
                    rl_cells[i - 1][j - 1] = _fmt3(meta_games[0][i, j] - inner)
            rl_col_w = max(len(s) for row in rl_cells for s in row)
            rl_hdr = " " * sep_w + "  ".join(f"j={j+1}".center(rl_col_w) for j in range(k))
            rl_sep = " " * sep_w + "--".join("-" * rl_col_w for _ in range(k))
            logger.debug(rl_hdr)
            logger.debug(rl_sep)
            for i, row in enumerate(rl_cells):
                row_lbl = f"  i={i+1} ".ljust(row_lbl_w)
                logger.debug(row_lbl + "    | " + "  ".join(s.rjust(rl_col_w) for s in row))
            logger.debug(rl_sep)

        logger.debug("  Per-policy payoff vs aggregate opponent:")
        for player in range(2):
            opponent = 1 - player
            for pi in range(1, min(N, k + 2)):
                opp_nash = subgame_nashes[pi][opponent]
                label = "full" if pi == k + 1 else f"pi={pi}"
                dist_str = "  ".join(f"p{j}={v:.2f}" for j, v in enumerate(opp_nash))

                # Build joint policy: player plays pi pure, opponent plays opp_nash mixture.
                n_opp = len(opp_nash)
                pol = [None, None]
                wts = [None, None]
                pol[player]   = [agents[player][pi]]
                wts[player]   = np.array([1.0])
                pol[opponent] = list(agents[opponent][:n_opp])
                wts[opponent] = np.array(opp_nash)
                agg = policy_aggregator.PolicyAggregator(game)
                joint = agg.aggregate(range(2), pol, wts)

                joint_pyspiel = _to_pyspiel_policy(joint)
                current_payoff = expected_game_score.policy_value(
                    game.new_initial_state(), [agents[player][pi], joint_pyspiel]
                )[player]
                _, expl_per_player = exploitability.nash_conv(
                    game, joint_pyspiel, return_only_nash_conv=False
                )
                br_payoff = current_payoff + expl_per_player[player]

                logger.debug("    p%s [%s]  opp=[%s]  payoff=%s  br_payoff=%s diff=%s",
                             player, label, dist_str, _fmt(current_payoff),
                             _fmt(br_payoff), _fmt(abs(current_payoff - br_payoff)))

    def _save_checkpoint():
        if not save_checkpoints:
            return
        # All policies share one network per player; overwrite a single file each time.
        for pid in range(2):
            net = agents[pid][1]._policy.agent.network
            net.save(os.path.join(experiment_dir, f'policy{pid}_ckpt.pt'))
        np.save(os.path.join(experiment_dir, 'meta_game.npy'), meta_games)

    if T is None:
        T = N * 15
    # num_iterations is a function param (default 680): run until k reaches N.

    # ── timing helpers ────────────────────────────────────────────────────────

    def _fmt_dur(secs):
        """Format a duration: '  0.03s', ' 12.34s', '  1m23s', ' 1h02m'."""
        if secs < 60:
            return f"{secs:6.2f}s"
        elif secs < 3600:
            m, s = divmod(int(secs), 60)
            return f"{m:3d}m{s:02d}s"
        else:
            h, rem = divmod(int(secs), 3600)
            m = rem // 60
            return f"{h:3d}h{m:02d}m"

    def _fmt_elapsed(secs):
        """Format elapsed time as HH:MM:SS."""
        h, rem = divmod(int(secs), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _print_iter_summary(i, k, lr, timings, iter_secs, run_secs):
        total = sum(dt for _, dt in timings)
        bar_width = 20

        col_name  = max(len(name) for name, _ in timings + [("TOTAL", 0)])
        col_name  = max(col_name, 24)

        header = f"  iteration i={i}/{num_iterations} ({T=})  k={k}/{N-1}  lr={lr:.2e}  total: {_fmt_dur(iter_secs).strip()}"
        width = col_name + 2 + 8 + 3 + 5 + 3 + bar_width + 2  # rough total width

        sep_thin = "  " + "─" * (col_name + 2) + "┼" + "─" * 9 + "┼" + "─" * 6 + "┼" + "─" * (bar_width + 2)
        sep_thick= "  " + "─" * (col_name + 2) + "┼" + "─" * 9 + "┼" + "─" * 6 + "┼" + "─" * (bar_width + 2)
        top      = "  ┌" + "─" * (col_name + 2) + "┬" + "─" * 9 + "┬" + "─" * 6 + "┬" + "─" * (bar_width + 2) + "┐"
        mid      = "  ├" + "─" * (col_name + 2) + "┼" + "─" * 9 + "┼" + "─" * 6 + "┼" + "─" * (bar_width + 2) + "┤"
        bot      = "  └" + "─" * (col_name + 2) + "┴" + "─" * 9 + "┴" + "─" * 6 + "┴" + "─" * (bar_width + 2) + "┘"

        def row(name, secs, pct, is_total=False):
            bar_fill = round(pct / 100 * bar_width)
            bar = "█" * bar_fill + "░" * (bar_width - bar_fill)
            marker = "Σ" if is_total else " "
            return (f"  │ {marker}{name:<{col_name}} │ {_fmt_dur(secs)} │ {pct:4.1f}% │ {bar} │")

        logger.info("\n%s", top)
        logger.info(f"  │ {header:<{col_name + 1 + 9 + 1 + 6 + 1 + bar_width + 1}} │")
        logger.info(mid)
        logger.info(f"  │  {'Step':<{col_name}} │ {'  Time':>8} │ {'  Pct':>5} │ {'Bar':<{bar_width}} │")
        logger.info(mid)
        for name, dt in timings:
            pct = dt / total * 100 if total > 0 else 0.0
            logger.info(row(name, dt, pct))
        logger.info(mid)
        logger.info(row("TOTAL", total, 100.0, is_total=True))
        logger.info(bot)

        # elapsed + ETA
        iters_done = i
        secs_per_iter = run_secs / iters_done
        eta = secs_per_iter * (num_iterations - iters_done)
        logger.info("  Elapsed: %s  │  ETA: %s  │  %s/%s iters  │  %s/iter avg",
                    _fmt_elapsed(run_secs), _fmt_elapsed(eta), iters_done,
                    num_iterations, _fmt_dur(secs_per_iter).strip())

    # ── main loop ─────────────────────────────────────────────────────────────

    base_lr = alg.inner_rl_agent.learning_rate
    final_lr = alg.inner_rl_agent.lr_final
    lr_anneal_iters = alg.lr_anneal_iters

    def _set_lr(lr):
        """Update optimizer LR for both players.

        Each policy index 1..N-1 has its own PPO instance with its own optimizer
        (they only share the underlying network), so the LR must be set on every
        one of them individually -- setting it on just one index leaves the rest
        training at their original constant LR forever.
        """
        for player in range(2):
            for pi in range(1, N):
                opt = agents[player][pi]._policy.agent.optimizer
                for pg in opt.param_groups:
                    pg['lr'] = lr

    def _set_player_lr(player, lr, start=1):
        """Set LR for one player's trainable policies (indices start..N-1)."""
        for pi in range(start, N):
            opt = agents[player][pi]._policy.agent.optimizer
            for pg in opt.param_groups:
                pg['lr'] = lr

    def _set_payoff_matrix_update_rate(update_rate):
        state.payoff_matrix_update_rate = update_rate

    if apsro:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from bandits import make_bandit

        reward_range = (game.min_utility(), game.max_utility())
        # One persistent bandit per exploited policy index i, over arms {0..i-1}.
        bandits = {i: make_bandit(apsro_bandit, num_arms=i, reward_range=reward_range, seed=i)
                   for i in range(1, N)}

        # Write config.json so the checkpoint loader sizes the embedding table correctly
        # (load_ppo_agents_from_neupl reads num_policies from it; default 100 would mismatch).
        apsro_config = OmegaConf.to_container(alg, resolve=True)
        apsro_config['num_policies'] = N
        apsro_config['use_randall_loss'] = use_randall_loss  # so select_neupl_directory surfaces it
        apsro_config['apsro'] = True
        apsro_config['exploited_player'] = apsro_exploited_player
        apsro_config['bandit'] = apsro_bandit
        if save_checkpoints:  # config.json is what select_neupl_directory lists; skip in tests
            with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
                json.dump(apsro_config, f)

        def _episode_agents(train_player, train_pol, opp_pol):
            """Order agents by player id for oracle.sample_episode."""
            pair = [None, None]
            pair[train_player] = train_pol
            pair[1 - train_player] = opp_pol
            return pair

        def _train_exploited(k):
            """Train E's policies 1..k against their bandit mixture over X's {0..i-1}."""
            _sampled = []
            for _ in range(num_pols_sampled):
                i = random.randint(1, k)  # k is the top unlocked index (randint is inclusive)
                _sampled.append(i)
                training_pol = agents[E][i]
                training_pol.unfreeze()
                b = bandits[i]
                for _ in range(total_episodes_per_policy):
                    j = b.sample()
                    opp = agents[X][j]
                    opp.freeze()
                    rewards = oracle.sample_episode(
                        None, _episode_agents(E, training_pol, opp), is_evaluation=False)
                    b.update(j, float(rewards[X]))   # reward = exploiter's payoff
                    state.episodes_played += 1
                    state.episodes_training += 1
                _force_learn(training_pol, E)
            if debug:
                logger.debug("  [apsro] exploited (E=p%s) policies sampled this iter: %s",
                             E + 1, _sampled)

        def _train_exploiter(k):
            """Train X's policies 0..k as a pure best response to E's same-index policy."""
            _sampled = []
            for _ in range(num_pols_sampled):
                i = random.randint(0, k)  # k is the top unlocked index (randint is inclusive)
                _sampled.append(i)
                training_pol = agents[X][i]
                training_pol.unfreeze()
                opp = agents[E][i]
                opp.freeze()
                for _ in range(total_episodes_per_policy):
                    oracle.sample_episode(
                        None, _episode_agents(X, training_pol, opp), is_evaluation=False)
                    state.episodes_played += 1
                    state.episodes_training += 1
                _force_learn(training_pol, X)
            if debug:
                logger.debug("  [apsro] exploiter (X=p%s) policies sampled this iter: %s",
                             X + 1, _sampled)

        def _plot_brv(records):
            if not records:
                return
            xs = [r["episodes"] for r in records]
            n_pol = max(len(r["brv_per_policy"]) for r in records)
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = plt.get_cmap("viridis")
            for idx in range(n_pol):
                ys = [(r["brv_per_policy"][idx] if idx < len(r["brv_per_policy"]) else np.nan)
                      for r in records]
                ax.plot(xs, ys, color=cmap(idx / max(n_pol - 1, 1)), lw=1, label=f"E_{idx}")
            ax.set_xlabel("episodes"); ax.set_ylabel("BRV of exploited policy")
            ax.set_title(f"APSRO BRV per exploited policy (E=p{E+1})")
            ax.grid(True, alpha=0.3)
            if n_pol <= 12:
                ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(experiment_dir, "brv.png"), dpi=120)
            plt.close(fig)

        def _measure_brv(k):
            if state.episodes_played - state.episodes_at_last_expl_check < expl_check_episode_interval:
                return
            for pid in range(2):
                for i in range(k + 1):
                    agents[pid][i].freeze()
            brvs = [best_response_value(game, agents[E][i], fixed_player=E, br_player=X)
                    for i in range(k + 1)]
            state.episodes_at_last_expl_check = state.episodes_played
            record = {
                "episodes": state.episodes_played,
                "episodes_training": state.episodes_training,
                "walltime": time.perf_counter() - run_start,
                "brv_per_policy": brvs,
                "mean_brv": float(np.mean(brvs)),
            }
            state.stats.append(record)
            with open(stats_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info("  BRV per exploited policy (k=%s): %s",
                        k, "  ".join(f"E_{i}={v:.4f}" for i, v in enumerate(brvs)))
            if debug:
                _debug_apsro(k)
            _plot_brv(state.stats)

        def _debug_apsro(k):
            """Per exploited policy i (1..k): E's exact payoff vs each opponent X_j, that
            opponent's current bandit probability, and E_i's EV against the mixture."""
            from open_spiel.python.algorithms import expected_game_score

            def _payoff_E_vs_X(i, j):
                pair = [None, None]
                pair[E], pair[X] = agents[E][i], agents[X][j]
                return expected_game_score.policy_value(game.new_initial_state(), pair)[E]

            logger.debug("  [apsro debug] E payoff vs each exploiter (payoff | bandit %%):")
            for i in range(1, k + 1):
                dist = bandits[i].distribution()
                cells = []
                ev = 0.0
                for j in range(i):
                    payoff = _payoff_E_vs_X(i, j)
                    ev += dist[j] * payoff
                    cells.append(f"X_{j}: {payoff:+.4f} {dist[j] * 100:5.1f}%")
                logger.debug("    E_%s  %s  |  EV(vs pop)=%+.4f",
                             i, "   ".join(cells), ev)

        run_start = time.perf_counter()
        for it in range(1, num_iterations + 1):
            k = min(N - 1, max(1, int(np.ceil(it / T * (N - 1)))))
            t = min(it - 1, lr_anneal_iters - 1) / max(lr_anneal_iters - 1, 1)
            lr = base_lr + (final_lr - base_lr) * t
            # Exploited (E) can train at a scaled (typically smaller) LR; E_0 is uniform (no
            # optimizer), so start at 1. X_0 is a trainable BR, so include index 0 for X.
            _set_player_lr(E, lr * apsro_exploited_lr_scale, start=1)
            _set_player_lr(X, lr, start=0)
            logger.info("  [apsro] iteration %s/%s  k=%s/%s  lr=%.2e (E x%.3g)",
                        it, num_iterations, k, N - 1, lr, apsro_exploited_lr_scale)
            _train_exploited(k)
            _train_exploiter(k)
            _measure_brv(k)
            _save_checkpoint()
        return

    run_start = time.perf_counter()
    nash = None  # initialised on first iteration; carried over thereafter

    for i in range(1, num_iterations + 1):
        # k is the number of active trained policies; total active = k+1 (including index 0).
        # Quadratic schedule: k = ceil((i/T)^2 * (N-1)), clamped to [1, N-1].
        # k = min(N - 1, max(1, int(np.ceil((i / T) ** 2 * (N - 1)))))
        # linear schedule:
        k = min(N-1, max(1, int(np.ceil(i/T * (N-1)))))
        t = min(i - 1, lr_anneal_iters - 1) / max(lr_anneal_iters - 1, 1)
        lr = base_lr + (final_lr - base_lr) * t
        _set_lr(lr)
        payoff_matrix_update_rate = min(0.9998, 0.9995 + (i-1)/lr_anneal_iters * (0.9998 - 0.9995))
        logger.info('payoff_matrix_update_rate: %s', payoff_matrix_update_rate)
        _set_payoff_matrix_update_rate(payoff_matrix_update_rate)
        iter_start = time.perf_counter()
        timings = []

        def _timed(label, fn, *args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            timings.append((label, time.perf_counter() - t0))
            return result

        _timed("payoffs",               _update_payoffs, k)
        _timed("randall loss",          _update_randall_loss, k)
        nashes = _timed("nashes",       _compute_subgame_nashes, k)
        logger.info("  Training Player 0 (%s episodes)...", total_episodes_per_policy * num_pols_sampled)
        _timed("train player 0",        _train_player, 0, k, nashes)

        _timed("payoffs",               _update_payoffs, k)
        _timed("randall loss",          _update_randall_loss, k)
        nashes = _timed("nashes",       _compute_subgame_nashes, k)
        logger.info("  Training Player 1 (%s episodes)...", total_episodes_per_policy * num_pols_sampled)
        _timed("train player 1",        _train_player, 1, k, nashes)

        nash = nashes[k + 1]  # full (k+1)-policy Nash for display / exploitability
        nash_str = [" ".join(_fmt(v) for v in n) for n in nash]
        logger.info("  Nash P0: [%s]  Nash P1: [%s]", nash_str[0], nash_str[1])
        _timed("print payoffs",      _print_payoffs, k)
        _timed("debug check",        _debug_check, k, nashes)
        _timed("exploitability",     _print_exploitability, k, nash)
        _timed("checkpoint",             _save_checkpoint)

        config_data = OmegaConf.to_container(alg, resolve=True)
        config_data['num_policies'] = N
        config_data['use_randall_loss'] = use_randall_loss
        config_data['gt_payoffs'] = gt_payoffs
        config_data['save_logs'] = save_logs
        config_data['randall_lr'] = RANDALL_LR
        config_data['randall_loss_epochs'] = RANDALL_LOSS_EPOCHS
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            json.dump(config_data, f)

        iter_secs = time.perf_counter() - iter_start
        run_secs  = time.perf_counter() - run_start
        _print_iter_summary(i, k, lr, timings, iter_secs, run_secs)

    # Final exploitability (unconditional — may repeat the last throttled check).
    logger.info("\n  [Final exploitability check]")
    _record_exploitability(k, nash)

    # Persist final exploitability into config.json so the directory picker can show it.
    config_path = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_data = json.load(f)
    else:
        config_data = {}
    config_data['final_exploitability'] = state.last_exploitability
    with open(config_path, 'w') as f:
        json.dump(config_data, f)

    _plot_stats(state.stats, experiment_dir, logger=logger)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    return experiment_dir


def _plot_stats(stats, experiment_dir, logger=None):
    if not stats:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes  = [r["episodes"]        for r in stats]
    walltime  = [r["walltime"]         for r in stats]
    expl      = [r["exploitability"]   for r in stats]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(episodes, expl, marker="o", markersize=3)
    axes[0].set_xlabel("Episodes played")
    axes[0].set_ylabel("Exploitability")
    axes[0].set_title("Exploitability vs Episodes")
    axes[0].grid(True)

    def _fmt_walltime(s):
        m = int(s) // 60
        return f"{m}m" if m > 0 else f"{s:.1f}s"

    axes[1].plot(walltime, expl, marker="o", markersize=3)
    axes[1].set_xlabel("Wall time")
    axes[1].set_ylabel("Exploitability")
    axes[1].set_title("Exploitability vs Wall time")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda s, _: _fmt_walltime(s)))
    axes[1].grid(True)

    fig.tight_layout()
    out_path = os.path.join(experiment_dir, "exploitability.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("  Saved exploitability plot → %s", out_path)
    else:
        print(f"  Saved exploitability plot → {out_path}")


def load_ppo_agents_from_psro(
        game_short_name: str = 'kuhn_poker',
        player_id: Union[int, None] = None,
        hidden_size: int = 512,
        shuffle: bool = True,
        device: str = 'cpu',
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
                agent = PPOAgent(num_actions, info_state_size, device, hidden_size=hidden_size)
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
        num_directories: int = 15,
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

    recent = filtered[:num_directories]
    print(f"\nNEUPL directories for use_randall_loss={use_randall_loss}:")
    dir_info = []
    for idx, subdir in enumerate(recent):
        pt_files = list(subdir.glob("*.pt"))
        dir_info.append((subdir, len(pt_files)))
        # Read optional summary fields from config.json (already parsed into filtered list above).
        cfg = {}
        config_path = subdir / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
            except Exception:
                pass
        num_pol = cfg.get("num_policies")
        final_expl = cfg.get("final_exploitability")
        extras = []
        if num_pol is not None:
            extras.append(f"num_policies={num_pol}")
        if cfg.get("apsro"):
            extras.append(f"apsro(exploited={cfg.get('exploited_player')},bandit={cfg.get('bandit')})")
        if final_expl is not None:
            extras.append(f"final_expl={final_expl:.4f}")
        extras_str = ("  " + "  ".join(extras)) if extras else ""
        print(f"  [{idx}]: {subdir.name}{extras_str}")

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
        device: str = 'cpu',
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
        num_actions, info_state_size, device,
        num_policies=num_policies, policy_embedding_size=policy_embedding_size, hidden_size=hidden_size
    )
    agent1 = PPOConditionedOnPolicyRepresentationAgent(
        num_actions, info_state_size, device,
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
    agent_device = next(agents[0].parameters()).device

    for player_id in range(2):
        player_policies = []

        if sampling_mode == "gaussian":
            # Collect all existing embeddings
            existing_embeddings = []
            for policy_idx in range(1, original_num_policies):  # Skip 0 (uniform random)
                policy_index_tensor = torch.tensor(policy_idx, device=agent_device)
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
                embedding_tensor = torch.tensor(sampled_embedding, dtype=torch.float32, device=agent_device)

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
                policy_index_1 = torch.tensor(random.randint(1, original_num_policies - 1), device=agent_device)
                policy_index_2 = torch.tensor(random.randint(1, original_num_policies - 1), device=agent_device)
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

def neupl_original_embeddings(agent):
    """Normed embeddings of the original NeuPL policies (indices 1..num_policies-1).

    These are the trained anchors the sampling Gaussian was fit to (index 0 = uniform
    random is skipped). Returned as an (N-1, embedding_dim) numpy array, in the same
    (post-norm) space as the sampled pool. Best-effort: returns None on any failure.
    """
    try:
        device = next(agent.parameters()).device
        embs = [agent.policy_representation_embedding(torch.tensor(i, device=device)).detach().cpu().numpy()
                for i in range(1, agent.num_policies)]
        return np.array(embs)
    except Exception:
        return None


def make_neupl_policies(
    game_short_name: str,
    neupl_config: dict,
    num_policies_to_make: int = 1000,
    directory: Optional[str] = None,
    interpolate_prenorm: bool = True,
    sampling_mode: Literal["interpolate", "gaussian"] = "interpolate",
    device: str = 'cpu',
):
    agents = load_ppo_agents_from_neupl(game_short_name=game_short_name, **neupl_config, dir_name=directory, device=device)
    # Read the embedding table size directly from the loaded agents so it always matches.
    original_num_policies = agents[0].num_policies
    return make_ppo_policies_from_neupl_agents(
        game_short_name,
        agents,
        original_num_policies=original_num_policies,
        num_policies_to_make=num_policies_to_make,
        interpolate_prenorm=interpolate_prenorm,
        sampling_mode=sampling_mode,
    )


def neupl_decoder(game, agent, embedding, player_id: int, use_observation: bool = False):
    """Embedding -> policy for NeuPL: condition `agent` on `embedding`.

    `agent` is a loaded PPOConditionedOnPolicyRepresentationAgent (e.g. obtained from an
    existing policy via `policy._ppo_agent`). `embedding` may be shape (D,) or (1, D);
    NeuPL conditioning expects (1, D).
    """
    from utils import PPONeuplAgentPolicy
    emb = embedding if isinstance(embedding, torch.Tensor) else torch.tensor(embedding, dtype=torch.float32)
    emb = emb.float()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return PPONeuplAgentPolicy(game, agent, player_id,
                               use_observation=use_observation, embedding=emb)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--neupl', action='store_true', help='Run neupl instead of standard psro')
    parser.add_argument('--neupl_v2', action='store_true', help='Run custom neupl_v2 loop')
    parser.add_argument('--T', type=int, default=None, help='Iteration cap for neupl_v2 k-schedule (default: N)')
    parser.add_argument('--use_randall_loss', action='store_true', help='Use randall loss instead of standard loss')
    parser.add_argument('--game_name', type=str, default='kuhn_poker', help='Game to run')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode for neupl_v2')
    parser.add_argument('--save_logs', action='store_true', help='Save NeuPL v2 logs under logs/neupl/')
    parser.add_argument('--gt_payoffs', action='store_true', help='Use ground-truth payoffs for Nash computation instead of the EMA payoff table')
    parser.add_argument('--apsro', action='store_true', help='Asymmetric APSRO in neupl_v2')
    parser.add_argument('--apsro_exploited_player', choices=['p1', 'p2'], default='p1',
                        help='Which player APSRO optimizes/measures')
    parser.add_argument('--apsro_bandit', choices=['hedge', 'rm'], default='hedge',
                        help='Adversarial bandit for the exploited player')
    parser.add_argument('--apsro_exploited_lr_scale', type=float, default=1.0,
                        help='Multiplier on the exploited player PPO learning rate (apsro)')
    args = parser.parse_args()
    if args.use_randall_loss:
        assert args.neupl or args.neupl_v2, "Use randall loss only with neupl"

    game_name = args.game_name
    if args.neupl_v2:
        run_neupl_v2(game_name, use_randall_loss=args.use_randall_loss, T=args.T,
                     debug=args.debug, gt_payoffs=args.gt_payoffs, save_logs=args.save_logs,
                     apsro=args.apsro, apsro_exploited_player=args.apsro_exploited_player,
                     apsro_bandit=args.apsro_bandit,
                     apsro_exploited_lr_scale=args.apsro_exploited_lr_scale)
    elif args.neupl:
        run_neupl(game_name, use_randall_loss=args.use_randall_loss)
    else:
        run_psro(game_name)
