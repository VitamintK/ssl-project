# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research project for self-supervised learning on multi-agent RL policies. The project trains policy encoders using different approaches (weight autoencoders, functional autoencoders, trajectory transformers) and evaluates them on downstream prediction tasks in imperfect-information games (Kuhn Poker, Leduc Poker).

## Common Commands

### Running Experiments

The main experiment runner is [main.py](main.py). It contains hardcoded experiment configurations in the `__main__` block that control which tasks and agent sources to run:

```bash
# Run all experiments (controlled by RUN_TASK_* and RUN_* flags in main.py __main__ block)
python main.py
```

The script runs downstream tasks (Task A, B, C, D) using different policy sources:
- **NEUPL**: Policies from Neural Policy Laundering with interpolated embeddings
- **PSRO**: Policies from Policy Space Response Oracles
- **Random**: Randomly initialized PPO agents

Results are saved to:
- `results/all_downstream_tasks.json` (main results file)
- `results/archive/downstream_results_<timestamp>.json` (timestamped backup)

### Running PSRO/NEUPL Training

To generate new policy checkpoints for experiments:

```bash
# Run PSRO training (generates policies in results/test/psro/...)
python -c "from psro import run_psro; run_psro('kuhn_poker')"
python -c "from psro import run_psro; run_psro('leduc_poker')"

# Run NEUPL training (generates policies with embeddings in results/test/neupl/...)
python -c "from psro import run_neupl; run_neupl('kuhn_poker', use_randall_loss=False)"
python -c "from psro import run_neupl; run_neupl('kuhn_poker', use_randall_loss=True)"
```

### Testing Specific Encoders

Individual encoder test scripts are available:

```bash
python test_weight_encoder.py          # Test weight autoencoder
python test_functional_encoder.py      # Test functional autoencoder
python test_trajectory_encoder.py      # Test trajectory transformer
python test_identity_encoder.py        # Test identity (raw weights) baseline
```

### Visualization

```bash
python visualize_embeddings.py         # t-SNE visualization of policy embeddings
python visualize_downstream.py         # Downstream task result analysis
python analyze_downstream_results.py   # Analyze results from all_downstream_tasks.json
```

## Architecture

### Core Components

**Policy Encoders** ([main.py](main.py:164-299)):
1. **Weight Autoencoder** ([weight_autoencoder.py](weight_autoencoder.py)): Compresses PPO agent weights into fixed-size embeddings
2. **Functional Autoencoder** ([functional_autoencoder.py](functional_autoencoder.py)): Encodes policies based on their action probability distributions across game states
3. **Trajectory Transformer** ([trajectory_encoder.py](trajectory_encoder.py)): Uses transformer to encode behavioral trajectories from policy rollouts
4. **Identity**: Baseline that uses raw flattened policy weights

**Downstream Tasks** ([downstream.py](downstream.py)):
- **Task A**: Predict policy payoff vs uniform random opponent
- **Task B**: Predict pairwise payoffs between policies
- **Task C**: Predict state-conditional payoffs (given game state)
- **Task D**: Predict exploitability (Nash gap)

**Policy Generation** ([psro.py](psro.py)):
- Loads and manages PPO policies from PSRO/NEUPL training runs
- NEUPL policies support embedding interpolation and Gaussian sampling in policy space
- Functions: `load_ppo_agents_from_psro()`, `make_neupl_policies()`

### Key Architectural Patterns

**Policy Representation Flow**:
```
PPOAgent → Encoder → Embedding → Downstream Predictor → Target Value
```

**NEUPL Policy Interpolation**:
NEUPL trains a policy conditioned on embeddings. During evaluation, new policies are created by:
1. Interpolating between existing policy embeddings (or sampling from Gaussian)
2. Passing interpolated embeddings to the conditioned policy network
3. This creates a continuous policy space for evaluation

**Functional Encoder Design**:
Unlike weight encoders, the functional encoder:
1. Collects (agent, state) → action_distribution pairs across all game states
2. Trains encoder to predict action distributions from encoded weights
3. Learns representations based on behavioral similarity, not weight similarity

**Data Loading Pattern**:
- PSRO/NEUPL checkpoints stored in: `results/test/{psro,neupl}/ppo/hs{hidden_size}/{game}/{timestamp}/`
- Each run directory contains: `policy{player_id}_ckpt{iter}.pt` files
- Load via `load_ppo_agents_from_psro()` which searches directory patterns

### Configuration System

Uses Hydra/OmegaConf YAML configs in [configs/](configs/):
- [experiment.yaml](configs/experiment.yaml): Top-level experiment settings
- [neupl.yaml](configs/neupl.yaml): NEUPL algorithm hyperparameters
- [psro_liars_dice_ppo.yaml](configs/psro_liars_dice_ppo.yaml): PSRO+PPO config examples

However, most experimental configuration is hardcoded in the `__main__` blocks of scripts rather than pulled from config files.

### Dependencies

Key external libraries:
- **open_spiel**: Provides imperfect-information game environments (imported as `pyspiel`)
- **iig_rl_benchmark**: Custom RL algorithms package (installed in editable mode from external repo)
  - Contains `PPOAgent`, `PPOConditionedOnPolicyRepresentationAgent`, and PSRO implementations
- **torch**: Neural network training
- **sklearn**: Random Forest regressor for some downstream tasks

### Device Management

Device selection via [utils.py](utils.py:11-17) `get_device_string()`:
- Tries CUDA → MPS → CPU in that order
- Most scripts default to `device='cpu'` in configs despite GPU availability

### Results Organization

```
results/
├── test/                              # Experiment results root
│   ├── psro/ppo/hs{size}/{game}/     # PSRO policy checkpoints
│   └── neupl/ppo/hs{size}/{game}/    # NEUPL policy checkpoints
├── all_downstream_tasks.json         # Main results file
└── archive/                           # Timestamped result backups
```

Each downstream task run appends to `all_downstream_tasks.json` with structure:
```json
{
  "run": [
    {
      "experiment_label": "...",
      "config": {...},
      "results": [{"mse": ..., "baseline_mse": ...}, ...]
    }
  ]
}
```

### Important Implementation Details

**PPOAgentPolicy Wrapper** ([utils.py](utils.py:19-52)):
Wraps `PPOAgent` to implement OpenSpiel's `Policy` interface for use in game rollouts. Critical for converting between RL agent API and game simulation API.

**Experiment Result Registration** ([main.py](main.py:34-72)):
Use `register_result(experiment_label, config_dict, mse, baseline_mse)` to log experiment results. Must call `save_results()` at end of script to persist to disk.

**Logging Pattern**:
Most scripts set up dual logging to both file and console:
```python
logger = logging.getLogger(__name__)
handler = logging.FileHandler('logs/script_name.log', mode='w')
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())
```

**Random Seeds**:
- Set via `set_seed(seed)` in [downstream.py](downstream.py:35-39)
- Covers random, numpy, torch, and CUDA
- Main experiments loop over `seed_num in range(3)` for multiple trials

**NEUPL Sampling Modes**:
When creating interpolated NEUPL policies, two modes available:
1. `interpolate_prenorm=True/False`: Whether to normalize embeddings before/after interpolation
2. `sampling_mode="linear"/"gaussian"`: Linear interpolation between pairs vs. sampling from Gaussian distribution fit to embeddings
