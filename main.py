import random
from typing import Literal, Optional
import pyspiel
import torch
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm

from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from psro import load_ppo_agents_from_psro
from utils import get_device_string, make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, StatePayoffPredictor, set_seed, sample_random_states
from weight_autoencoder import (
    AutoencoderConfig,
    WeightAutoencoder,
    ppo_agent_to_vector,
    VectorDataset,
    train_autoencoder,
    get_encoder,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
handler = logging.FileHandler('logs/all_downstream_tasks.log', mode='w')
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

def test_downstream_task_a(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"],
        encoder_type: Literal["identity", "weight_autoencoder"],
        autoencoder_ppo_agents: Optional[list[PPOAgent]] = None,
        downstream_task_ppo_agents: Optional[list[PPOAgent]] = None,
        experiment_label: str = "downstream_a",
        device: str = "cpu",
):
    """
    Test the PayoffPredictor on Kuhn Poker with the specified configuration.

    Args:
        game: The OpenSpiel game to use
        predictor_type: Type of predictor ("mlp" or "linear")
        encoder_type: Type of encoder ("identity" or "weight_autoencoder")
        experiment_label: Label for the experiment subdirectory (default: "downstream_a")
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    if encoder_type == 'weight_autoencoder':
        # Train autoencoder on agent weights
        NUM_AGENTS_AUTOENCODE = 1000
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for i in range(NUM_AGENTS_AUTOENCODE)]
        print("\nTraining autoencoder on agent weights...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            device=device,
        )
        weight_autoencoder = WeightAutoencoder(ae_config, autoencoder_ppo_agents, ppo_agent_to_vector)
        _, ae_history = weight_autoencoder.train()
        logger.info(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
            f"val loss: {ae_history['val_loss'][-1]:.6f}")
        encoder_fn = weight_autoencoder.get_encoder(device=device)
    elif encoder_type == 'identity':
        encoder_fn = ppo_agent_to_vector
    else:
        raise ValueError(f"Invalid encoder type: {encoder_type}")

    # Create and train payoff predictor
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")
    NUM_AGENTS_2 = 1000
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for i in range(NUM_AGENTS_2)]
    predictor = PayoffPredictor(
        game=game,
        ppo_agents=downstream_task_ppo_agents,
        opponent_policy=opponent_policy,
        encoder_fn=encoder_fn,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate payoff predictor on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict 0 for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    baseline_mse = np.mean(val_payoffs ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs))

    print(f"\nBaseline (constant 0 prediction):")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"MAE: {baseline_mae:.6f}")
    print(f"\nModel improvement over baseline:")
    print(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate payoff predictor on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics

def test_downstream_task_b(game: pyspiel.Game):
    """
    Test the PayoffPredictor given list of P1 agents and list of P2 agents.
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Create two lists of agents
    NUM_P1_AGENTS = 30
    NUM_P2_AGENTS = 30
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    print("\nCreating P1 and P2 agents...")
    p1_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init) for _ in range(NUM_P1_AGENTS)]
    p2_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init) for _ in range(NUM_P2_AGENTS)]

    # Combine all agents for autoencoder training
    all_agents = p1_agents + p2_agents

    # Extract weights from all agents for training the autoencoder
    print("\nExtracting weights from all agents...")
    weight_vectors = []
    for agent in tqdm(all_agents):
        # Extract weights from first layer of actor network
        first_layer_weights = agent.actor[0].weight.flatten()
        weight_vectors.append(first_layer_weights.detach().cpu())

    weight_tensor = torch.stack(weight_vectors)
    print(f"Weight tensor shape: {weight_tensor.shape}")

    # Train autoencoder on agent weights
    print("\nTraining autoencoder on agent weights...")
    input_dim = weight_tensor.shape[1]
    ae_config = AutoencoderConfig(
        input_dim=input_dim,
        hidden_dims=(512, 256),
        bottleneck_dim=64,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device="cpu",
    )
    dataset = VectorDataset(weight_tensor)
    autoencoder, ae_history = train_autoencoder(dataset, ae_config)
    print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
        f"val loss: {ae_history['val_loss'][-1]:.6f}")

    # Create encoder function using the trained autoencoder
    ae_encoder = get_encoder(autoencoder, device="cpu")
    def encoder_fn(ppo_agent):
        """Extract weights from first layer and encode using autoencoder."""
        first_layer = ppo_agent.actor[0]
        weights = first_layer.weight.flatten()
        return ae_encoder(weights)

    # Create and train predictor with agent pairs
    predictor = PayoffPredictor(
        game=game,
        p1_agents=p1_agents,
        p2_agents=p2_agents,
        encoder_fn=encoder_fn,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        device="cpu"
    )

    # Train the model
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict 0 for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    baseline_mse = np.mean(val_payoffs ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs))

    print(f"\nBaseline (constant 0 prediction):")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"MAE: {baseline_mae:.6f}")
    print(f"\nModel improvement over baseline:")
    print(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the model
    predictor.save(f"payoff_predictor_{game_short_name}_task_b.pth")

    return predictor, history, val_metrics, train_metrics

def test_downstream_task_c(game: pyspiel.Game):
    """
    Test the StatePayoffPredictor given list of P1 agents, P2 agents, and states.
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    NUM_P1_AGENTS = 20
    NUM_P2_AGENTS = 20
    NUM_STATES = 10
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    print("\nCreating P1 and P2 agents...")
    p1_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init) for _ in range(NUM_P1_AGENTS)]
    p2_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init) for _ in range(NUM_P2_AGENTS)]

    # Combine all agents for autoencoder training
    all_agents = p1_agents + p2_agents

    # Extract weights from all agents for training the autoencoder
    print("\nExtracting weights from all agents...")
    weight_vectors = []
    for agent in tqdm(all_agents):
        # Extract weights from first layer of actor network
        first_layer_weights = agent.actor[0].weight.flatten()
        weight_vectors.append(first_layer_weights.detach().cpu())

    weight_tensor = torch.stack(weight_vectors)
    print(f"Weight tensor shape: {weight_tensor.shape}")

    # Train autoencoder on agent weights
    print("\nTraining autoencoder on agent weights...")
    input_dim = weight_tensor.shape[1]
    ae_config = AutoencoderConfig(
        input_dim=input_dim,
        hidden_dims=(512, 256),
        bottleneck_dim=64,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device="cpu",
    )
    dataset = VectorDataset(weight_tensor)
    autoencoder, ae_history = train_autoencoder(dataset, ae_config)
    print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
        f"val loss: {ae_history['val_loss'][-1]:.6f}")

    # Create encoder function using the trained autoencoder
    ae_encoder = get_encoder(autoencoder, device="cpu")
    def encoder_fn(ppo_agent):
        """Extract weights from first layer and encode using autoencoder."""
        first_layer = ppo_agent.actor[0]
        weights = first_layer.weight.flatten()
        return ae_encoder(weights)

    # Create state sampler
    def state_sampler(game, num_states):
        return sample_random_states(game, num_states, max_depth=5)

    # Create and train state-conditioned predictor
    predictor = StatePayoffPredictor(
        game=game,
        p1_agents=p1_agents,
        p2_agents=p2_agents,
        encoder_fn=encoder_fn,
        state_sampler=state_sampler,
        num_states_per_pair=NUM_STATES,
        hidden_dims=[128, 64, 32],
        dropout=0.2,
        device="cpu"
    )

    # Train the model
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-3,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")

    # Baseline: predict 0 for everything
    val_payoffs = predictor.ground_truth_payoffs[predictor.val_indices]
    baseline_mse = np.mean(val_payoffs ** 2)
    baseline_mae = np.mean(np.abs(val_payoffs))

    print(f"\nBaseline (constant 0 prediction):")
    print(f"MSE: {baseline_mse:.6f}")
    print(f"MAE: {baseline_mae:.6f}")
    print(f"\nModel improvement over baseline:")
    print(f"MSE reduction: {(1 - val_metrics['mse']/baseline_mse)*100:.2f}%")
    print(f"MAE reduction: {(1 - val_metrics['mae']/baseline_mae)*100:.2f}%")

    # Evaluate on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")

    # Save the model
    predictor.save(f"payoff_predictor_{game_short_name}_task_c.pth")

    return predictor, history, val_metrics, train_metrics


if __name__ == "__main__":
    device = get_device_string()
    print("Using device:", device)
    # set_seed(42)

    for game_name in ["kuhn_poker", "leduc_poker"]:
        game = pyspiel.load_game(game_name)
        if game_name == "kuhn_poker":
            psro_ppo_agents_256 = load_ppo_agents_from_psro(hidden_size=256, shuffle=True)
            first_half, second_half = psro_ppo_agents_256[:len(psro_ppo_agents_256)//2], psro_ppo_agents_256[len(psro_ppo_agents_256)//2:]
            exp_label = f"Task A: psro {game_name} linear weight_autoencoder"
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", autoencoder_ppo_agents=first_half, downstream_task_ppo_agents=second_half,
device=device)
            exp_label = f"a psro {game_name} linear identity"
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", autoencoder_ppo_agents=first_half, downstream_task_ppo_agents=second_half, device=device)

        exp_label = f"Task A: random {game_name} linear weight_autoencoder"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", device=device)

        exp_label = f"Task A: random {game_name} linear identity"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", device=device)