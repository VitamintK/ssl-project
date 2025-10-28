from typing import Literal
import pyspiel
import torch
from torch import nn
from pathlib import Path

from tqdm import tqdm
from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed
from weight_autoencoder import (
    AutoencoderConfig,
    train_autoencoder,
    VectorDataset,
    get_encoder,
)




def test_downstream_task_a(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"],
        encoder_type: Literal["identity", "weight_autoencoder"],
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
    NUM_AGENTS = 1000
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)
    ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for i in range(NUM_AGENTS)]

    if encoder_type == 'weight_autoencoder':
        # Extract weights from all agents for training the autoencoder
        print("\nExtracting weights from agents...")
        weight_vectors = []
        for agent in tqdm(ppo_agents):
            parameters = [param.detach() for param in agent.actor.parameters()]
            parameters = nn.utils.parameters_to_vector(parameters)
            weight_vectors.append(parameters)
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
            device=device,
        )
        dataset = VectorDataset(weight_tensor)
        autoencoder, ae_history = train_autoencoder(dataset, ae_config)
        print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
            f"val loss: {ae_history['val_loss'][-1]:.6f}")

        # Create encoder function using the trained autoencoder
        ae_encoder = get_encoder(autoencoder, device="cpu")
        def encoder_fn(ppo_agent: PPOAgent):
            """Extract weights from first layer and encode using autoencoder."""
            parameters = [param.detach() for param in ppo_agent.actor.parameters()]
            parameters = nn.utils.parameters_to_vector(parameters)
            return ae_encoder(parameters)
    else:
        def encoder_fn(ppo_agent: PPOAgent):
            parameters = [param.detach() for param in ppo_agent.actor.parameters()]
            # return ppo_agent.actor[0].weight.flatten()
            parameters = nn.utils.parameters_to_vector(parameters)
            return parameters

    # Create and train predictor
    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")
    NUM_AGENTS_2 = 1000
    ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for i in range(NUM_AGENTS_2)]
    predictor = PayoffPredictor(
        game=game,
        ppo_agents=ppo_agents,
        opponent_policy=opponent_policy,
        encoder_fn=encoder_fn,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )

    # Train the model
    print("\nTraining predictor model...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    # Evaluate on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")
    print(f"R2: {val_metrics['r2']:.6f}")

    # Evaluate on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")
    print(f"R2: {train_metrics['r2']:.6f}")

    # Save the model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    # set_seed(42)
    game = pyspiel.load_game("kuhn_poker")
    test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", device=device)
    game = pyspiel.load_game("kuhn_poker")
    test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", device=device)
    # game = pyspiel.load_game("leduc_poker")
    # test_downstream_task_a(game, device=device)