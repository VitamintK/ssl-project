import pyspiel
import torch

from tqdm import tqdm
from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed
from weight_autoencoder import (
    WeightAutoencoder,
    AutoencoderConfig,
    train_autoencoder,
    VectorDataset,
    get_encoder,
)


def test_kuhn_poker_example():
    """
    Test the PayoffPredictor on Kuhn Poker with the specified configuration.
    """

    # Load Kuhn Poker game
    game = pyspiel.load_game("kuhn_poker")
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    # Create opponent policy (UniformRandomPolicy)
    opponent_policy = policy_lib.UniformRandomPolicy(game)

    # Create random PPO agents using diverse initialization
    print("Creating random PPO agents...")
    num_agents = 300
    ppo_agents = []

    for _ in tqdm(range(num_agents)):
        layer_init = make_diverse_random_kuhn_poker_layer_init(game)
        agent = PPOAgent(
            num_actions=num_actions,
            observation_shape=info_state_size,
            device="cpu",
            layer_init=layer_init
        )
        ppo_agents.append(agent)

    # Extract weights from all agents for training the autoencoder
    print("\nExtracting weights from agents...")
    weight_vectors = []
    for agent in tqdm(ppo_agents):
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

    # Create and train predictor
    predictor = PayoffPredictor(
        game=game,
        ppo_agents=ppo_agents,
        opponent_policy=opponent_policy,
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
    print(f"R2: {val_metrics['r2']:.6f}")

    # Evaluate on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")
    print(f"R2: {train_metrics['r2']:.6f}")

    # Save the model
    predictor.save("payoff_predictor_kuhn.pth")

    return predictor, history, val_metrics, train_metrics


if __name__ == "__main__":
    set_seed(42)
    test_kuhn_poker_example()