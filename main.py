from typing import Literal
import math
import pyspiel
import torch
from pathlib import Path

from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from utils import make_diverse_random_kuhn_poker_layer_init
from downstream import PayoffPredictor, set_seed
from functional_autoencoder import (
    TrainingConfig,
    train_functional_autoencoder,
    FunctionalEncoderAdapter,
)
from weight_autoencoder import (
    AutoencoderConfig,
    WeightAutoencoder,
    ppo_agent_to_vector,
    save_autoencoder,
    load_autoencoder,
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
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    if encoder_type == 'weight_autoencoder':
        # Train autoencoder on agent weights
        NUM_AGENTS_AUTOENCODE = 1000
        ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for i in range(NUM_AGENTS_AUTOENCODE)]
        print("\nTraining autoencoder on agent weights...")
        ae_config = AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=64,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            device=device,
        )
        weight_autoencoder = WeightAutoencoder(ae_config, ppo_agents, ppo_agent_to_vector)
        autoencoder_model, ae_history = weight_autoencoder.train()
        save_autoencoder(
            autoencoder_model,
            ae_config,
            Path("results") / experiment_label / f"{game_short_name}_autoencoder.pth",
        )
        print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
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
    print(f"R2: {val_metrics['r2']:.6f}")

    # Evaluate payoff predictor on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")
    print(f"R2: {train_metrics['r2']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics



def test_downstream_task_feature_encoder(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"] = "linear",
        experiment_label: str = "downstream_feature",
        device: str = "cpu",
):
    print("\n" + "="*80)
    print("Testing Downstream Task with Functional Feature Encoder")
    print("="*80 + "\n")
    """
    Train a functional feature encoder on Kuhn Poker PPO agents and evaluate
    the downstream payoff predictor using the learned embeddings.
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()
    print(f"Game: {game_short_name}, Info State Size: {info_state_size}, Num Actions: {num_actions}")
    opponent_policy = policy_lib.UniformRandomPolicy(game)
    PPO_AGENT_HIDDEN_SIZE = 64
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    NUM_ENCODER_AGENTS = 1000
    feature_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(NUM_ENCODER_AGENTS)
    ]
    feature_cfg = TrainingConfig(
        num_agents=NUM_ENCODER_AGENTS,
        ppo_hidden_size=PPO_AGENT_HIDDEN_SIZE,
        autoencoder=AutoencoderConfig(
            hidden_dims=(512, 256),
            bottleneck_dim=128,
            epochs=10,
            batch_size=32,
            lr=3e-4,
            device=device,
        ),
    )
    print("\nTraining functional feature encoder...")
    feature_model, feature_history = train_functional_autoencoder(
        feature_cfg,
        game=game,
        agents=feature_agents,
    )
    print(f"Feature encoder trained. Final KL: {feature_history[-1]:.6f}")
    weight_autoencoder = FunctionalEncoderAdapter(feature_model)
    encoder_fn = weight_autoencoder.get_encoder(device=device)

    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    NUM_PREDICTOR_AGENTS = 200
    predictor_agents = [
        PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE)
        for _ in range(NUM_PREDICTOR_AGENTS)
    ]
    predictor = PayoffPredictor(
        game=game,
        ppo_agents=predictor_agents,
        opponent_policy=opponent_policy,
        encoder_fn=encoder_fn,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )

    print("\nTraining payoff predictor with feature encoder...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    print("\nEvaluating feature-encoder model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results (feature encoder):")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")
    print(f"R2: {val_metrics['r2']:.6f}")

    print("\nEvaluating feature-encoder model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results (feature encoder):")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")
    print(f"R2: {train_metrics['r2']:.6f}")

    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(str(results_dir / f"{game_short_name}_predictor_{predictor_type}_feature.pth"))

    return predictor, history, val_metrics, train_metrics, feature_history


def test_downstream_task_load(
        game: pyspiel.Game,
        predictor_type: Literal["mlp", "linear"],
        experiment_label: str = "downstream_a",
        device: str = "cpu",
        autoencoder_path: Path | None = None,
        expected_val_metrics: dict[str, float] | None = None,
        expected_train_metrics: dict[str, float] | None = None,
        metric_tolerance: float = 1e-6,
):
    """
    Load a previously trained autoencoder and run the payoff prediction task.

    Args:
        game: The OpenSpiel game to use.
        predictor_type: Predictor architecture ("mlp" or "linear").
        experiment_label: Subdirectory used when saving checkpoints.
        device: Torch device string.
        autoencoder_path: Optional explicit checkpoint path; defaults to the training path.
        expected_val_metrics: Optional reference metrics for validation split.
        expected_train_metrics: Optional reference metrics for training split.
        metric_tolerance: Allowed absolute/relative tolerance when comparing metrics.
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)
    PPO_AGENT_HIDDEN_SIZE = 256
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    checkpoint_path = autoencoder_path or Path("results") / experiment_label / f"{game_short_name}_autoencoder.pth"
    autoencoder, _ = load_autoencoder(checkpoint_path, device=device)

    def encoder_fn(policy: PPOAgent) -> torch.Tensor:
        """Encode a policy using the restored autoencoder bottleneck."""
        with torch.no_grad():
            vector = ppo_agent_to_vector(policy)
            if not isinstance(vector, torch.Tensor):
                vector = torch.tensor(vector)
            vector = vector.float().to(device)
            if vector.ndim == 1:
                vector = vector.unsqueeze(0)
            embedding = autoencoder.encoder(vector)
            return embedding.squeeze(0)

    if predictor_type == "mlp":
        hidden_dims = [128, 64, 32]
    elif predictor_type == "linear":
        hidden_dims = []
    else:
        raise ValueError(f"Invalid predictor type: {predictor_type}")

    NUM_AGENTS = 1000
    ppo_agents = [PPOAgent(num_actions, info_state_size, 'cpu', layer_init, PPO_AGENT_HIDDEN_SIZE) for _ in range(NUM_AGENTS)]
    predictor = PayoffPredictor(
        game=game,
        ppo_agents=ppo_agents,
        opponent_policy=opponent_policy,
        encoder_fn=encoder_fn,
        hidden_dims=hidden_dims,
        dropout=0.2,
        device="cpu"
    )

    print("\nTraining predictor model with loaded autoencoder...")
    history = predictor.train(
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        validation_split=0.2,
        verbose=True
    )

    print("\nEvaluating loaded-model predictor on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results (loaded):")
    print(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")
    print(f"R2: {val_metrics['r2']:.6f}")

    print("\nEvaluating loaded-model predictor on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results (loaded):")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")
    print(f"R2: {train_metrics['r2']:.6f}")

    return predictor, history, val_metrics, train_metrics


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    game = pyspiel.load_game("kuhn_poker")

    set_seed(42)
    # _, _, baseline_val_metrics, baseline_train_metrics = test_downstream_task_a(
    #     game,
    #     predictor_type="linear",
    #     encoder_type="weight_autoencoder",
    #     device=device,
    # )

    # set_seed(42)
    # test_downstream_task_load(
    #     game,
    #     predictor_type="linear",
    #     experiment_label="downstream_a",
    #     device=device,
    #     expected_val_metrics=baseline_val_metrics,
    #     expected_train_metrics=baseline_train_metrics,
    # )


    test_downstream_task_feature_encoder(
        game,
        predictor_type="linear",
        experiment_label="downstream_feature",
        device=device,
    )
