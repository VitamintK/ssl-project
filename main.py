import math
import random
from typing import Literal, Optional
import pyspiel
import torch
from pathlib import Path
import logging

from open_spiel.python import policy as policy_lib
from iig_rl_benchmark.algorithms.ppo.ppo import PPOAgent
from psro import load_ppo_agents_from_psro
from utils import get_device_string, make_diverse_random_kuhn_poker_layer_init
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
        encoder_type: Literal["identity", "weight_autoencoder", "functional_autoencoder"],
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
        encoder_type: Type of encoder ("identity", "weight_autoencoder", or "functional_autoencoder")
        experiment_label: Label for the experiment subdirectory (default: "downstream_a")
    """
    game_short_name = game.get_type().short_name
    info_state_size = game.information_state_tensor_shape()
    num_actions = game.num_distinct_actions()

    opponent_policy = policy_lib.UniformRandomPolicy(game)
    layer_init = make_diverse_random_kuhn_poker_layer_init(game)

    encoder_defaults = {
        "identity": {
            "predictor_hidden_size": 256,
            "predictor_agent_count": 100,
            "encoder_hidden_size": None,
            "autoencoder_agent_count": 0,
        },
        "weight_autoencoder": {
            "predictor_hidden_size": 256,
            "predictor_agent_count": 100,
            "encoder_hidden_size": 256,
            "autoencoder_agent_count": 100,
        },
        "functional_autoencoder": {
            "predictor_hidden_size": 64,
            "predictor_agent_count": 100,
            "encoder_hidden_size": 64,
            "autoencoder_agent_count": 100,
        },
    }

    if encoder_type not in encoder_defaults:
        raise ValueError(f"Invalid encoder type: {encoder_type}")

    defaults = encoder_defaults[encoder_type]
    predictor_hidden_size = defaults["predictor_hidden_size"]
    predictor_agent_count = defaults["predictor_agent_count"]
    encoder_agents_hidden_size = defaults["encoder_hidden_size"]
    autoencoder_agent_count = defaults["autoencoder_agent_count"]

    if encoder_type == 'weight_autoencoder':
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, encoder_agents_hidden_size)
                for _ in range(autoencoder_agent_count)
            ]
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
        autoencoder_model, ae_history = weight_autoencoder.train()
        save_autoencoder(
            autoencoder_model,
            ae_config,
            Path("results") / experiment_label / f"{game_short_name}_autoencoder.pth",
        )
        print(f"Autoencoder trained. Final train loss: {ae_history['train_loss'][-1]:.6f}, "
            f"val loss: {ae_history['val_loss'][-1]:.6f}")
        encoder_fn = weight_autoencoder.get_encoder(device=device)
    elif encoder_type == 'functional_autoencoder':
        if autoencoder_ppo_agents is None:
            autoencoder_ppo_agents = [
                PPOAgent(num_actions, info_state_size, 'cpu', layer_init, encoder_agents_hidden_size)
                for _ in range(autoencoder_agent_count)
            ]
        feature_cfg = TrainingConfig(
            num_agents=len(autoencoder_ppo_agents),
            ppo_hidden_size=encoder_agents_hidden_size,
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
            agents=autoencoder_ppo_agents,
        )
        print(f"Feature encoder trained. Final KL: {feature_history[-1]:.6f}")
        encoder_adapter = FunctionalEncoderAdapter(feature_model)
        encoder_fn = encoder_adapter.get_encoder(device=device)
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
    if downstream_task_ppo_agents is None:
        downstream_task_ppo_agents = [
            PPOAgent(num_actions, info_state_size, 'cpu', layer_init, predictor_hidden_size)
            for _ in range(predictor_agent_count)
        ]
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

    # Evaluate payoff predictor on training set for comparison
    print("\nEvaluating model on training set...")
    train_metrics = predictor.evaluate(eval_set="train")
    print(f"\nTraining Set Results:")
    print(f"MSE: {train_metrics['mse']:.6f}")
    print(f"MAE: {train_metrics['mae']:.6f}")
    print(f"R2: {train_metrics['r2']:.6f}")

    # Evaluate payoff predictor on validation set
    print("\nEvaluating model on validation set...")
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"\nValidation Set Results:")
    logger.info(f"MSE: {val_metrics['mse']:.6f}")
    print(f"MAE: {val_metrics['mae']:.6f}")
    print(f"R2: {val_metrics['r2']:.6f}")

    # Save the payoff predictor model
    results_dir = Path("results") / experiment_label
    results_dir.mkdir(parents=True, exist_ok=True)
    save_path = results_dir / f"{game_short_name}_predictor_{predictor_type}_{encoder_type}.pth"
    predictor.save(str(save_path))
    print(f"\nModel saved to: {save_path}")

    return predictor, history, val_metrics, train_metrics



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
            test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", autoencoder_ppo_agents=first_half, downstream_task_ppo_agents=second_half, device=device)
            exp_label = f"a psro {game_name} linear identity"
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", autoencoder_ppo_agents=first_half, downstream_task_ppo_agents=second_half, device=device)
            exp_label = f"Task Feature Encoder: psro {game_name} linear"
            logger.info(f"Running experiment: {exp_label}")
            test_downstream_task_a(
                game,
                predictor_type="linear",
                encoder_type="functional_autoencoder",
                autoencoder_ppo_agents=first_half,
                downstream_task_ppo_agents=second_half,
                experiment_label=exp_label,
                device=device,
            )

          

        exp_label = f"Task A: random {game_name} linear weight_autoencoder"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(game, predictor_type="linear", encoder_type="weight_autoencoder", device=device)

        exp_label = f"Task A: random {game_name} linear identity"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(game, predictor_type="linear", encoder_type="identity", device=device)

        exp_label = f"Task Feature Encoder: {game_name} linear"
        logger.info(f"Running experiment: {exp_label}")
        test_downstream_task_a(
            game,
            predictor_type="linear",
            encoder_type="functional_autoencoder",
            experiment_label=exp_label,
            device=device,
        )
