"""
Example usage of the refactored downstream task architecture.

This script demonstrates how to:
1. Generate policies and embeddings
2. Configure tasks using TaskConfig objects
3. Run all four downstream tasks with the new unified interface
4. Compare different model types (MLP, linear, random forest)

The new architecture separates agent generation (done here) from prediction
(done in tasks.py), making it easy to test different model types on the same
set of policies.
"""

import pyspiel
import numpy as np
from open_spiel.python.policy import UniformRandomPolicy

from config import (
    TaskAConfig, TaskBConfig, TaskCConfig, TaskDConfig,
    ModelConfig
)
from tasks import run_task_a, run_task_b, run_task_c, run_task_d
from utils import ppo_agent_to_vector


def generate_random_policies_and_embeddings(game, num_agents=50):
    """
    Generate random policies and their embeddings.

    For this example, we use uniform random policies with simple identity embeddings.
    In real experiments, you would use trained PPO agents or PSRO/NEUPL policies.
    """
    policies = []
    embeddings = []

    for i in range(num_agents):
        policy = UniformRandomPolicy(game)
        # Simple identity embedding: just use agent index as a feature
        # In real usage, this would be ppo_agent_to_vector(agent) or autoencoder output
        embedding = np.array([i / num_agents])  # Normalized agent index

        policies.append(policy)
        embeddings.append(embedding)

    return policies, embeddings


def example_task_a():
    """
    Example: Task A with different model types.

    Task A predicts payoff of agents vs a fixed opponent (uniform random).
    """
    print("\n" + "="*80)
    print("EXAMPLE: Task A - Agent vs Fixed Opponent")
    print("="*80 + "\n")

    # Setup
    game = pyspiel.load_game("kuhn_poker")
    policies, embeddings = generate_random_policies_and_embeddings(game, num_agents=50)

    # Test different model types
    model_types = ["linear", "mlp", "random_forest"]

    for model_type in model_types:
        print(f"\n--- Testing with {model_type.upper()} model ---")

        config = TaskAConfig(
            model_config=ModelConfig(
                model_type=model_type,
                # For MLP: hidden_dims=[128, 64, 32] (auto-set)
                # For linear: hidden_dims=[] (auto-set)
                # For RF: hidden_dims=None (auto-set)
                learning_rate=1e-4,
                num_epochs=1000 if model_type != "random_forest" else 5000,  # RF needs more epochs
                batch_size=16,
                early_stopping_patience=50
            ),
            validation_split=0.2
        )

        results = run_task_a(
            game=game,
            policies=policies,
            embeddings=embeddings,
            config=config,
            exp_label=f"example_task_a_{model_type}",
            device="cpu"
        )

        print(f"Results for {model_type}:")
        print(f"  Val MSE: {results['val_metrics']['mse']:.6f}")
        print(f"  Baseline MSE: {results['val_metrics']['baseline_mse']:.6f}")
        improvement = (1 - results['val_metrics']['mse'] / results['val_metrics']['baseline_mse']) * 100
        print(f"  Improvement: {improvement:.2f}%")


def example_task_b():
    """
    Example: Task B with agent vs agent matchups.

    Task B predicts payoff for matchups between two variable agents.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Task B - Agent vs Agent")
    print("="*80 + "\n")

    # Setup
    game = pyspiel.load_game("kuhn_poker")
    p1_policies, p1_embeddings = generate_random_policies_and_embeddings(game, num_agents=30)
    p2_policies, p2_embeddings = generate_random_policies_and_embeddings(game, num_agents=30)

    # Task B now supports random_forest (NEW CAPABILITY!)
    config = TaskBConfig(
        model_config=ModelConfig(
            model_type="random_forest",
            learning_rate=1e-4,
            num_epochs=5000,
            batch_size=16
        ),
        validation_split=0.2
    )

    results = run_task_b(
        game=game,
        p1_policies=p1_policies,
        p1_embeddings=p1_embeddings,
        p2_policies=p2_policies,
        p2_embeddings=p2_embeddings,
        config=config,
        exp_label="example_task_b_rf",
        device="cpu"
    )

    print("Results:")
    print(f"  Val MSE: {results['val_metrics']['mse']:.6f}")
    print(f"  Baseline MSE: {results['val_metrics']['baseline_mse']:.6f}")
    improvement = (1 - results['val_metrics']['mse'] / results['val_metrics']['baseline_mse']) * 100
    print(f"  Improvement: {improvement:.2f}%")


def example_task_c():
    """
    Example: Task C with state-conditioned prediction.

    Task C predicts payoff conditioned on game state.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Task C - State-Conditioned Prediction")
    print("="*80 + "\n")

    # Setup
    game = pyspiel.load_game("kuhn_poker")
    p1_policies, p1_embeddings = generate_random_policies_and_embeddings(game, num_agents=20)
    p2_policies, p2_embeddings = generate_random_policies_and_embeddings(game, num_agents=20)

    # Task C now supports random_forest (NEW CAPABILITY!)
    config = TaskCConfig(
        model_config=ModelConfig(
            model_type="mlp",
            hidden_dims=[128, 64, 32],
            learning_rate=1e-4,
            num_epochs=2000,
            batch_size=32
        ),
        num_states=15,  # Sample 15 game states
        max_state_depth=5,
        validation_split=0.2
    )

    results = run_task_c(
        game=game,
        p1_policies=p1_policies,
        p1_embeddings=p1_embeddings,
        p2_policies=p2_policies,
        p2_embeddings=p2_embeddings,
        config=config,
        exp_label="example_task_c_mlp",
        device="cpu"
    )

    print("Results:")
    print(f"  Val MSE: {results['val_metrics']['mse']:.6f}")
    print(f"  Baseline MSE: {results['val_metrics']['baseline_mse']:.6f}")
    improvement = (1 - results['val_metrics']['mse'] / results['val_metrics']['baseline_mse']) * 100
    print(f"  Improvement: {improvement:.2f}%")


def example_task_d():
    """
    Example: Task D with exploitability prediction.

    Task D predicts how exploitable a policy is.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Task D - Exploitability Prediction")
    print("="*80 + "\n")

    # Setup
    game = pyspiel.load_game("kuhn_poker")
    policies, embeddings = generate_random_policies_and_embeddings(game, num_agents=40)

    # Task D now supports random_forest (NEW CAPABILITY!)
    config = TaskDConfig(
        model_config=ModelConfig(
            model_type="random_forest",
            learning_rate=1e-4,
            num_epochs=5000,
            batch_size=16
        ),
        player_id=0,  # Evaluate exploitability from player 0's perspective
        validation_split=0.2
    )

    results = run_task_d(
        game=game,
        policies=policies,
        embeddings=embeddings,
        config=config,
        exp_label="example_task_d_rf",
        device="cpu"
    )

    print("Results:")
    print(f"  Val MSE: {results['val_metrics']['mse']:.6f}")
    print(f"  Baseline MSE: {results['val_metrics']['baseline_mse']:.6f}")
    improvement = (1 - results['val_metrics']['mse'] / results['val_metrics']['baseline_mse']) * 100
    print(f"  Improvement: {improvement:.2f}%")


def example_config_serialization():
    """
    Example: Saving and loading configuration for experiment tracking.

    The config objects can be easily serialized to JSON for reproducibility.
    """
    print("\n" + "="*80)
    print("EXAMPLE: Configuration Serialization")
    print("="*80 + "\n")

    import json
    from config import config_to_dict

    # Create a config
    config = TaskAConfig(
        model_config=ModelConfig(
            model_type="mlp",
            hidden_dims=[256, 128, 64],
            dropout=0.1,
            learning_rate=5e-4,
            num_epochs=3000,
            batch_size=32,
            early_stopping_patience=100
        ),
        validation_split=0.15
    )

    # Serialize to dict/JSON
    config_dict = config_to_dict(config)
    config_json = json.dumps(config_dict, indent=2)

    print("Configuration as JSON:")
    print(config_json)

    print("\nThis can be saved to a file for experiment tracking:")
    print("  - Track which hyperparameters were used")
    print("  - Reproduce experiments exactly")
    print("  - Compare configurations across runs")

    # Could save to file:
    # with open("experiments/exp123/config.json", "w") as f:
    #     json.dump(config_dict, f, indent=2)

    # And load back:
    # with open("experiments/exp123/config.json", "r") as f:
    #     loaded_dict = json.load(f)
    # loaded_config = TaskAConfig(**loaded_dict)  # Reconstruct config


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("REFACTORED DOWNSTREAM TASK ARCHITECTURE - EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates the new unified interface for all tasks.")
    print("Key features:")
    print("  - Consistent interface across all tasks")
    print("  - Easy to swap model types (MLP, linear, random_forest)")
    print("  - Type-safe configuration via dataclasses")
    print("  - All tasks automatically register results")
    print("  - Separation of agent generation from prediction")

    # Run examples (comment out any you don't want to run)
    example_task_a()
    # example_task_b()  # Uncomment to run
    # example_task_c()  # Uncomment to run
    # example_task_d()  # Uncomment to run
    # example_config_serialization()  # Uncomment to run

    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
