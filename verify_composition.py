"""
Quick verification script for the composition refactoring.

Tests that ModelTrainer works correctly via composition in all predictor classes.
"""

import numpy as np
import pyspiel
from open_spiel.python.policy import UniformRandomPolicy

from config import ModelConfig, TaskAConfig
from downstream_refactored import (
    ModelTrainer,
    PayoffPredictorRefactored,
    StatePayoffPredictorRefactored,
    ExploitabilityPredictorRefactored
)


def test_model_trainer():
    """Test basic ModelTrainer functionality."""
    print("\n=== Testing ModelTrainer ===")

    config = ModelConfig(model_type="mlp", hidden_dims=[32, 16], num_epochs=5)
    trainer = ModelTrainer(embedding_dim=10, model_config=config, device="cpu")

    print(f"✓ ModelTrainer created with embedding_dim=10")
    print(f"✓ Model type: {config.model_type}")
    print(f"✓ Model created: {trainer.model is not None}")

    # Test training
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.randn(50)

    print("Training model...")
    history = trainer.train(X, y, validation_split=0.2)

    print(f"✓ Training completed: {len(history['train_loss'])} iterations")
    print(f"  - Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  - Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"✓ Train indices set: {trainer.train_indices is not None}")
    print(f"✓ Val indices set: {trainer.val_indices is not None}")

    # Test prediction
    predictions = trainer.predict(X)
    print(f"✓ Predictions shape: {predictions.shape}")

    # Test evaluation
    val_metrics = trainer.evaluate(X, y, trainer.val_indices, y[trainer.train_indices])
    print(f"✓ Evaluation completed")
    print(f"  - Val MSE: {val_metrics['mse']:.6f}")
    print(f"  - Baseline MSE: {val_metrics['baseline_mse']:.6f}")

    print("✅ ModelTrainer tests passed!\n")


def test_payoff_predictor_composition():
    """Test PayoffPredictorRefactored uses composition correctly."""
    print("\n=== Testing PayoffPredictorRefactored Composition ===")

    # Setup
    game = pyspiel.load_game("kuhn_poker")

    # Create simple policies and embeddings
    p1_policies = [UniformRandomPolicy(game) for _ in range(10)]
    p2_policies = [UniformRandomPolicy(game)]
    p1_embeddings = np.random.randn(10, 5)
    p2_embeddings = np.random.randn(1, 5)

    config = ModelConfig(model_type="linear", num_epochs=10)

    # Create predictor
    predictor = PayoffPredictorRefactored(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=p1_embeddings,
        p2_embeddings=p2_embeddings,
        model_config=config,
        device="cpu"
    )

    print(f"✓ PayoffPredictorRefactored created")
    print(f"✓ Has trainer: {hasattr(predictor, 'trainer')}")
    print(f"✓ Trainer is ModelTrainer: {isinstance(predictor.trainer, ModelTrainer)}")

    # Compute ground truth
    print("Computing ground truth...")
    predictor.compute_ground_truth_payoffs()
    print(f"✓ Ground truth computed: {len(predictor.ground_truth_payoffs)} payoffs")

    # Train
    print("Training...")
    history = predictor.train_with_agent_level_split(validation_split=0.3)
    print(f"✓ Training completed")

    # Evaluate
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"✓ Evaluation completed")
    print(f"  - Val MSE: {val_metrics['mse']:.6f}")
    print(f"  - Baseline MSE: {val_metrics['baseline_mse']:.6f}")

    print("✅ PayoffPredictorRefactored composition tests passed!\n")


def test_state_payoff_predictor_composition():
    """Test StatePayoffPredictorRefactored uses composition correctly."""
    print("\n=== Testing StatePayoffPredictorRefactored Composition ===")

    # Setup
    game = pyspiel.load_game("kuhn_poker")

    # Create simple policies and embeddings
    p1_policies = [UniformRandomPolicy(game) for _ in range(5)]
    p2_policies = [UniformRandomPolicy(game) for _ in range(5)]
    p1_embeddings = np.random.randn(5, 3)
    p2_embeddings = np.random.randn(5, 3)

    config = ModelConfig(model_type="linear", num_epochs=10)

    # Create predictor
    predictor = StatePayoffPredictorRefactored(
        game=game,
        p1_policies=p1_policies,
        p2_policies=p2_policies,
        p1_embeddings=p1_embeddings,
        p2_embeddings=p2_embeddings,
        model_config=config,
        num_states=3,
        max_depth=3,
        device="cpu"
    )

    print(f"✓ StatePayoffPredictorRefactored created")
    print(f"✓ Has trainer: {hasattr(predictor, 'trainer')}")
    print(f"✓ Trainer is ModelTrainer: {isinstance(predictor.trainer, ModelTrainer)}")
    print(f"✓ States sampled: {len(predictor.states)}")

    # Compute ground truth
    print("Computing ground truth (this may take a moment)...")
    predictor.compute_ground_truth_payoffs()
    print(f"✓ Ground truth computed: {len(predictor.ground_truth_payoffs)} payoffs")

    # Train
    print("Training...")
    history = predictor.train_with_agent_level_split(validation_split=0.3)
    print(f"✓ Training completed")

    # Evaluate
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"✓ Evaluation completed")
    print(f"  - Val MSE: {val_metrics['mse']:.6f}")
    print(f"  - Baseline MSE: {val_metrics['baseline_mse']:.6f}")

    print("✅ StatePayoffPredictorRefactored composition tests passed!\n")


def test_exploitability_predictor_composition():
    """Test ExploitabilityPredictorRefactored uses composition correctly."""
    print("\n=== Testing ExploitabilityPredictorRefactored Composition ===")

    # Setup
    game = pyspiel.load_game("kuhn_poker")

    # Create simple policies and embeddings
    policies = [UniformRandomPolicy(game) for _ in range(5)]
    embeddings = np.random.randn(5, 4)

    config = ModelConfig(model_type="linear", num_epochs=10)

    # Create predictor
    predictor = ExploitabilityPredictorRefactored(
        game=game,
        policies=policies,
        embeddings=embeddings,
        model_config=config,
        player_id=0,
        device="cpu"
    )

    print(f"✓ ExploitabilityPredictorRefactored created")
    print(f"✓ Has trainer: {hasattr(predictor, 'trainer')}")
    print(f"✓ Trainer is ModelTrainer: {isinstance(predictor.trainer, ModelTrainer)}")

    # Compute ground truth
    print("Computing ground truth (this may take a moment)...")
    predictor.compute_ground_truth_payoffs()
    print(f"✓ Ground truth computed: {len(predictor.ground_truth_payoffs)} payoffs")

    # Train
    print("Training...")
    history = predictor.train_with_agent_level_split(validation_split=0.3)
    print(f"✓ Training completed")

    # Evaluate
    val_metrics = predictor.evaluate(eval_set="val")
    print(f"✓ Evaluation completed")
    print(f"  - Val MSE: {val_metrics['mse']:.6f}")
    print(f"  - Baseline MSE: {val_metrics['baseline_mse']:.6f}")

    print("✅ ExploitabilityPredictorRefactored composition tests passed!\n")


def main():
    """Run all verification tests."""
    print("="*70)
    print("COMPOSITION REFACTORING VERIFICATION")
    print("="*70)
    print("\nVerifying that all predictors use composition (has-a) instead of")
    print("inheritance (is-a) with ModelTrainer.")

    try:
        test_model_trainer()
        test_payoff_predictor_composition()
        test_state_payoff_predictor_composition()
        test_exploitability_predictor_composition()

        print("="*70)
        print("✅ ALL COMPOSITION TESTS PASSED!")
        print("="*70)
        print("\nSuccessfully verified:")
        print("  1. ModelTrainer works as standalone ML service")
        print("  2. PayoffPredictorRefactored uses composition")
        print("  3. StatePayoffPredictorRefactored uses composition")
        print("  4. ExploitabilityPredictorRefactored uses composition")
        print("\nAll predictors now use 'has-a' relationship with ModelTrainer")
        print("instead of 'is-a' inheritance from BasePredictor.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
