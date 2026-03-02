"""
Unit tests for ModelTrainer composition pattern.

Tests that ModelTrainer provides correct ML operations when used
via composition by the predictor classes.
"""

import pytest
import numpy as np
import torch
from config import ModelConfig
from downstream_refactored import ModelTrainer


class TestModelTrainer:
    """Test ModelTrainer used via composition."""

    def test_initialization(self):
        """Test ModelTrainer initializes correctly."""
        config = ModelConfig(model_type="mlp", hidden_dims=[64, 32])
        trainer = ModelTrainer(embedding_dim=10, model_config=config, device="cpu")

        assert trainer.embedding_dim == 10
        assert trainer.model_config.model_type == "mlp"
        assert trainer.device == "cpu"
        assert trainer.model is not None
        assert trainer.train_indices is None
        assert trainer.val_indices is None

    def test_mlp_model_creation(self):
        """Test that MLP model is created with correct architecture."""
        config = ModelConfig(model_type="mlp", hidden_dims=[64, 32])
        trainer = ModelTrainer(embedding_dim=10, model_config=config, device="cpu")

        # Check model exists and is a PyTorch module
        assert isinstance(trainer.model, torch.nn.Module)

        # Test forward pass
        X = np.random.randn(5, 10)
        predictions = trainer.predict(X)
        assert predictions.shape == (5,)

    def test_linear_model_creation(self):
        """Test that linear model is created correctly."""
        config = ModelConfig(model_type="linear")
        trainer = ModelTrainer(embedding_dim=10, model_config=config, device="cpu")

        # Check model exists
        assert isinstance(trainer.model, torch.nn.Module)

        # Test forward pass
        X = np.random.randn(5, 10)
        predictions = trainer.predict(X)
        assert predictions.shape == (5,)

    def test_random_forest_model_creation(self):
        """Test that random forest model is created correctly."""
        config = ModelConfig(model_type="random_forest")
        trainer = ModelTrainer(embedding_dim=10, model_config=config, device="cpu")

        # Check model is sklearn RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(trainer.model, RandomForestRegressor)

    def test_predict_before_training(self):
        """Test that predict works before training (random predictions)."""
        config = ModelConfig(model_type="mlp")
        trainer = ModelTrainer(embedding_dim=10, model_config=config, device="cpu")

        X = np.random.randn(10, 10)
        predictions = trainer.predict(X)

        assert predictions.shape == (10,)
        assert not np.isnan(predictions).any()

    def test_train_neural_network(self):
        """Test training a neural network model."""
        config = ModelConfig(
            model_type="mlp",
            hidden_dims=[32, 16],
            num_epochs=10,
            batch_size=8,
            learning_rate=1e-3
        )
        trainer = ModelTrainer(embedding_dim=5, model_config=config, device="cpu")

        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(50) * 0.1  # Linear relationship

        # Train with split
        history = trainer.train(X, y, validation_split=0.2)

        # Check history contains expected keys
        assert 'train_mse' in history
        assert 'val_mse' in history
        assert len(history['train_mse']) > 0
        assert len(history['val_mse']) > 0

        # Check that training reduces loss
        assert history['train_mse'][-1] < history['train_mse'][0]

    def test_train_random_forest(self):
        """Test training a random forest model."""
        config = ModelConfig(
            model_type="random_forest",
            num_epochs=100,  # For RF, this is n_estimators
            learning_rate=1e-3  # Ignored for RF
        )
        trainer = ModelTrainer(embedding_dim=5, model_config=config, device="cpu")

        # Generate synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(50) * 0.1

        # Train with split
        history = trainer.train(X, y, validation_split=0.2)

        # Check history contains expected keys
        assert 'train_mse' in history
        assert 'val_mse' in history

        # Check model is trained
        predictions = trainer.predict(X)
        assert predictions.shape == (50,)
        assert not np.isnan(predictions).any()

    def test_evaluate_uses_training_mean_baseline(self):
        """Test that evaluate() uses training set mean for baseline."""
        config = ModelConfig(model_type="linear", num_epochs=10)
        trainer = ModelTrainer(embedding_dim=3, model_config=config, device="cpu")

        # Generate data with clear train/val split
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        # Train
        trainer.train(X, y, validation_split=0.2)

        # Evaluate on validation set
        val_metrics = trainer.evaluate(
            X, y, trainer.val_indices, y[trainer.train_indices]
        )

        # Manually compute expected baseline
        train_mean = np.mean(y[trainer.train_indices])
        val_y = y[trainer.val_indices]
        expected_baseline = np.mean((val_y - train_mean) ** 2)

        assert 'mse' in val_metrics
        assert 'mae' in val_metrics
        assert 'baseline_mse' in val_metrics
        assert np.isclose(val_metrics['baseline_mse'], expected_baseline, rtol=1e-5)

    def test_evaluate_on_different_sets(self):
        """Test that evaluate can be called on train, val, or all data."""
        config = ModelConfig(model_type="mlp", num_epochs=10)
        trainer = ModelTrainer(embedding_dim=3, model_config=config, device="cpu")

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Train
        trainer.train(X, y, validation_split=0.3)

        # Evaluate on train set
        train_metrics = trainer.evaluate(
            X, y, trainer.train_indices, y[trainer.train_indices]
        )
        assert train_metrics['mse'] >= 0

        # Evaluate on val set
        val_metrics = trainer.evaluate(
            X, y, trainer.val_indices, y[trainer.train_indices]
        )
        assert val_metrics['mse'] >= 0

        # Evaluate on all data
        all_indices = np.arange(len(y))
        all_metrics = trainer.evaluate(
            X, y, all_indices, y[trainer.train_indices]
        )
        assert all_metrics['mse'] >= 0

    def test_train_indices_set_correctly(self):
        """Test that train and val indices are set correctly after training."""
        config = ModelConfig(model_type="linear", num_epochs=5)
        trainer = ModelTrainer(embedding_dim=3, model_config=config, device="cpu")

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        # Train with 20% validation
        trainer.train(X, y, validation_split=0.2)

        # Check indices are set
        assert trainer.train_indices is not None
        assert trainer.val_indices is not None

        # Check sizes
        assert len(trainer.train_indices) + len(trainer.val_indices) == 50
        assert len(trainer.val_indices) == 10  # 20% of 50

        # Check no overlap
        train_set = set(trainer.train_indices)
        val_set = set(trainer.val_indices)
        assert len(train_set.intersection(val_set)) == 0

    def test_early_stopping(self):
        """Test that early stopping works for neural networks."""
        config = ModelConfig(
            model_type="mlp",
            hidden_dims=[32, 16],
            num_epochs=1000,  # Large number to trigger early stopping
            early_stopping_patience=5,
            learning_rate=1e-3
        )
        trainer = ModelTrainer(embedding_dim=3, model_config=config, device="cpu")

        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(50) * 0.1

        # Train
        history = trainer.train(X, y, validation_split=0.2)

        # Should stop early (not reach 1000 epochs)
        assert len(history['train_mse']) < 1000
        print(f"Stopped after {len(history['train_mse'])} epochs (early stopping)")

    def test_predict_consistent_with_model_type(self):
        """Test that predict() delegates correctly based on model type."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10)

        # Test MLP
        mlp_trainer = ModelTrainer(5, ModelConfig(model_type="mlp", num_epochs=5), "cpu")
        mlp_trainer.train(X, y, validation_split=0.2)
        mlp_preds = mlp_trainer.predict(X)
        assert mlp_preds.shape == (10,)

        # Test Linear
        linear_trainer = ModelTrainer(5, ModelConfig(model_type="linear", num_epochs=5), "cpu")
        linear_trainer.train(X, y, validation_split=0.2)
        linear_preds = linear_trainer.predict(X)
        assert linear_preds.shape == (10,)

        # Test Random Forest
        rf_trainer = ModelTrainer(5, ModelConfig(model_type="random_forest"), "cpu")
        rf_trainer.train(X, y, validation_split=0.2)
        rf_preds = rf_trainer.predict(X)
        assert rf_preds.shape == (10,)


class TestModelTrainerWithDifferentConfigs:
    """Test ModelTrainer with various configurations."""

    def test_different_hidden_dims(self):
        """Test creating models with different hidden layer sizes."""
        configs = [
            ModelConfig(model_type="mlp", hidden_dims=[128, 64, 32]),
            ModelConfig(model_type="mlp", hidden_dims=[256, 128]),
            ModelConfig(model_type="mlp", hidden_dims=[64]),
        ]

        for config in configs:
            trainer = ModelTrainer(10, config, "cpu")
            assert trainer.model is not None

            # Test forward pass
            X = np.random.randn(5, 10)
            predictions = trainer.predict(X)
            assert predictions.shape == (5,)

    def test_different_learning_rates(self):
        """Test training with different learning rates."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        learning_rates = [1e-2, 1e-3, 1e-4]

        for lr in learning_rates:
            config = ModelConfig(
                model_type="mlp",
                learning_rate=lr,
                num_epochs=10
            )
            trainer = ModelTrainer(5, config, "cpu")
            history = trainer.train(X, y, validation_split=0.2)

            # Should complete training
            assert len(history['train_mse']) > 0

    def test_different_batch_sizes(self):
        """Test training with different batch sizes."""
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        batch_sizes = [8, 16, 32]

        for batch_size in batch_sizes:
            config = ModelConfig(
                model_type="mlp",
                batch_size=batch_size,
                num_epochs=10
            )
            trainer = ModelTrainer(5, config, "cpu")
            history = trainer.train(X, y, validation_split=0.2)

            # Should complete training
            assert len(history['train_mse']) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
