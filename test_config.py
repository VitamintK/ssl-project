"""
Unit tests for configuration dataclasses.

Tests validation logic in ModelConfig and all TaskConfig classes.
"""

import pytest
from config import (
    ModelConfig,
    TaskAConfig,
    TaskBConfig,
    TaskCConfig,
    TaskDConfig,
    config_to_dict
)


class TestModelConfig:
    """Test ModelConfig validation and defaults."""

    def test_mlp_defaults(self):
        """Test MLP model gets correct default hidden_dims."""
        config = ModelConfig(model_type="mlp")
        assert config.hidden_dims == [128, 64, 32]
        assert config.model_type == "mlp"

    def test_linear_defaults(self):
        """Test linear model gets empty hidden_dims."""
        config = ModelConfig(model_type="linear")
        assert config.hidden_dims == []
        assert config.model_type == "linear"

    def test_random_forest_defaults(self):
        """Test random forest gets None for hidden_dims."""
        config = ModelConfig(model_type="random_forest")
        assert config.hidden_dims is None
        assert config.model_type == "random_forest"

    def test_custom_hidden_dims(self):
        """Test that custom hidden_dims override defaults."""
        config = ModelConfig(model_type="mlp", hidden_dims=[256, 128])
        assert config.hidden_dims == [256, 128]

    def test_invalid_learning_rate(self):
        """Test that negative learning_rate raises ValueError."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ModelConfig(learning_rate=-0.001)

        with pytest.raises(ValueError, match="learning_rate must be positive"):
            ModelConfig(learning_rate=0)

    def test_invalid_batch_size(self):
        """Test that batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ModelConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ModelConfig(batch_size=-5)

    def test_invalid_early_stopping_patience(self):
        """Test that early_stopping_patience < 1 raises ValueError."""
        with pytest.raises(ValueError, match="early_stopping_patience must be >= 1"):
            ModelConfig(early_stopping_patience=0)

        with pytest.raises(ValueError, match="early_stopping_patience must be >= 1"):
            ModelConfig(early_stopping_patience=-10)

    def test_invalid_dropout(self):
        """Test that dropout outside [0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.0)

        with pytest.raises(ValueError, match="dropout must be in"):
            ModelConfig(dropout=1.5)

    def test_valid_dropout_boundary(self):
        """Test that dropout=0 and dropout=0.999 are valid."""
        config1 = ModelConfig(dropout=0.0)
        assert config1.dropout == 0.0

        config2 = ModelConfig(dropout=0.999)
        assert config2.dropout == 0.999

    def test_all_parameters_custom(self):
        """Test creating config with all custom parameters."""
        config = ModelConfig(
            model_type="mlp",
            hidden_dims=[512, 256, 128, 64],
            dropout=0.2,
            learning_rate=5e-4,
            num_epochs=10000,
            batch_size=64,
            early_stopping_patience=200
        )
        assert config.model_type == "mlp"
        assert config.hidden_dims == [512, 256, 128, 64]
        assert config.dropout == 0.2
        assert config.learning_rate == 5e-4
        assert config.num_epochs == 10000
        assert config.batch_size == 64
        assert config.early_stopping_patience == 200


class TestTaskAConfig:
    """Test TaskAConfig validation."""

    def test_defaults(self):
        """Test TaskAConfig uses correct defaults."""
        config = TaskAConfig()
        assert config.validation_split == 0.2
        assert isinstance(config.model_config, ModelConfig)
        assert config.model_config.model_type == "mlp"

    def test_custom_model_config(self):
        """Test TaskAConfig with custom ModelConfig."""
        model_config = ModelConfig(model_type="random_forest")
        config = TaskAConfig(model_config=model_config)
        assert config.model_config.model_type == "random_forest"
        assert config.validation_split == 0.2

    def test_invalid_validation_split_zero(self):
        """Test that validation_split=0 raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskAConfig(validation_split=0.0)

    def test_invalid_validation_split_one(self):
        """Test that validation_split=1 raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskAConfig(validation_split=1.0)

    def test_invalid_validation_split_negative(self):
        """Test that negative validation_split raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskAConfig(validation_split=-0.1)

    def test_invalid_validation_split_greater_than_one(self):
        """Test that validation_split > 1 raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskAConfig(validation_split=1.5)

    def test_valid_validation_split_boundaries(self):
        """Test validation_split just inside valid range."""
        config1 = TaskAConfig(validation_split=0.01)
        assert config1.validation_split == 0.01

        config2 = TaskAConfig(validation_split=0.99)
        assert config2.validation_split == 0.99


class TestTaskBConfig:
    """Test TaskBConfig validation."""

    def test_defaults(self):
        """Test TaskBConfig uses correct defaults."""
        config = TaskBConfig()
        assert config.validation_split == 0.2
        assert isinstance(config.model_config, ModelConfig)

    def test_invalid_validation_split(self):
        """Test that invalid validation_split raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskBConfig(validation_split=0.0)

        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskBConfig(validation_split=1.0)


class TestTaskCConfig:
    """Test TaskCConfig validation."""

    def test_defaults(self):
        """Test TaskCConfig uses correct defaults."""
        config = TaskCConfig()
        assert config.validation_split == 0.2
        assert config.num_states == 20
        assert config.max_state_depth == 5
        assert isinstance(config.model_config, ModelConfig)

    def test_custom_state_parameters(self):
        """Test TaskCConfig with custom state sampling parameters."""
        config = TaskCConfig(num_states=50, max_state_depth=10)
        assert config.num_states == 50
        assert config.max_state_depth == 10

    def test_invalid_num_states(self):
        """Test that num_states < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_states must be >= 1"):
            TaskCConfig(num_states=0)

        with pytest.raises(ValueError, match="num_states must be >= 1"):
            TaskCConfig(num_states=-5)

    def test_invalid_max_state_depth(self):
        """Test that max_state_depth < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_state_depth must be >= 1"):
            TaskCConfig(max_state_depth=0)

        with pytest.raises(ValueError, match="max_state_depth must be >= 1"):
            TaskCConfig(max_state_depth=-3)

    def test_invalid_validation_split(self):
        """Test that invalid validation_split raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskCConfig(validation_split=0.0)

    def test_valid_boundaries(self):
        """Test boundary values for state parameters."""
        config = TaskCConfig(num_states=1, max_state_depth=1)
        assert config.num_states == 1
        assert config.max_state_depth == 1


class TestTaskDConfig:
    """Test TaskDConfig validation."""

    def test_defaults(self):
        """Test TaskDConfig uses correct defaults."""
        config = TaskDConfig()
        assert config.validation_split == 0.2
        assert config.player_id == 0
        assert isinstance(config.model_config, ModelConfig)

    def test_player_id_one(self):
        """Test TaskDConfig with player_id=1."""
        config = TaskDConfig(player_id=1)
        assert config.player_id == 1

    def test_invalid_player_id_negative(self):
        """Test that negative player_id raises ValueError."""
        with pytest.raises(ValueError, match="player_id must be 0 or 1"):
            TaskDConfig(player_id=-1)

    def test_invalid_player_id_greater_than_one(self):
        """Test that player_id > 1 raises ValueError."""
        with pytest.raises(ValueError, match="player_id must be 0 or 1"):
            TaskDConfig(player_id=2)

    def test_invalid_validation_split(self):
        """Test that invalid validation_split raises ValueError."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            TaskDConfig(validation_split=1.5)


class TestConfigSerialization:
    """Test config_to_dict serialization."""

    def test_model_config_serialization(self):
        """Test ModelConfig serializes to dict correctly."""
        config = ModelConfig(
            model_type="mlp",
            hidden_dims=[128, 64],
            dropout=0.1,
            learning_rate=1e-3,
            num_epochs=1000,
            batch_size=32,
            early_stopping_patience=25
        )
        config_dict = config_to_dict(config)

        assert config_dict["model_type"] == "mlp"
        assert config_dict["hidden_dims"] == [128, 64]
        assert config_dict["dropout"] == 0.1
        assert config_dict["learning_rate"] == 1e-3
        assert config_dict["num_epochs"] == 1000
        assert config_dict["batch_size"] == 32
        assert config_dict["early_stopping_patience"] == 25

    def test_task_a_config_serialization(self):
        """Test TaskAConfig serializes to nested dict correctly."""
        model_config = ModelConfig(model_type="linear")
        config = TaskAConfig(
            model_config=model_config,
            validation_split=0.15
        )
        config_dict = config_to_dict(config)

        assert "model_config" in config_dict
        assert config_dict["model_config"]["model_type"] == "linear"
        assert config_dict["validation_split"] == 0.15

    def test_task_c_config_serialization(self):
        """Test TaskCConfig serializes all fields correctly."""
        config = TaskCConfig(
            num_states=30,
            max_state_depth=8,
            validation_split=0.25
        )
        config_dict = config_to_dict(config)

        assert config_dict["num_states"] == 30
        assert config_dict["max_state_depth"] == 8
        assert config_dict["validation_split"] == 0.25
        assert "model_config" in config_dict

    def test_task_d_config_serialization(self):
        """Test TaskDConfig serializes all fields correctly."""
        config = TaskDConfig(
            player_id=1,
            validation_split=0.3
        )
        config_dict = config_to_dict(config)

        assert config_dict["player_id"] == 1
        assert config_dict["validation_split"] == 0.3
        assert "model_config" in config_dict

    def test_serialization_roundtrip(self):
        """Test that serialization can be used for reconstruction."""
        import json

        # Create config
        original_config = TaskAConfig(
            model_config=ModelConfig(
                model_type="random_forest",
                learning_rate=5e-4
            ),
            validation_split=0.18
        )

        # Serialize to JSON
        config_dict = config_to_dict(original_config)
        json_str = json.dumps(config_dict)

        # Deserialize
        loaded_dict = json.loads(json_str)

        # Check values match
        assert loaded_dict["validation_split"] == 0.18
        assert loaded_dict["model_config"]["model_type"] == "random_forest"
        assert loaded_dict["model_config"]["learning_rate"] == 5e-4


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
