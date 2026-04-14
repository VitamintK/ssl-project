"""
Configuration dataclasses for downstream task experiments.

This module centralizes all hyperparameters and settings for:
- Model training (ModelConfig)
- Task execution (TaskAConfig, TaskBConfig, TaskCConfig, TaskDConfig)

Benefits:
- Type safety and IDE autocomplete
- Easy serialization for experiment tracking
- Single source of truth for defaults
- Clear grouping of related settings
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


@dataclass
class ExperimentInfo:
    """
    Information about an experiment, used for labeling and tracking.

    Attributes:
        _label_string: The human-readable label for this experiment
        embedding_type: The type of embedding used (e.g., 'identity', 'weight_autoencoder', 'functional_encoder')
    """
    _label_string: str
    embedding_type: str

    @property
    def label_string(self) -> str:
        """Return the experiment label string."""
        return self._label_string

    def __str__(self) -> str:
        """Return the label string when converted to string."""
        return self._label_string


@dataclass
class ModelConfig:
    """
    Configuration for predictor models (neural networks or random forest).

    This unified config applies to all downstream tasks and supports three model types:
    - mlp: Multi-layer perceptron with configurable hidden layers
    - linear: Linear regression (no hidden layers)
    - random_forest: scikit-learn RandomForestRegressor
    """
    model_type: Optional[Literal["mlp", "linear", "random_forest"]] = None
    hidden_dims: Optional[list[int]] = None  # Auto-set based on model_type in __post_init__
    dropout: float = 0.0
    learning_rate: float = 1e-4
    num_epochs: int = 5000
    batch_size: int = 16
    early_stopping_patience: int = 50
    optimizer_type: Literal["adam", "adamw"] = "adam"

    def __post_init__(self):
        """Set default hidden_dims based on model_type if not explicitly provided."""
        if self.hidden_dims is None:
            if self.model_type == "mlp":
                self.hidden_dims = [128, 64, 32]
            elif self.model_type == "linear":
                self.hidden_dims = []
            else:  # random_forest
                self.hidden_dims = None

        # Validation
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1, got {self.early_stopping_patience}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass
class TaskAConfig:
    """
    Configuration for Task A: Predict payoff of agents vs fixed opponent.

    Task A evaluates how well we can predict a policy's expected payoff when
    playing against a fixed opponent (typically uniform random).
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_split: float = 0.2

    def __post_init__(self):
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")


@dataclass
class TaskBConfig:
    """
    Configuration for Task B: Predict payoff for agent vs agent matchups.

    Task B evaluates how well we can predict expected payoffs for matchups
    between two variable agents (not against a fixed opponent).
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_split: float = 0.2

    def __post_init__(self):
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")


@dataclass
class TaskCConfig:
    """
    Configuration for Task C: State-conditioned payoff prediction.

    Task C evaluates how well we can predict expected payoffs conditioned on
    the current game state (in addition to the agents playing).
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    num_states: int = 20  # Number of game states to sample for training
    max_state_depth: int = 5  # Maximum depth for state sampling
    validation_split: float = 0.2

    def __post_init__(self):
        if self.num_states < 1:
            raise ValueError(f"num_states must be >= 1, got {self.num_states}")
        if self.max_state_depth < 1:
            raise ValueError(f"max_state_depth must be >= 1, got {self.max_state_depth}")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")


@dataclass
class TaskDConfig:
    """
    Configuration for Task D: Exploitability prediction.

    Task D evaluates how well we can predict a policy's exploitability
    (the best response value against it).
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    player_id: int = 0  # Which player perspective to evaluate exploitability from
    validation_split: float = 0.2

    def __post_init__(self):
        if self.player_id not in [0, 1]:
            raise ValueError(f"player_id must be 0 or 1, got {self.player_id}")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")

@dataclass
class TaskEConfig:
    """
    Configuration for Task E: Best response learner.
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    player_id: int = 0  # Which player perspective to evaluate exploitability from
    validation_split: float = 0.2
    num_steps_per_policy_per_epoch: int = 20
    num_trajectories_per_policy_per_epoch: int = 2
    epochs: int = 5
    compare_to_control: bool = False  # If True, also train/eval with shuffled embeddings as control

    def __post_init__(self):
        if self.player_id not in [0, 1]:
            raise ValueError(f"player_id must be 0 or 1, got {self.player_id}")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")
        if self.num_steps_per_policy_per_epoch < 1:
            raise ValueError(f"num_steps_per_policy_per_epoch must be >= 1, got {self.num_steps_per_policy_per_epoch}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

def config_to_dict(config) -> dict:
    """
    Convert a config object to a dictionary for serialization.

    Args:
        config: Any config dataclass (TaskAConfig, TaskBConfig, etc.)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    return asdict(config)
