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
        task_id: Single uppercase letter identifying the task ('A', 'B', 'C', 'D', 'E')
    """
    _label_string: str
    embedding_type: str
    task_id: str = ""

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
    max_batch_size: int = 20
    num_trajectories_per_policy_per_epoch: int = 2
    epochs: int = 5
    compare_to_control: bool = False  # If True, also train/eval with shuffled embeddings as control

    def __post_init__(self):
        if self.player_id not in [0, 1]:
            raise ValueError(f"player_id must be 0 or 1, got {self.player_id}")
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")
        if self.max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {self.max_batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")

@dataclass
class TaskFConfig:
    """
    Configuration for Task F: find equilibria by descent-ascent in embedding space.

    Trains a differentiable value function over embedding pairs, runs two-timescale
    gradient descent-ascent in embedding space, decodes the result to policies, and
    evaluates NashConv. Optionally posttrains the value function on visited pairs.
    """
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(model_type="mlp"))
    validation_split: float = 0.2
    # descent-ascent
    outer_steps: int = 200
    inner_steps: int = 5
    lr_p1: float = 1e-2
    lr_p2: float = 5e-2
    num_restarts: int = 4
    bound_embeddings: bool = True
    # posttraining
    posttrain: bool = False
    posttrain_rounds: int = 2
    posttrain_budget: int = 256
    posttrain_epochs: int = 200
    posttrain_lr: float = 1e-3
    # eval
    nashconv_baseline_samples: int = 8

    def __post_init__(self):
        if self.model_config.model_type not in ("mlp", "linear"):
            raise ValueError(
                "Task F requires a differentiable value function "
                f"(model_type must be 'mlp' or 'linear'), got {self.model_config.model_type!r}."
            )
        if not 0 < self.validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {self.validation_split}")
        for name in ("outer_steps", "inner_steps", "num_restarts",
                     "posttrain_rounds", "posttrain_budget", "posttrain_epochs",
                     "nashconv_baseline_samples"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        for name in ("lr_p1", "lr_p2", "posttrain_lr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

def config_to_dict(config) -> dict:
    """
    Convert a config object to a dictionary for serialization.

    Args:
        config: Any config dataclass (TaskAConfig, TaskBConfig, etc.)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    return asdict(config)
