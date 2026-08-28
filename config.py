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
    value_function_model_config: ModelConfig = field(default_factory=lambda: ModelConfig(model_type="mlp"))
    value_function_validation_split: float = 0.2
    # value-function pretraining: if set, sample this many (P1, P2) pairs instead of the full N x M grid
    value_function_num_pairs: Optional[int] = None
    # payoff targets for value-function training/posttraining: True = exact tree traversal
    # (policy_value, deterministic), False = Monte-Carlo sampled (150 episodes, noisy).
    exact_payoff_targets: bool = False
    # descent-ascent
    # optimizing_player is the slow / committed (outer-loop) player whose exploitability we
    # track; the other player is the fast (inner-loop) best-responder. "p1" (player 0,
    # maximizes V) is the default; "p2" (player 1, minimizes V) swaps the two roles.
    optimizing_player: Literal["p1", "p2"] = "p1"
    # how each player's per-step update is computed:
    #   "gradient" (descent-ascent on V, the default),
    #   "cem"  (gradient-free cross-entropy method -- sample a population, keep the top-k
    #           elites, move to their mean),
    #   "mppi" (like cem, but softmax-weight ALL samples by value instead of a hard top-k).
    optimizer: Literal["gradient", "cem", "mppi"] = "gradient"
    outer_steps: int = 200
    inner_steps: int = 5
    lr_exploited: float = 2e-2  # step size for the exploited (committed / outer) player [gradient]
    lr_exploiter: float = 5e-2  # step size for the exploiter (fast / inner) player [gradient]
    # CEM / MPPI hyperparameters (used only when optimizer in {"cem", "mppi"}):
    cem_population: int = 64      # candidates sampled per step
    cem_elite_frac: float = 0.125  # fraction kept as elites [cem only]
    # sampling std of the Gaussian around the current embedding, per role (shared by cem/mppi):
    cem_noise_exploited: float = 0.1  # for the exploited (committed / outer) player
    cem_noise_exploiter: float = 0.1  # for the exploiter (fast / inner) player
    cem_step_size: float = 1.0    # interpolation toward the (weighted) mean (1.0 = jump fully)
    mppi_temperature: float = 1.0  # softmax temperature for MPPI weighting (low -> argmax-like)
    num_restarts: int = 4
    bound_embeddings: bool = True
    # Mahalanobis trust region: after each descent-ascent step, project each embedding back
    # onto the ball (in the pool's Mahalanobis metric) whose radius is the trust_region_quantile
    # of the pool points' own Mahalanobis distances. Keeps the search in-distribution so the
    # value function is not evaluated far outside its training support. Off by default.
    trust_region: bool = False
    trust_region_quantile: float = 1.0
    # multiplies the quantile-derived radius; > 1 lets the search extend beyond the pool.
    trust_region_scale: float = 1.0
    # online posttraining: co-train the value function *during* the descent-ascent solve.
    # After each embedding step, the new (e1, e2) is decoded, its true payoff computed, and
    # one SGD step is taken on the value model toward that target -- mixed with a random
    # anchor minibatch of the original pool pairs to avoid forgetting the pool fit. The
    # model is reset to its pretrained state before each restart.
    posttrain: bool = False
    posttrain_lr: float = 2e-4
    posttrain_anchor_batch: int = 32  # pool pairs mixed into each online update (0 = none)
    # eval
    nashconv_baseline_samples: int = 8
    # diagnostics: decode policies each inner step to log true value / exploitability
    # for the solve-curve plots. Expensive (decode + NashConv per inner step), so
    # off by default; enable only for diagnostic runs.
    log_real_values: bool = False
    # value-function caching: if set, the trained value function is saved to / loaded from
    # this directory, keyed by experiment label (which identifies the checkpoint) plus a
    # config + embedding-dimensionality fingerprint. The sampled embedding *values* are not
    # part of the key, so a fresh NeuPL draw from the same checkpoint still hits the cache.
    # Skips ground-truth payoff computation + model training on a hit. None disables caching.
    value_function_cache_dir: Optional[str] = None
    # if True, ignore any existing cached value function (always retrain from scratch); the
    # freshly trained model is still saved to the cache dir afterward (refreshing it).
    value_function_cache_ignore: bool = False
    # exploitability landscape: after each solve, plot P1's true exploitability over a 2D
    # slice of embedding space. Expensive (grid_size**2 decode + best-response evals), so
    # off by default. dim_selection is "path" (dims the P1 path varied most) or "random".
    plot_landscape: bool = False
    landscape_dim_selection: Literal["path", "random"] = "path"
    landscape_grid_size: int = 25

    def __post_init__(self):
        if self.value_function_model_config.model_type not in ("mlp", "linear"):
            raise ValueError(
                "Task F requires a differentiable value function "
                f"(model_type must be 'mlp' or 'linear'), got {self.value_function_model_config.model_type!r}."
            )
        if not 0 < self.value_function_validation_split < 1:
            raise ValueError(f"value_function_validation_split must be in (0, 1), got {self.validation_split}")
        for name in ("outer_steps", "inner_steps", "num_restarts",
                     "nashconv_baseline_samples"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.posttrain_anchor_batch < 0:
            raise ValueError(f"posttrain_anchor_batch must be >= 0, got {self.posttrain_anchor_batch}")
        if self.value_function_num_pairs is not None and self.value_function_num_pairs < 1:
            raise ValueError(f"value_function_num_pairs must be >= 1 or None, got {self.num_pairs}")
        for name in ("lr_exploited", "lr_exploiter", "posttrain_lr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.landscape_grid_size < 2:
            raise ValueError(f"landscape_grid_size must be >= 2, got {self.landscape_grid_size}")
        if not 0 < self.trust_region_quantile <= 1:
            raise ValueError(f"trust_region_quantile must be in (0, 1], got {self.trust_region_quantile}")
        if self.trust_region_scale <= 0:
            raise ValueError(f"trust_region_scale must be positive, got {self.trust_region_scale}")
        if self.optimizing_player not in ("p1", "p2"):
            raise ValueError(f"optimizing_player must be 'p1' or 'p2', got {self.optimizing_player!r}")
        if self.optimizer not in ("gradient", "cem", "mppi"):
            raise ValueError(f"optimizer must be 'gradient', 'cem', or 'mppi', got {self.optimizer!r}")
        if self.cem_population < 2:
            raise ValueError(f"cem_population must be >= 2, got {self.cem_population}")
        if not 0 < self.cem_elite_frac <= 1:
            raise ValueError(f"cem_elite_frac must be in (0, 1], got {self.cem_elite_frac}")
        for name in ("cem_noise_exploited", "cem_noise_exploiter", "mppi_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0 < self.cem_step_size <= 1:
            raise ValueError(f"cem_step_size must be in (0, 1], got {self.cem_step_size}")

def config_to_dict(config) -> dict:
    """
    Convert a config object to a dictionary for serialization.

    Args:
        config: Any config dataclass (TaskAConfig, TaskBConfig, etc.)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    return asdict(config)
