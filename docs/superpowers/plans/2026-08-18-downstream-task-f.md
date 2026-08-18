# Downstream Task F Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a downstream task ("Task F") that trains a value function over policy-embedding pairs, finds an approximate equilibrium by two-timescale gradient descent–ascent in embedding space, decodes the result back into real policies, and measures its NashConv.

**Architecture:** Reuse the Task B `PayoffPredictor` as the (differentiable) value function `V(e_p1,e_p2)`. A new `EmbeddingEquilibriumSolver` optimizes `V` in embedding space. Per-source **decoder callables** (`embedding → Policy`) turn the found embeddings back into policies; NashConv is computed with OpenSpiel best-response. An optional posttraining loop recomputes value targets for visited embedding pairs and continue-trains `V`.

**Tech Stack:** Python, PyTorch, OpenSpiel (`pyspiel`, `open_spiel.python`), `iig_rl_benchmark` PPO agents, pytest. Run everything with `uv run python3` / `uv run pytest` (never activate the venv).

**Spec:** `docs/superpowers/specs/2026-08-18-downstream-task-f-design.md`

## Global Constraints

- Repository is the flat `conditional-br-2` layout: modules live at repo root (`config.py`, `downstream.py`, `tasks.py`, `main.py`, `psro.py`, `weight_autoencoder.py`, `functional_autoencoder.py`, `utils.py`). No `src/` package.
- Run all Python via `uv run python3` and tests via `uv run pytest` — do NOT activate the venv manually.
- The Task F value function MUST be differentiable: `model_type ∈ {"mlp","linear"}`. `random_forest` is rejected.
- `V(e_p1,e_p2)` predicts **player 0's** expected payoff. Zero-sum convention: P1 (player 0) maximizes `V`; P2 (player 1) minimizes `V`.
- Embedding vectors from the pool are 1-D numpy arrays of shape `(D,)`. NeuPL agent conditioning expects a `(1, D)` tensor — unsqueeze before use.
- PPO actor weight vectors are produced by `ppo_agent_to_vector(agent)` = `parameters_to_vector(agent.actor.parameters())`; decoding is the inverse via `torch.nn.utils.vector_to_parameters` into a `copy.deepcopy` of a template agent.
- `PPOAgent(num_actions, observation_shape, device, layer_init, hidden_size)`; PSRO/NeuPL/random pools in this project use `hidden_size=256`.
- `PPOAgentPolicy(game, agent, player_id, use_observation=False)` and `PPONeuplAgentPolicy(game, agent, player_id, use_observation=False, embedding=<(1,D) tensor>)` implement `.action_probabilities(state) -> {action: prob}`.
- Task functions (`run_task_*`) return a metrics dict; the caller (`_run_experiment`) calls `register_result`. Do NOT register inside `run_task_f`.
- Put new tests in `tests/` at repo root (create the dir if absent). Commit after each task.

---

### Task 1: `TaskFConfig`

**Files:**
- Modify: `config.py` (add dataclass after `TaskEConfig`)
- Test: `tests/test_task_f_config.py`

**Interfaces:**
- Consumes: `ModelConfig` (existing, `config.py`).
- Produces: `TaskFConfig` dataclass with fields `model_config: ModelConfig`, `validation_split: float=0.2`, `outer_steps: int=200`, `inner_steps: int=5`, `lr_p1: float=1e-2`, `lr_p2: float=5e-2`, `num_restarts: int=4`, `bound_embeddings: bool=True`, `posttrain: bool=False`, `posttrain_rounds: int=2`, `posttrain_budget: int=256`, `posttrain_epochs: int=200`, `posttrain_lr: float=1e-3`, `nashconv_baseline_samples: int=8`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task_f_config.py
import pytest
from config import TaskFConfig, ModelConfig


def test_defaults_are_valid():
    cfg = TaskFConfig(model_config=ModelConfig(model_type="mlp"))
    assert cfg.outer_steps == 200
    assert cfg.inner_steps == 5
    assert cfg.bound_embeddings is True
    assert cfg.posttrain is False


def test_rejects_random_forest():
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="random_forest"))


def test_rejects_bad_hyperparams():
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), outer_steps=0)
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), inner_steps=0)
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), num_restarts=0)
    with pytest.raises(ValueError):
        TaskFConfig(model_config=ModelConfig(model_type="mlp"), validation_split=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_task_f_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'TaskFConfig'`.

- [ ] **Step 3: Write minimal implementation**

Add to `config.py` (after the `TaskEConfig` dataclass; `field` and `Literal` are already imported there):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_task_f_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_task_f_config.py
git commit -m "feat(task-f): add TaskFConfig with differentiable-model validation"
```

---

### Task 2: Weight-vector decode helper + autoencoder `get_decoder` + identity decoder

**Files:**
- Modify: `weight_autoencoder.py` (add `vector_to_ppo_agent`, `WeightAutoencoder.get_decoder`)
- Modify: `functional_autoencoder.py` (add `FunctionalEncoderAdapter.get_decoder`)
- Test: `tests/test_decoders.py`

**Interfaces:**
- Consumes: `ppo_agent_to_vector` (existing, `weight_autoencoder.py`); `Autoencoder` with `.encoder`/`.decoder` (existing); `PPOAgentPolicy` (existing, `utils.py`); `WeightAutoencoder.autoencoder` and `FunctionalEncoderAdapter.model` (existing attrs, each an `Autoencoder`).
- Produces:
  - `vector_to_ppo_agent(template_agent, vector) -> PPOAgent` (in `weight_autoencoder.py`) — deep-copies `template_agent`, loads `vector` into `agent.actor`.
  - `WeightAutoencoder.get_decoder(game, player_id, template_agent, device="cpu") -> Callable[[np.ndarray|torch.Tensor], PPOAgentPolicy]`.
  - `FunctionalEncoderAdapter.get_decoder(game, player_id, template_agent, device="cpu") -> Callable[[np.ndarray|torch.Tensor], PPOAgentPolicy]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decoders.py
import numpy as np
import pyspiel
from utils import make_diverse_random_kuhn_poker_layer_init
from iig_rl_benchmark.algorithms.ppo import ppo
from weight_autoencoder import (
    WeightAutoencoder, AutoencoderConfig, ppo_agent_to_vector, vector_to_ppo_agent,
)


def _kuhn_agent(game):
    li = make_diverse_random_kuhn_poker_layer_init(game)
    return ppo.PPOAgent(game.num_distinct_actions(),
                        game.information_state_tensor_shape(), "cpu", li, 256)


def test_vector_to_ppo_agent_roundtrip():
    game = pyspiel.load_game("kuhn_poker")
    agent = _kuhn_agent(game)
    vec = ppo_agent_to_vector(agent)
    rebuilt = vector_to_ppo_agent(agent, vec)
    assert np.allclose(ppo_agent_to_vector(rebuilt).detach().numpy(),
                       vec.detach().numpy(), atol=1e-6)


def test_weight_autoencoder_get_decoder_returns_policy():
    game = pyspiel.load_game("kuhn_poker")
    agents = [_kuhn_agent(game) for _ in range(40)]
    ae = WeightAutoencoder(
        AutoencoderConfig(hidden_dims=(64,), bottleneck_dim=16, epochs=2,
                          batch_size=8, device="cpu"),
        agents, ppo_agent_to_vector)
    ae.train()
    encode = ae.get_encoder(device="cpu")
    decode = ae.get_decoder(game, player_id=0, template_agent=agents[0], device="cpu")
    emb = encode(agents[0])
    policy = decode(emb)
    state = game.new_initial_state()
    probs = policy.action_probabilities(state)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert set(probs.keys()) == set(state.legal_actions())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_decoders.py -v`
Expected: FAIL with `ImportError: cannot import name 'vector_to_ppo_agent'`.

- [ ] **Step 3: Write minimal implementation**

In `weight_autoencoder.py`, add after `ppo_agent_to_vector`:

```python
import copy


def vector_to_ppo_agent(template_agent, vector):
    """Inverse of ppo_agent_to_vector: load `vector` into a copy of template_agent's actor."""
    agent = copy.deepcopy(template_agent)
    device = next(agent.actor.parameters()).device
    vec = vector if isinstance(vector, torch.Tensor) else torch.tensor(vector)
    vec = vec.detach().float().reshape(-1).to(device)
    torch.nn.utils.vector_to_parameters(vec, agent.actor.parameters())
    return agent
```

Add as a method on `WeightAutoencoder`:

```python
    def get_decoder(self, game, player_id: int, template_agent, device: str = "cpu"):
        """Return decode(embedding) -> PPOAgentPolicy (inverse of get_encoder)."""
        from utils import PPOAgentPolicy
        self.autoencoder.eval()
        decoder = self.autoencoder.decoder.to(device)

        def decode(embedding):
            with torch.no_grad():
                z = embedding if isinstance(embedding, torch.Tensor) else torch.tensor(embedding)
                z = z.float().to(device)
                if z.ndim == 1:
                    z = z.unsqueeze(0)
                weight_vector = decoder(z).squeeze(0)
            agent = vector_to_ppo_agent(template_agent, weight_vector)
            return PPOAgentPolicy(game, agent, player_id, False)

        return decode
```

In `functional_autoencoder.py`, add as a method on `FunctionalEncoderAdapter` (mirrors weight AE but uses `self.model`):

```python
    def get_decoder(self, game, player_id: int, template_agent, device: str = "cpu"):
        """Return decode(embedding) -> PPOAgentPolicy (inverse of get_encoder)."""
        from utils import PPOAgentPolicy
        from weight_autoencoder import vector_to_ppo_agent
        self.model.eval()
        decoder = self.model.decoder.to(device)

        def decode(embedding):
            with torch.no_grad():
                z = embedding if isinstance(embedding, torch.Tensor) else torch.tensor(embedding)
                z = z.float().to(device)
                if z.ndim == 1:
                    z = z.unsqueeze(0)
                weight_vector = decoder(z).squeeze(0)
            agent = vector_to_ppo_agent(template_agent, weight_vector)
            return PPOAgentPolicy(game, agent, player_id, False)

        return decode
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_decoders.py -v`
Expected: PASS (2 tests). (Warnings about `nn.init.uniform`/`pokerkit` are pre-existing and harmless.)

- [ ] **Step 5: Commit**

```bash
git add weight_autoencoder.py functional_autoencoder.py tests/test_decoders.py
git commit -m "feat(task-f): add weight/functional autoencoder decoders and vector_to_ppo_agent"
```

---

### Task 3: `neupl_decoder`

**Files:**
- Modify: `psro.py` (add top-level `neupl_decoder`)
- Test: `tests/test_neupl_decoder.py`

**Interfaces:**
- Consumes: `PPONeuplAgentPolicy` (existing, `utils.py`); a loaded NeuPL agent reachable as `some_neupl_policy._ppo_agent`.
- Produces: `neupl_decoder(game, agent, embedding, player_id, use_observation=False) -> PPONeuplAgentPolicy`. Accepts `(D,)` or `(1,D)` embedding; unsqueezes to `(1,D)` internally.

- [ ] **Step 1: Write the failing test**

This test loads a real NeuPL run if one is present; otherwise it is skipped (no checkpoints in CI). It always exercises the shape/return-contract via a lightweight stub agent.

```python
# tests/test_neupl_decoder.py
import torch
import pyspiel
from psro import neupl_decoder
from utils import PPONeuplAgentPolicy


class _StubNeuplAgent(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self._p = torch.nn.Parameter(torch.zeros(1))

    def get_action(self, info_state, embedding=None, legal_actions_mask=None):
        n = self.num_actions
        probs = torch.ones(1, n) / n
        if legal_actions_mask is not None:
            probs = probs * legal_actions_mask
            probs = probs / probs.sum()
        return None, None, None, probs


def test_neupl_decoder_returns_valid_policy():
    game = pyspiel.load_game("kuhn_poker")
    agent = _StubNeuplAgent(game.num_distinct_actions())
    emb = torch.zeros(8)  # (D,)
    policy = neupl_decoder(game, agent, emb, player_id=0)
    assert isinstance(policy, PPONeuplAgentPolicy)
    assert policy._embedding.ndim == 2 and policy._embedding.shape[0] == 1
    state = game.new_initial_state()
    probs = policy.action_probabilities(state)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert set(probs.keys()) == set(state.legal_actions())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_neupl_decoder.py -v`
Expected: FAIL with `ImportError: cannot import name 'neupl_decoder'`.

- [ ] **Step 3: Write minimal implementation**

Add to `psro.py` (top level; `torch` is already imported there):

```python
def neupl_decoder(game, agent, embedding, player_id: int, use_observation: bool = False):
    """Embedding -> policy for NeuPL: condition `agent` on `embedding`.

    `agent` is a loaded PPOConditionedOnPolicyRepresentationAgent (e.g. obtained from an
    existing policy via `policy._ppo_agent`). `embedding` may be shape (D,) or (1, D);
    NeuPL conditioning expects (1, D).
    """
    from utils import PPONeuplAgentPolicy
    emb = embedding if isinstance(embedding, torch.Tensor) else torch.tensor(embedding, dtype=torch.float32)
    emb = emb.float()
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    return PPONeuplAgentPolicy(game, agent, player_id,
                               use_observation=use_observation, embedding=emb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_neupl_decoder.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add psro.py tests/test_neupl_decoder.py
git commit -m "feat(task-f): add neupl_decoder (embedding -> PPONeuplAgentPolicy)"
```

---

### Task 4: NashConv evaluation helper

**Files:**
- Modify: `downstream.py` (add top-level `compute_nash_conv`)
- Test: `tests/test_nash_conv.py`

**Interfaces:**
- Consumes: OpenSpiel `open_spiel.python.policy.TabularPolicy`, `open_spiel.python.algorithms.exploitability.nash_conv` (verified available); any two policies with `.action_probabilities(state) -> {action: prob}`.
- Produces: `compute_nash_conv(game, p1_policy, p2_policy) -> float`. Builds a joint tabular policy (player-0 rows from `p1_policy`, player-1 rows from `p2_policy`) and returns its NashConv (>= 0; 0 at equilibrium).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nash_conv.py
import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability
from downstream import compute_nash_conv


def test_matches_openspiel_on_uniform():
    game = pyspiel.load_game("kuhn_poker")
    uni = policy_lib.UniformRandomPolicy(game)
    got = compute_nash_conv(game, uni, uni)
    want = exploitability.nash_conv(game, policy_lib.UniformRandomPolicy(game))
    assert abs(got - want) < 1e-9
    assert got > 0.0  # uniform is exploitable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nash_conv.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_nash_conv'`.

- [ ] **Step 3: Write minimal implementation**

Add to `downstream.py` (top level):

```python
def compute_nash_conv(game, p1_policy, p2_policy) -> float:
    """NashConv of the joint profile (p1_policy plays player 0, p2_policy plays player 1)."""
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import exploitability as _exploitability

    joint = policy_lib.TabularPolicy(game)
    for state in joint.states:
        pid = state.current_player()
        src = p1_policy if pid == 0 else p2_policy
        probs = src.action_probabilities(state)
        row = joint.action_probability_array[joint.state_index(state)]
        row[:] = 0.0
        for action, p in probs.items():
            row[action] = p
    return float(_exploitability.nash_conv(game, joint))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_nash_conv.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add downstream.py tests/test_nash_conv.py
git commit -m "feat(task-f): add compute_nash_conv joint-profile helper"
```

---

### Task 5: `EmbeddingEquilibriumSolver`

**Files:**
- Modify: `downstream.py` (add `EmbeddingEquilibriumSolver` + `SolveResult`)
- Test: `tests/test_embedding_solver.py`

**Interfaces:**
- Consumes: a torch `nn.Module` value model taking input `concat([e_p1, e_p2])` of dim `D1+D2` and returning shape `(batch,)`; pool arrays `pool_p1: np.ndarray (N1,D1)`, `pool_p2: np.ndarray (N2,D2)`.
- Produces:
  - `SolveResult` dataclass: `e_p1: np.ndarray (D1,)`, `e_p2: np.ndarray (D2,)`, `value: float`, `visited: list[tuple[np.ndarray, np.ndarray]]`.
  - `EmbeddingEquilibriumSolver(value_model, pool_p1, pool_p2, config, device="cpu")` with `.solve(init_p1=None, init_p2=None) -> SolveResult` and `.solve_best_of_restarts(score_fn) -> SolveResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embedding_solver.py
import numpy as np
import torch
from config import TaskFConfig, ModelConfig
from downstream import EmbeddingEquilibriumSolver


class _Bilinear(torch.nn.Module):
    # V(e1,e2) = e1 . e2 ; saddle of a max_e1 min_e2 game at e1=e2=0 (with bounding).
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        e1, e2 = x[..., : self.d], x[..., self.d:]
        return (e1 * e2).sum(-1)


def test_bounding_keeps_iterates_in_pool_range():
    d = 3
    pool = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cfg = TaskFConfig(model_config=ModelConfig(model_type="mlp"),
                      outer_steps=50, inner_steps=5, lr_p1=0.5, lr_p2=0.5,
                      num_restarts=1, bound_embeddings=True)
    solver = EmbeddingEquilibriumSolver(_Bilinear(d), pool, pool, cfg, device="cpu")
    res = solver.solve(init_p1=np.array([0.9, 0.9, 0.9]), init_p2=np.array([-0.9, -0.9, -0.9]))
    assert np.all(res.e_p1 <= 1.0 + 1e-5) and np.all(res.e_p1 >= -1.0 - 1e-5)
    assert np.all(res.e_p2 <= 1.0 + 1e-5) and np.all(res.e_p2 >= -1.0 - 1e-5)
    assert len(res.visited) == 50


def test_solver_moves_toward_saddle():
    d = 2
    pool = np.array([[-2.0, -2.0], [2.0, 2.0]])
    cfg = TaskFConfig(model_config=ModelConfig(model_type="mlp"),
                      outer_steps=300, inner_steps=10, lr_p1=0.02, lr_p2=0.1,
                      num_restarts=1, bound_embeddings=True)
    solver = EmbeddingEquilibriumSolver(_Bilinear(d), pool, pool, cfg, device="cpu")
    res = solver.solve(init_p1=np.array([1.5, 1.5]), init_p2=np.array([1.5, 1.5]))
    # near the saddle the value magnitude should be small
    assert abs(res.value) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_solver.py -v`
Expected: FAIL with `ImportError: cannot import name 'EmbeddingEquilibriumSolver'`.

- [ ] **Step 3: Write minimal implementation**

Add to `downstream.py` (top level; `torch`, `np`, `dataclass` may need importing — `torch` and `numpy as np` are already imported; add `from dataclasses import dataclass` near the other imports if not present):

```python
@dataclass
class SolveResult:
    e_p1: np.ndarray
    e_p2: np.ndarray
    value: float
    visited: list  # list[tuple[np.ndarray, np.ndarray]]


class EmbeddingEquilibriumSolver:
    """Two-timescale gradient descent-ascent on a frozen value model, in embedding space.

    Player 0 (P1) maximizes V; player 1 (P2) minimizes V.
    """

    def __init__(self, value_model, pool_p1, pool_p2, config, device="cpu"):
        self.model = value_model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.config = config
        self.pool_p1 = np.asarray(pool_p1, dtype=np.float32)
        self.pool_p2 = np.asarray(pool_p2, dtype=np.float32)
        self.d1 = self.pool_p1.shape[1]
        self.d2 = self.pool_p2.shape[1]
        self.lo1 = torch.tensor(self.pool_p1.min(0), device=device)
        self.hi1 = torch.tensor(self.pool_p1.max(0), device=device)
        self.lo2 = torch.tensor(self.pool_p2.min(0), device=device)
        self.hi2 = torch.tensor(self.pool_p2.max(0), device=device)

    def _value(self, e1, e2):
        return self.model(torch.cat([e1, e2]).unsqueeze(0)).squeeze(0)

    def _clamp(self, e, lo, hi):
        if self.config.bound_embeddings:
            return torch.max(torch.min(e, hi), lo)
        return e

    def solve(self, init_p1=None, init_p2=None) -> SolveResult:
        cfg = self.config
        rng = np.random
        if init_p1 is None:
            init_p1 = self.pool_p1[rng.randint(len(self.pool_p1))]
        if init_p2 is None:
            init_p2 = self.pool_p2[rng.randint(len(self.pool_p2))]
        e1 = torch.tensor(np.asarray(init_p1, np.float32), device=self.device, requires_grad=True)
        e2 = torch.tensor(np.asarray(init_p2, np.float32), device=self.device, requires_grad=True)
        visited = []
        for _ in range(cfg.outer_steps):
            # inner loop: P2 minimizes V (fast)
            for _ in range(cfg.inner_steps):
                v = self._value(e1, e2)
                (g2,) = torch.autograd.grad(v, e2)
                with torch.no_grad():
                    e2 = self._clamp(e2 - cfg.lr_p2 * g2, self.lo2, self.hi2)
                e2.requires_grad_(True)
            # outer step: P1 maximizes V (slow)
            v = self._value(e1, e2)
            (g1,) = torch.autograd.grad(v, e1)
            with torch.no_grad():
                e1 = self._clamp(e1 + cfg.lr_p1 * g1, self.lo1, self.hi1)
            e1.requires_grad_(True)
            visited.append((e1.detach().cpu().numpy().copy(),
                            e2.detach().cpu().numpy().copy()))
        with torch.no_grad():
            final_v = float(self._value(e1, e2))
        return SolveResult(e1.detach().cpu().numpy(), e2.detach().cpu().numpy(),
                           final_v, visited)

    def solve_best_of_restarts(self, score_fn) -> SolveResult:
        """Run num_restarts solves; return the result minimizing score_fn(result) (lower=better)."""
        best, best_score = None, float("inf")
        for _ in range(self.config.num_restarts):
            res = self.solve()
            s = score_fn(res)
            if s < best_score:
                best, best_score = res, s
        return best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_solver.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add downstream.py tests/test_embedding_solver.py
git commit -m "feat(task-f): add EmbeddingEquilibriumSolver (two-timescale descent-ascent)"
```

---

### Task 6: `run_task_f` (pretrain + solve + eval, no posttraining)

**Files:**
- Modify: `tasks.py` (add `run_task_f`)
- Modify: `downstream.py` (add `continue_train_value_model` helper — used here for nothing yet, but defined so Task 7 only edits `run_task_f`)
- Test: `tests/test_run_task_f.py`

**Interfaces:**
- Consumes: `PayoffPredictor` (existing), `EmbeddingEquilibriumSolver`, `SolveResult`, `compute_nash_conv` (Tasks 4–5); `TaskFConfig` (Task 1); `ExperimentInfo`, `config_to_dict` (existing, `config.py`). Decoders are callables `embedding -> Policy`.
- Produces:
  - `continue_train_value_model(model, X, y, epochs, lr, device="cpu") -> None` (in `downstream.py`): standard MSE SGD updating `model` in place on `X (n,D)`, `y (n,)` numpy arrays.
  - `run_task_f(game, p1_policies, p1_embeddings, p2_policies, p2_embeddings, p1_decoder, p2_decoder, config, experiment_info, device="cpu") -> dict` with keys `nashconv`, `nashconv_baseline`, `value_at_star`, `sampled_payoff_at_star`, `val_metrics`, `config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_task_f.py
import numpy as np
import pyspiel
from config import TaskFConfig, ModelConfig
from utils import make_diverse_random_kuhn_poker_layer_init, PPOAgentPolicy
from iig_rl_benchmark.algorithms.ppo import ppo
from weight_autoencoder import ppo_agent_to_vector, vector_to_ppo_agent
from config import ExperimentInfo
from tasks import run_task_f


def _agents(game, n):
    li = make_diverse_random_kuhn_poker_layer_init(game)
    return [ppo.PPOAgent(game.num_distinct_actions(),
                         game.information_state_tensor_shape(), "cpu", li, 256)
            for _ in range(n)]


def test_run_task_f_smoke_identity_source():
    game = pyspiel.load_game("kuhn_poker")
    p1_agents, p2_agents = _agents(game, 6), _agents(game, 6)
    p1_pol = [PPOAgentPolicy(game, a, 0, False) for a in p1_agents]
    p2_pol = [PPOAgentPolicy(game, a, 1, False) for a in p2_agents]
    p1_emb = [ppo_agent_to_vector(a).detach().numpy() for a in p1_agents]
    p2_emb = [ppo_agent_to_vector(a).detach().numpy() for a in p2_agents]
    # identity decoders: embedding is the actor weight-vector
    p1_dec = lambda e, t=p1_agents[0]: PPOAgentPolicy(game, vector_to_ppo_agent(t, e), 0, False)
    p2_dec = lambda e, t=p2_agents[0]: PPOAgentPolicy(game, vector_to_ppo_agent(t, e), 1, False)
    cfg = TaskFConfig(
        model_config=ModelConfig(model_type="mlp", num_epochs=30, early_stopping_patience=10),
        outer_steps=5, inner_steps=2, num_restarts=1, nashconv_baseline_samples=2)
    info = ExperimentInfo("kuhn identity Task F", embedding_type="identity", task_id="F")
    out = run_task_f(game, p1_pol, p1_emb, p2_pol, p2_emb, p1_dec, p2_dec, cfg, info, "cpu")
    assert "nashconv" in out and out["nashconv"] >= 0.0
    assert "nashconv_baseline" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_task_f.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_task_f'`.

- [ ] **Step 3: Write minimal implementation**

Add to `downstream.py`:

```python
def continue_train_value_model(model, X, y, epochs, lr, device="cpu"):
    """In-place MSE SGD on a torch value model. X: (n, D), y: (n,)."""
    model.to(device).train()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
```

Add to `tasks.py` (import additions at top: `from config import TaskFConfig`; `from downstream import PayoffPredictor, EmbeddingEquilibriumSolver, compute_nash_conv`; `from utils import get_expected_payoffs`):

```python
def run_task_f(
    game,
    p1_policies, p1_embeddings,
    p2_policies, p2_embeddings,
    p1_decoder, p2_decoder,
    config: TaskFConfig,
    experiment_info: ExperimentInfo,
    device: str = "cpu",
) -> dict:
    """Task F: find an equilibrium by descent-ascent in embedding space; report NashConv."""
    logger.info(f"Running Task F: {experiment_info.label_string}")

    # 1. Pretrain the value function V(e_p1, e_p2) (Task B).
    predictor = PayoffPredictor(
        game=game, p1_policies=p1_policies, p2_policies=p2_policies,
        p1_embeddings=p1_embeddings, p2_embeddings=p2_embeddings,
        model_config=config.model_config, device=device)
    predictor.compute_ground_truth_payoffs()
    predictor.train_with_agent_level_split(config.validation_split)
    val_metrics = predictor.evaluate(eval_set="val")

    pool_p1 = np.array(p1_embeddings)
    pool_p2 = np.array(p2_embeddings)

    # 2. Solve for an equilibrium, keeping the restart with lowest decoded NashConv.
    solver = EmbeddingEquilibriumSolver(
        predictor.trainer.model, pool_p1, pool_p2, config, device=device)

    def score(res):
        return compute_nash_conv(game, p1_decoder(res.e_p1), p2_decoder(res.e_p2))

    result = solver.solve_best_of_restarts(score)

    # 3. Evaluate the recovered profile.
    p1_star = p1_decoder(result.e_p1)
    p2_star = p2_decoder(result.e_p2)
    nashconv = compute_nash_conv(game, p1_star, p2_star)
    sampled_payoff = get_expected_payoffs(game, p1_star, p2_star)

    # random-embedding-pair baseline
    baseline_vals = []
    for _ in range(config.nashconv_baseline_samples):
        e1 = pool_p1[np.random.randint(len(pool_p1))]
        e2 = pool_p2[np.random.randint(len(pool_p2))]
        baseline_vals.append(compute_nash_conv(game, p1_decoder(e1), p2_decoder(e2)))
    nashconv_baseline = float(np.mean(baseline_vals))

    logger.info(f"Task F NashConv: {nashconv:.6f}  baseline: {nashconv_baseline:.6f}")

    return {
        "nashconv": float(nashconv),
        "nashconv_baseline": nashconv_baseline,
        "value_at_star": float(result.value),
        "sampled_payoff_at_star": float(sampled_payoff),
        "val_metrics": val_metrics,
        "config": config_to_dict(config),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_task_f.py -v`
Expected: PASS (1 test). May take ~1–2 min (payoff sampling on kuhn).

- [ ] **Step 5: Commit**

```bash
git add tasks.py downstream.py tests/test_run_task_f.py
git commit -m "feat(task-f): add run_task_f (pretrain V, solve, NashConv eval)"
```

---

### Task 7: Posttraining loop in `run_task_f`

**Files:**
- Modify: `tasks.py` (`run_task_f` — add posttraining branch)
- Test: `tests/test_run_task_f.py` (add a posttraining test)

**Interfaces:**
- Consumes: `continue_train_value_model` (Task 6), `SolveResult.visited`.
- Produces: when `config.posttrain=True`, `run_task_f` continue-trains `V` on decoded-target rows from visited pairs and re-solves; adds `posttrain_rounds` int to the returned dict.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_run_task_f.py

def test_run_task_f_posttraining_runs():
    import pyspiel
    from config import TaskFConfig, ModelConfig, ExperimentInfo
    from utils import make_diverse_random_kuhn_poker_layer_init, PPOAgentPolicy
    from iig_rl_benchmark.algorithms.ppo import ppo
    from weight_autoencoder import ppo_agent_to_vector, vector_to_ppo_agent
    from tasks import run_task_f
    game = pyspiel.load_game("kuhn_poker")
    li = make_diverse_random_kuhn_poker_layer_init(game)
    mk = lambda: ppo.PPOAgent(game.num_distinct_actions(),
                              game.information_state_tensor_shape(), "cpu", li, 256)
    p1_agents = [mk() for _ in range(6)]
    p2_agents = [mk() for _ in range(6)]
    p1_pol = [PPOAgentPolicy(game, a, 0, False) for a in p1_agents]
    p2_pol = [PPOAgentPolicy(game, a, 1, False) for a in p2_agents]
    p1_emb = [ppo_agent_to_vector(a).detach().numpy() for a in p1_agents]
    p2_emb = [ppo_agent_to_vector(a).detach().numpy() for a in p2_agents]
    p1_dec = lambda e, t=p1_agents[0]: PPOAgentPolicy(game, vector_to_ppo_agent(t, e), 0, False)
    p2_dec = lambda e, t=p2_agents[0]: PPOAgentPolicy(game, vector_to_ppo_agent(t, e), 1, False)
    cfg = TaskFConfig(
        model_config=ModelConfig(model_type="mlp", num_epochs=30, early_stopping_patience=10),
        outer_steps=4, inner_steps=2, num_restarts=1, nashconv_baseline_samples=1,
        posttrain=True, posttrain_rounds=1, posttrain_budget=4, posttrain_epochs=5)
    info = ExperimentInfo("kuhn identity Task F posttrain", embedding_type="identity", task_id="F")
    out = run_task_f(game, p1_pol, p1_emb, p2_pol, p2_emb, p1_dec, p2_dec, cfg, info, "cpu")
    assert out["posttrain_rounds"] == 1
    assert "nashconv" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_task_f.py::test_run_task_f_posttraining_runs -v`
Expected: FAIL with `KeyError: 'posttrain_rounds'`.

- [ ] **Step 3: Write minimal implementation**

In `tasks.py` `run_task_f`, replace the single `result = solver.solve_best_of_restarts(score)` line with a solve-then-posttrain loop, and add `posttrain_rounds` to the returned dict:

```python
    result = solver.solve_best_of_restarts(score)

    if config.posttrain:
        for _ in range(config.posttrain_rounds):
            # gather decoded value targets for a budget-subsample of visited pairs
            visited = result.visited
            if len(visited) > config.posttrain_budget:
                idx = np.random.choice(len(visited), config.posttrain_budget, replace=False)
                visited = [visited[i] for i in idx]
            X_new, y_new = [], []
            for e1, e2 in visited:
                target = get_expected_payoffs(game, p1_decoder(e1), p2_decoder(e2))
                X_new.append(np.concatenate([e1, e2]))
                y_new.append(target)
            continue_train_value_model(
                predictor.trainer.model, np.array(X_new), np.array(y_new),
                epochs=config.posttrain_epochs, lr=config.posttrain_lr, device=device)
            result = solver.solve_best_of_restarts(score)
```

Add the import in `tasks.py`: `from downstream import continue_train_value_model`. Add `"posttrain_rounds": config.posttrain_rounds if config.posttrain else 0` to the returned dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_task_f.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tasks.py tests/test_run_task_f.py
git commit -m "feat(task-f): add posttraining loop to run_task_f"
```

---

### Task 8: `main.py` wiring (encoders from loaders, `_run_experiment` task 'f', `RUN_TASK_F`)

**Files:**
- Modify: `main.py` (`get_policies_and_embeddings`, `get_policies_and_embeddings2`, `_run_experiment`, `__main__`)
- Test: `tests/test_task_f_wiring.py`

**Interfaces:**
- Consumes: everything above; existing loaders and `_run_experiment` (`main.py`).
- Produces:
  - `get_policies_and_embeddings(...) -> (policies, embeddings, encoder)` where `encoder` is the `WeightAutoencoder` (has `get_decoder`).
  - `get_policies_and_embeddings2(...) -> (policies, embeddings, None)` (identity: no encoder object).
  - `_run_experiment` handles `task == 'f'`: loads both players (like `task == 'b'`), builds `p1_decoder`/`p2_decoder`, and calls `run_task_f`.
  - `RUN_TASK_F` flag + spec building in `__main__`.

**Note on decoder construction per source (inside `_run_experiment`):**
- `reconstruction-autoencoder`: `encoder.get_decoder(game, player_id, template_agent, device)` where `encoder` is returned by the loader and `template_agent` is any agent from that player's pool (reconstruct one via `vector_to_ppo_agent` from an identity embedding is NOT available here; instead keep a reference agent — see Step 3).
- `identity`: `lambda e: PPOAgentPolicy(game, vector_to_ppo_agent(template_agent, e), player_id, False)`.
- `neupl`: `lambda e: neupl_decoder(game, neupl_agent, e, player_id)` where `neupl_agent = policies[0]._ppo_agent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_task_f_wiring.py
from config import ExperimentInfo
from main import _run_experiment


def test_run_experiment_task_f_random_identity():
    spec = {
        "source": "random", "game_name": "kuhn_poker", "player_id": 0,
        "embedding_type": "identity", "N_random": 6, "task": "f",
        "predictor_type": "mlp", "device": "cpu", "policy_device": "cpu",
        "task_f_overrides": {"outer_steps": 4, "inner_steps": 2, "num_restarts": 1,
                             "nashconv_baseline_samples": 1, "model_num_epochs": 30},
        "experiment_info": ExperimentInfo("kuhn ppo random identity Task F",
                                          embedding_type="identity", task_id="F"),
    }
    exp_info, result = _run_experiment(spec)
    assert result["nashconv"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_task_f_wiring.py -v`
Expected: FAIL (either `Unknown task: f` from `_run_experiment`, or a `KeyError`).

- [ ] **Step 3: Write minimal implementation**

1. Update loader return values. In `get_policies_and_embeddings` (`main.py`), return the encoder and keep one template agent reachable:

```python
    encoder_fn = weight_autoencoder.get_encoder(device=device)
    embeddings = [encoder_fn(agent).detach().cpu().numpy() for agent in downstream_ppo_agents]
    policies = [PPOAgentPolicy(game, agent, player_id, False) for agent in downstream_ppo_agents]
    return policies, embeddings, weight_autoencoder, downstream_ppo_agents[0]
```

and `get_policies_and_embeddings2`:

```python
    embeddings = [ppo_agent_to_vector(agent).detach().cpu().numpy() for agent in ppo_agents]
    policies = [PPOAgentPolicy(game, agent, player_id, False) for agent in ppo_agents]
    return policies, embeddings, None, ppo_agents[0]
```

Update every existing call site of these two functions (currently in `_run_experiment` for tasks a/d/e/b) to unpack four values: `policies, embeddings, encoder, template_agent = get_policies_and_embeddings(...)` (and `...2(...)`). For the p2 loads in the `task == 'b'` block, unpack into `p2_..., p2_encoder, p2_template` similarly.

2. In `_run_experiment`, treat `task == 'f'` like `task == 'b'` for loading the opponent (reuse the existing `task in ('b',)` opponent-loading branch — change it to `task in ('b', 'f')`), capturing encoders and template agents for both players. Then add the dispatch branch before the final `else`:

```python
    elif task == 'f':
        from config import TaskFConfig, ModelConfig
        from psro import neupl_decoder
        from weight_autoencoder import vector_to_ppo_agent
        from tasks import run_task_f
        from utils import PPOAgentPolicy

        def make_decoder(src_encoder, src_template, src_policies, pid):
            if source == 'neupl':
                neupl_agent = src_policies[0]._ppo_agent
                return lambda e: neupl_decoder(game, neupl_agent, e, pid)
            if src_encoder is not None:  # reconstruction-autoencoder
                return src_encoder.get_decoder(game, pid, src_template, device)
            # identity
            return lambda e: PPOAgentPolicy(game, vector_to_ppo_agent(src_template, e), pid, False)

        p1_decoder = make_decoder(encoder, template_agent, policies, 0)
        p2_decoder = make_decoder(p2_encoder, p2_template, p2_policies, 1)

        ov = spec.get('task_f_overrides', {})
        model_cfg = ModelConfig(model_type=spec['predictor_type'],
                                num_epochs=ov.get('model_num_epochs', 5000))
        config = TaskFConfig(
            model_config=model_cfg,
            outer_steps=ov.get('outer_steps', 200),
            inner_steps=ov.get('inner_steps', 5),
            num_restarts=ov.get('num_restarts', 4),
            bound_embeddings=ov.get('bound_embeddings', True),
            posttrain=ov.get('posttrain', False),
            nashconv_baseline_samples=ov.get('nashconv_baseline_samples', 8))
        result = run_task_f(game=game, p1_policies=policies, p1_embeddings=embeddings,
                            p2_policies=p2_policies, p2_embeddings=p2_embeddings,
                            p1_decoder=p1_decoder, p2_decoder=p2_decoder,
                            config=config, experiment_info=experiment_info, device=device)
```

(Ensure the loader unpacking in `_run_experiment` sets `encoder`, `template_agent`, `p2_encoder`, `p2_template` in all source branches; for `neupl`, set `encoder = p2_encoder = None`, `template_agent = p2_template = None`, since neupl uses `policies[0]._ppo_agent`.)

3. In `__main__`, add `RUN_TASK_F = False` beside the other flags, and in the spec-building loops add Task F specs mirroring the `RUN_TASK_B` blocks (both players are loaded inside `_run_experiment`, so a Task F spec looks like a Task B spec with `'task': 'f'` and `predictor_type` `'mlp'`). Example for the PSRO block:

```python
            if RUN_TASK_F:
                for emb_type in ['reconstruction-autoencoder', 'identity']:
                    specs.append({'source': 'psro', 'game_name': game_name, 'player_id': 0,
                                  'embedding_type': emb_type, 'task': 'f', 'predictor_type': 'mlp',
                                  'experiment_info': ExperimentInfo(
                                      f'{game_short_name} psro {emb_type} Task F',
                                      embedding_type=emb_type, task_id='F')})
```

Add analogous `RUN_TASK_F` blocks in the `neupl` (predictor_type `'mlp'`) and `random` sections.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_task_f_wiring.py -v`
Expected: PASS (1 test).

Then run the whole suite to confirm no regressions in the loader-unpacking changes:

Run: `uv run pytest tests/ -v`
Expected: all Task F tests PASS; pre-existing tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_task_f_wiring.py
git commit -m "feat(task-f): wire Task F into _run_experiment and __main__ (RUN_TASK_F)"
```

---

## Self-Review

**Spec coverage:**
- Decoders (weight/functional `get_decoder`, `neupl_decoder`, identity) → Tasks 2, 3, 8.
- Differentiable value function / pretraining (Task B reuse) → Task 6.
- Two-timescale descent-ascent + bounding + restarts → Task 5.
- Posttraining toggle → Task 7.
- NashConv eval + random baseline + value/sampled sanity → Tasks 4, 6.
- `TaskFConfig` (mlp/linear only) → Task 1.
- `main.py` wiring: loaders return encoders, `_run_experiment` task 'f', `RUN_TASK_F` → Task 8.
- Deferred items (PBS, zero-shot BR, time-averaging, MP/RPS) → intentionally out of scope, no tasks. ✓

**Placeholder scan:** No TBD/TODO; every code step has real code. ✓

**Type consistency:** `SolveResult(e_p1, e_p2, value, visited)` used consistently in Tasks 5–7; `run_task_f` signature identical in Tasks 6–8; decoder callables are `embedding -> Policy` throughout; loaders return a 4-tuple consistently after Task 8. ✓

**Risks flagged in spec:** autoencoder decode fidelity (Task 2 round-trip test surfaces it), NashConv acceptance of neural policies (Task 4 builds a tabular joint policy, avoiding the issue), descent-ascent cycling (best-of-restarts by NashConv in Task 6). ✓
