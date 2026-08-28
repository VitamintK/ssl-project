# Asymmetric APSRO for `run_neupl_v2` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `--aspro` mode to the custom NeuPL loop (`run_neupl_v2` in `psro.py`) that runs asymmetric Anytime-PSRO: one "exploited" player is trained against an opponent mixture chosen online by an abstract adversarial bandit, the other "exploiter" player best-responds, and we log the best-response value (BRV) of each exploited policy.

**Architecture:** A new `bandits.py` module provides an `AdversarialBandit` ABC with `Hedge` and `RegretMatching` implementations. `run_neupl_v2` gains `aspro`/`exploited_player`/`bandit` parameters; when `aspro` is set it branches (after the shared agent/oracle setup) into an APSRO loop that trains via bandits + pure best-response and measures BRV per policy via exact game-tree best response. No payoff matrix or LP-Nash on the aspro path.

**Tech Stack:** Python, NumPy, PyTorch (PPO agents via `iig_rl_benchmark`), OpenSpiel (`pyspiel`, `open_spiel.python`), pytest. Run everything with `uv run python3` / `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-08-27-aspro-neupl-v2-design.md`

## Global Constraints

- Run all Python via `uv run python3`; tests via `uv run pytest` (never activate the venv).
- The default (non-`--aspro`) `run_neupl_v2` behavior must be byte-for-byte unchanged.
- `--aspro` is incompatible with `use_randall_loss` and `gt_payoffs` (both assume the payoff matrix); combining them raises `ValueError`.
- Exploited player `E`: player id 0 if `exploited_player == "p1"` else 1. Exploiter `X = 1 - E`.
- Bandit reward = the exploiter X's episode payoff (= −E's payoff); the bandit maximizes it.
- One persistent bandit per exploited policy index `i` (`i = 1 … N-1`), over `i` arms `{0 … i-1}`.

## File Structure

- **Create** `bandits.py` — `AdversarialBandit` ABC, `Hedge`, `RegretMatching`, `make_bandit`.
- **Create** `tests/test_bandits.py` — bandit unit tests.
- **Create** `tests/test_aspro.py` — `best_response_value` unit test + aspro smoke test + guard test.
- **Modify** `psro.py` — add module-level `best_response_value(...)`; add `aspro`/`exploited_player`/`bandit` params + test-sizing params to `run_neupl_v2`; the aspro branch and its helpers; `__main__` argparse flags.

---

### Task 1: Adversarial bandit module

**Files:**
- Create: `bandits.py`
- Test: `tests/test_bandits.py`

**Interfaces:**
- Consumes: nothing (pure NumPy).
- Produces:
  - `class AdversarialBandit(ABC)` with `__init__(self, num_arms: int, reward_range=(-1.0, 1.0), seed=None)`, `distribution(self) -> np.ndarray` (abstract), `update(self, arm: int, reward: float) -> None` (abstract), and concrete `sample(self) -> int`.
  - `class Hedge(AdversarialBandit)` — `__init__(self, num_arms, reward_range=(-1.0, 1.0), eta=0.1, gamma=0.05, seed=None)`.
  - `class RegretMatching(AdversarialBandit)` — `__init__(self, num_arms, reward_range=(-1.0, 1.0), seed=None)`.
  - `make_bandit(name: str, num_arms: int, reward_range=(-1.0, 1.0), seed=None) -> AdversarialBandit` (`name` in `{"hedge", "rm"}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bandits.py
import numpy as np
import pytest
from bandits import AdversarialBandit, Hedge, RegretMatching, make_bandit


def _run(bandit, means, n=4000, seed=0):
    """Drive a bandit against fixed per-arm reward means (bandit feedback)."""
    rng = np.random.RandomState(seed)
    for _ in range(n):
        arm = bandit.sample()
        reward = means[arm] + 0.01 * rng.randn()
        bandit.update(arm, reward)
    return bandit.distribution()


@pytest.mark.parametrize("cls", [Hedge, RegretMatching])
def test_distribution_is_valid_simplex(cls):
    b = cls(num_arms=4, reward_range=(-1.0, 1.0), seed=1)
    for _ in range(50):
        b.update(b.sample(), 0.3)
    d = b.distribution()
    assert d.shape == (4,)
    assert np.all(d >= -1e-9)
    assert abs(d.sum() - 1.0) < 1e-6


@pytest.mark.parametrize("cls", [Hedge, RegretMatching])
def test_concentrates_on_best_arm(cls):
    means = np.array([0.9, 0.1, -0.2, 0.0])  # arm 0 is best
    d = _run(cls(num_arms=4, reward_range=(-1.0, 1.0), seed=0), means)
    assert int(np.argmax(d)) == 0
    assert d[0] > 0.5


@pytest.mark.parametrize("cls", [Hedge, RegretMatching])
def test_single_arm_is_degenerate_safe(cls):
    b = cls(num_arms=1, reward_range=(-1.0, 1.0), seed=0)
    assert b.sample() == 0
    b.update(0, 0.5)
    assert abs(b.distribution()[0] - 1.0) < 1e-9


def test_make_bandit_maps_names():
    assert isinstance(make_bandit("hedge", 3), Hedge)
    assert isinstance(make_bandit("rm", 3), RegretMatching)
    with pytest.raises(ValueError):
        make_bandit("nope", 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bandits.py -q`
Expected: FAIL (ModuleNotFoundError: No module named 'bandits').

- [ ] **Step 3: Write the implementation**

```python
# bandits.py
"""Abstract adversarial bandits (online learners) over a fixed set of arms.

Used by APSRO to choose, online, the opponent mixture that a policy trains against.
Feedback is partial (bandit): only the sampled arm's reward is observed. Rewards are
normalized from ``reward_range`` to [0, 1] internally so step sizes transfer across games.
"""
from abc import ABC, abstractmethod
import numpy as np


class AdversarialBandit(ABC):
    def __init__(self, num_arms: int, reward_range=(-1.0, 1.0), seed=None):
        if num_arms < 1:
            raise ValueError(f"num_arms must be >= 1, got {num_arms}")
        self.num_arms = num_arms
        self._lo, self._hi = float(reward_range[0]), float(reward_range[1])
        self._span = max(self._hi - self._lo, 1e-12)
        self.rng = np.random.RandomState(seed)

    def _normalize(self, reward: float) -> float:
        return float(np.clip((reward - self._lo) / self._span, 0.0, 1.0))

    @abstractmethod
    def distribution(self) -> np.ndarray:
        """Current mixture over arms: shape (num_arms,), non-negative, sums to 1."""

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """Observe ``reward`` (raw, in reward_range) for the sampled ``arm``."""

    def sample(self) -> int:
        return int(self.rng.choice(self.num_arms, p=self.distribution()))


class Hedge(AdversarialBandit):
    """Exponential weights with importance-weighted (Exp3-style) bandit updates."""

    def __init__(self, num_arms, reward_range=(-1.0, 1.0), eta=0.1, gamma=0.05, seed=None):
        super().__init__(num_arms, reward_range, seed)
        self.eta = eta
        self.gamma = gamma if num_arms > 1 else 0.0
        self.log_weights = np.zeros(num_arms)

    def distribution(self) -> np.ndarray:
        w = self.log_weights - self.log_weights.max()
        p = np.exp(w)
        p /= p.sum()
        return (1.0 - self.gamma) * p + self.gamma / self.num_arms

    def update(self, arm: int, reward: float) -> None:
        p = self.distribution()
        est = self._normalize(reward) / max(p[arm], 1e-12)  # unbiased reward estimate
        self.log_weights[arm] += self.eta * est


class RegretMatching(AdversarialBandit):
    """Regret matching (RM+) with mean-tracked counterfactual utilities (bandit feedback).

    Each arm's counterfactual utility is its running empirical mean reward, with optimistic
    initialization (an unsampled arm is assumed max reward 1.0 so it gets explored).
    Cumulative regret accumulates (mean_i - realized_reward) each round, clamped at 0 (RM+);
    the mixture is proportional to positive cumulative regret, plus a small uniform
    exploration floor gamma. (Naive importance-weighted bandit RM was tried first but its
    1/p variance made it lock onto lucky arms — see tests/test_bandits.py.)
    """

    def __init__(self, num_arms, reward_range=(-1.0, 1.0), gamma=0.02, seed=None):
        super().__init__(num_arms, reward_range, seed)
        self.gamma = gamma if num_arms > 1 else 0.0
        self.cum_regret = np.zeros(num_arms)
        self.sum_reward = np.zeros(num_arms)
        self.count = np.zeros(num_arms)

    def _means(self) -> np.ndarray:
        return np.where(self.count > 0, self.sum_reward / np.maximum(self.count, 1.0), 1.0)

    def distribution(self) -> np.ndarray:
        pos = np.maximum(self.cum_regret, 0.0)
        s = pos.sum()
        base = np.full(self.num_arms, 1.0 / self.num_arms) if s <= 1e-12 else pos / s
        return (1.0 - self.gamma) * base + self.gamma / self.num_arms

    def update(self, arm: int, reward: float) -> None:
        r = self._normalize(reward)
        self.sum_reward[arm] += r
        self.count[arm] += 1.0
        self.cum_regret = np.maximum(self.cum_regret + (self._means() - r), 0.0)


def make_bandit(name: str, num_arms: int, reward_range=(-1.0, 1.0), seed=None) -> AdversarialBandit:
    name = name.lower()
    if name == "hedge":
        return Hedge(num_arms, reward_range=reward_range, seed=seed)
    if name == "rm":
        return RegretMatching(num_arms, reward_range=reward_range, seed=seed)
    raise ValueError(f"unknown bandit {name!r}; expected 'hedge' or 'rm'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bandits.py -q`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add bandits.py tests/test_bandits.py
git commit -m "feat(aspro): abstract adversarial bandit module (Hedge, RegretMatching)"
```

---

### Task 2: Exact best-response-value helper

**Files:**
- Modify: `psro.py` (add a module-level function near the other top-level helpers, e.g. just above `def run_neupl_v2`)
- Test: `tests/test_aspro.py`

**Interfaces:**
- Consumes: `pyspiel`, `open_spiel.python.policy`, `open_spiel.python.algorithms.best_response`.
- Produces: `best_response_value(game, fixed_policy, fixed_player: int, br_player: int) -> float` — the value `br_player` achieves by exactly best-responding to `fixed_policy` (which plays `fixed_player`). `fixed_policy` must expose `action_probabilities(state)` (OpenSpiel Policy interface); the NeuPL PPO agent policies do.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aspro.py
import numpy as np
import pyspiel
from open_spiel.python.policy import UniformRandomPolicy
from psro import best_response_value


def test_brv_of_uniform_p0_on_kuhn():
    game = pyspiel.load_game("kuhn_poker")
    u = UniformRandomPolicy(game)
    # Best response value for player 1 against a uniform player 0 (known ~0.4167).
    brv = best_response_value(game, u, fixed_player=0, br_player=1)
    assert game.min_utility() <= brv <= game.max_utility()
    assert abs(brv - 0.41666667) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aspro.py::test_brv_of_uniform_p0_on_kuhn -q`
Expected: FAIL (ImportError: cannot import name 'best_response_value').

- [ ] **Step 3: Write the implementation**

```python
# psro.py  (module-level, above run_neupl_v2)
def best_response_value(game, fixed_policy, fixed_player, br_player):
    """Value ``br_player`` gets by exactly best-responding to ``fixed_policy``.

    ``fixed_policy`` plays ``fixed_player``; only its states matter. Requires
    ``fixed_policy.action_probabilities(state)`` (OpenSpiel Policy interface).
    """
    from open_spiel.python import policy as policy_lib
    from open_spiel.python.algorithms import best_response as _best_response

    profile = policy_lib.TabularPolicy(game)  # opponent (br_player) rows are ignored
    for state in profile.states:
        if state.current_player() != fixed_player:
            continue
        probs = fixed_policy.action_probabilities(state)
        row = profile.action_probability_array[profile.state_index(state)]
        row[:] = 0.0
        for action, p in probs.items():
            row[action] = p
    responder = _best_response.BestResponsePolicy(game, br_player, profile)
    return float(responder.value(game.new_initial_state()))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_aspro.py::test_brv_of_uniform_p0_on_kuhn -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add psro.py tests/test_aspro.py
git commit -m "feat(aspro): exact best-response-value helper"
```

---

### Task 3: Test-sizing params + aspro guard on `run_neupl_v2`

Make the run loop's sizes overridable (so the aspro smoke test is fast) and add the incompatibility guard. Both are additive and preserve current behavior.

**Files:**
- Modify: `psro.py` (`run_neupl_v2` signature + the `num_pols_sampled` / `total_episodes_per_policy` / `num_iterations` locals + a guard near the top of the function)
- Test: `tests/test_aspro.py`

**Interfaces:**
- Produces: `run_neupl_v2(game_name="kuhn_poker", use_randall_loss=False, T=None, debug=False, gt_payoffs=False, save_logs=False, aspro=False, exploited_player="p1", bandit="hedge", num_iterations=680, num_pols_sampled=8, total_episodes_per_policy=400, expl_check_episode_interval=None)`. When `expl_check_episode_interval` is `None` it falls back to the config value (unchanged behavior); tests pass a small value to force a measurement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aspro.py  (append)
import pytest
from psro import run_neupl_v2


def test_aspro_rejects_randall_loss():
    with pytest.raises(ValueError):
        run_neupl_v2(game_name="kuhn_poker", aspro=True, use_randall_loss=True)


def test_aspro_rejects_gt_payoffs():
    with pytest.raises(ValueError):
        run_neupl_v2(game_name="kuhn_poker", aspro=True, gt_payoffs=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aspro.py -k "rejects" -q`
Expected: FAIL (TypeError: unexpected keyword 'aspro', or no ValueError raised).

- [ ] **Step 3: Update the signature, promote the size locals to params, add the guard**

Change the signature (`psro.py` around line 76):

```python
def run_neupl_v2(game_name: str = 'kuhn_poker', use_randall_loss: bool = False, T: int = None,
                 debug: bool = False, gt_payoffs: bool = False, save_logs: bool = False,
                 aspro: bool = False, exploited_player: str = 'p1', bandit: str = 'hedge',
                 num_iterations: int = 680, num_pols_sampled: int = 8,
                 total_episodes_per_policy: int = 400, expl_check_episode_interval: int = None):
```

Add the guard immediately after `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` (around line 89):

```python
    if aspro and (use_randall_loss or gt_payoffs):
        raise ValueError("aspro is incompatible with use_randall_loss / gt_payoffs "
                         "(they require the payoff matrix, which aspro does not build)")
    if exploited_player not in ("p1", "p2"):
        raise ValueError(f"exploited_player must be 'p1' or 'p2', got {exploited_player!r}")
```

Delete the local assignments that these params now supersede:
- `num_pols_sampled = 8   # outer iters: ...` (around line 112)
- `total_episodes_per_policy = 400 ...` (around line 113)
- `num_iterations = 680  # ...` (around line 551)

(The parameter defaults carry the identical values, so behavior is unchanged.)

Make `expl_check_episode_interval` respect the new param — change the config read (around line 115) from:

```python
    expl_check_episode_interval = args.algorithm.expl_check_episode_interval
```
to:
```python
    if expl_check_episode_interval is None:
        expl_check_episode_interval = args.algorithm.expl_check_episode_interval
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aspro.py -k "rejects" -q`
Expected: PASS. The guard raises before any heavy setup.

- [ ] **Step 5: Commit**

```bash
git add psro.py tests/test_aspro.py
git commit -m "feat(aspro): run_neupl_v2 test-sizing params and aspro/randall guard"
```

---

### Task 4: Asymmetric APSRO loop + argparse

Add the aspro branch to `run_neupl_v2`: index-0 setup, per-index bandits, exploited/exploiter training, BRV measurement + plot, the aspro main loop, and CLI flags.

**Files:**
- Modify: `psro.py` (index-0 setup block ~line 197; add aspro branch before the existing `# ── main loop ──` section ~line 615; `__main__` argparse ~line 1135)
- Test: `tests/test_aspro.py`

**Interfaces:**
- Consumes: `make_bandit` (Task 1), `best_response_value` (Task 2), and existing `run_neupl_v2` closures/locals: `agents`, `oracle`, `env`, `num_actions`, `state`, `_force_learn`, `_save_checkpoint`, `N`, `experiment_dir`, `stats_path`, `logger`, `run_start`, `expl_check_episode_interval`, `num_pols_sampled`, `total_episodes_per_policy`, `num_iterations`, `game`, LR helpers `base_lr/final_lr/lr_anneal_iters/_set_lr`, `rl_policy`.
- Produces: an `--aspro` run that writes `stats.jsonl` records with a `brv_per_policy` list and a `brv.png` plot in `experiment_dir`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_aspro.py  (append)
import json, os, glob


def _latest_experiment_dir():
    dirs = glob.glob(os.path.join("results", "*", "neupl", "*", "*", "kuhn_poker", "*"))
    return max(dirs, key=os.path.getmtime)


@pytest.mark.parametrize("exploited,bandit", [("p1", "hedge"), ("p2", "rm")])
def test_aspro_smoke_runs_and_logs_brv(exploited, bandit):
    run_neupl_v2(
        game_name="kuhn_poker", aspro=True, exploited_player=exploited, bandit=bandit,
        num_iterations=2, num_pols_sampled=2, total_episodes_per_policy=5, T=2,
        expl_check_episode_interval=1,  # force a BRV measurement every iteration
    )
    exp = _latest_experiment_dir()
    with open(os.path.join(exp, "stats.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert records, "no stats recorded"
    assert "brv_per_policy" in records[-1]
    assert len(records[-1]["brv_per_policy"]) >= 1
    assert os.path.exists(os.path.join(exp, "brv.png"))
    assert os.path.exists(os.path.join(exp, "policy0_ckpt.pt"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aspro.py -k smoke -q`
Expected: FAIL (aspro path not implemented — either falls through to the symmetric loop and never writes `brv_per_policy`, or errors).

- [ ] **Step 3a: Make index-0 setup aspro-aware**

Replace the two unconditional index-0 overrides (`psro.py` ~line 197-198):

```python
    agents[0][0] = rl_policy.UniformRandomAgentPolicy(env, 0, num_actions=num_actions)
    agents[1][0] = rl_policy.UniformRandomAgentPolicy(env, 1, num_actions=num_actions)
```

with:

```python
    if aspro:
        E = 0 if exploited_player == 'p1' else 1   # exploited (uniform anchor at index 0)
        X = 1 - E                                  # exploiter (index 0 stays trainable BR)
        agents[E][0] = rl_policy.UniformRandomAgentPolicy(env, E, num_actions=num_actions)
    else:
        agents[0][0] = rl_policy.UniformRandomAgentPolicy(env, 0, num_actions=num_actions)
        agents[1][0] = rl_policy.UniformRandomAgentPolicy(env, 1, num_actions=num_actions)
```

- [ ] **Step 3b: Add the aspro branch + helpers**

**Insertion point (important):** insert this block **after** the `_set_payoff_matrix_update_rate` definition (`psro.py` ~line 636) and **before** the symmetric loop's `run_start = time.perf_counter()` (~line 638). It must come after the LR helpers `base_lr` / `final_lr` / `lr_anneal_iters` / `_set_lr` (defined ~617-633, i.e. *below* the `# ── main loop ──` comment) because the aspro loop calls them; and after `if T is None: T = N * 15` (~line 549) so `T` is set. Placing it here also lets it see every earlier closure (`_force_learn`, `_save_checkpoint`, `state`, `_episode_agents`, etc.); the `return` at its end exits before the symmetric loop runs.

```python
    if aspro:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reward_range = (game.min_utility(), game.max_utility())
        # One persistent bandit per exploited policy index i, over arms {0..i-1}.
        bandits = {i: make_bandit(bandit, num_arms=i, reward_range=reward_range, seed=i)
                   for i in range(1, N)}

        def _episode_agents(train_player, train_pol, opp_pol):
            """Order agents by player id for oracle.sample_episode."""
            pair = [None, None]
            pair[train_player] = train_pol
            pair[1 - train_player] = opp_pol
            return pair

        def _train_exploited(k):
            """Train E's policies 1..k against their bandit mixture over X's {0..i-1}."""
            for _ in range(num_pols_sampled):
                i = min(random.randint(1, k + 1), N - 1)
                training_pol = agents[E][i]
                training_pol.unfreeze()
                b = bandits[i]
                for _ in range(total_episodes_per_policy):
                    j = b.sample()
                    opp = agents[X][j]
                    opp.freeze()
                    rewards = oracle.sample_episode(
                        None, _episode_agents(E, training_pol, opp), is_evaluation=False)
                    b.update(j, float(rewards[X]))   # reward = exploiter's payoff
                    state.episodes_played += 1
                    state.episodes_training += 1
                _force_learn(training_pol, E)

        def _train_exploiter(k):
            """Train X's policies 0..k as a pure best response to E's same-index policy."""
            for _ in range(num_pols_sampled):
                i = min(random.randint(0, k + 1), N - 1)
                training_pol = agents[X][i]
                training_pol.unfreeze()
                opp = agents[E][i]
                opp.freeze()
                for _ in range(total_episodes_per_policy):
                    oracle.sample_episode(
                        None, _episode_agents(X, training_pol, opp), is_evaluation=False)
                    state.episodes_played += 1
                    state.episodes_training += 1
                _force_learn(training_pol, X)

        def _plot_brv(records):
            if not records:
                return
            xs = [r["episodes"] for r in records]
            n_pol = max(len(r["brv_per_policy"]) for r in records)
            fig, ax = plt.subplots(figsize=(8, 5))
            cmap = plt.get_cmap("viridis")
            for idx in range(n_pol):
                ys = [(r["brv_per_policy"][idx] if idx < len(r["brv_per_policy"]) else np.nan)
                      for r in records]
                ax.plot(xs, ys, color=cmap(idx / max(n_pol - 1, 1)), lw=1, label=f"E_{idx}")
            ax.set_xlabel("episodes"); ax.set_ylabel("BRV of exploited policy")
            ax.set_title(f"APSRO BRV per exploited policy (E=p{E+1})")
            ax.grid(True, alpha=0.3)
            if n_pol <= 12:
                ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(experiment_dir, "brv.png"), dpi=120)
            plt.close(fig)

        def _measure_brv(k):
            if state.episodes_played - state.episodes_at_last_expl_check < expl_check_episode_interval:
                return
            for pid in range(2):
                for i in range(k + 1):
                    agents[pid][i].freeze()
            brvs = [best_response_value(game, agents[E][i], fixed_player=E, br_player=X)
                    for i in range(k + 1)]
            state.episodes_at_last_expl_check = state.episodes_played
            record = {
                "episodes": state.episodes_played,
                "episodes_training": state.episodes_training,
                "walltime": time.perf_counter() - run_start,
                "brv_per_policy": brvs,
                "mean_brv": float(np.mean(brvs)),
            }
            state.stats.append(record)
            with open(stats_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info("  BRV per exploited policy (k=%s): %s",
                        k, "  ".join(f"E_{i}={v:.4f}" for i, v in enumerate(brvs)))
            _plot_brv(state.stats)

        run_start = time.perf_counter()
        for it in range(1, num_iterations + 1):
            k = min(N - 1, max(1, int(np.ceil(it / T * (N - 1)))))
            t = min(it - 1, lr_anneal_iters - 1) / max(lr_anneal_iters - 1, 1)
            _set_lr(base_lr + (final_lr - base_lr) * t)
            logger.info("  [aspro] iteration %s/%s  k=%s/%s", it, num_iterations, k, N - 1)
            _train_exploited(k)
            _train_exploiter(k)
            _measure_brv(k)
            _save_checkpoint()
        return
```

Note: `T` defaults to `N * 15` (already set just above the symmetric main loop at `if T is None: T = N * 15`); ensure that default assignment happens **before** this aspro block. If the `if T is None:` line currently sits below this insertion point, move it up so `T` is set before the aspro loop uses it.

- [ ] **Step 3c: Add argparse flags**

In `__main__` (`psro.py` ~line 1135), after the existing `--neupl_v2` argument add:

```python
    parser.add_argument('--aspro', action='store_true', help='Asymmetric APSRO in neupl_v2')
    parser.add_argument('--exploited_player', choices=['p1', 'p2'], default='p1',
                        help='Which player APSRO optimizes/measures')
    parser.add_argument('--bandit', choices=['hedge', 'rm'], default='hedge',
                        help='Adversarial bandit for the exploited player')
```

Then forward the new flags in the dispatch (`psro.py` ~line 1149-1151). Replace:

```python
    if args.neupl_v2:
        run_neupl_v2(game_name, use_randall_loss=args.use_randall_loss, T=args.T,
                     debug=args.debug, gt_payoffs=args.gt_payoffs, save_logs=args.save_logs)
```

with:

```python
    if args.neupl_v2:
        run_neupl_v2(game_name, use_randall_loss=args.use_randall_loss, T=args.T,
                     debug=args.debug, gt_payoffs=args.gt_payoffs, save_logs=args.save_logs,
                     aspro=args.aspro, exploited_player=args.exploited_player, bandit=args.bandit)
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_aspro.py -k smoke -q`
Expected: PASS for both `("p1","hedge")` and `("p2","rm")` — a 2-iteration kuhn run writes `stats.jsonl` with `brv_per_policy`, `brv.png`, and `policy0_ckpt.pt`.

- [ ] **Step 5: Run the full test module + confirm the symmetric path still imports/parses**

Run: `uv run pytest tests/test_aspro.py tests/test_bandits.py -q`
Expected: PASS.
Run: `uv run python3 -c "import ast; ast.parse(open('psro.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add psro.py tests/test_aspro.py
git commit -m "feat(aspro): asymmetric APSRO loop, BRV logging/plot, and CLI flags"
```

---

## Self-Review

**Spec coverage:**
- `--aspro` / `--exploited_player` / `--bandit` flags → Task 4 (argparse) + Task 3 (params). ✓
- E_0 uniform, X_i BR to E_i, E_i vs bandit mixture over {X_0..X_{i-1}} → Task 4 (`_train_exploited` / `_train_exploiter`, index-0 setup). ✓
- Reward = exploiter's payoff, bandit maximizes → Task 4 (`b.update(j, rewards[X])`) + Task 1. ✓
- One persistent bandit per exploited index over {0..i-1} → Task 4 (`bandits` dict). ✓
- Abstract bandit (Hedge, RM) → Task 1. ✓
- BRV per exploited policy logged + plotted; no payoff matrix/Nash → Task 4 (`_measure_brv`, `_plot_brv`) + Task 2. ✓
- Guard against randall/gt_payoffs → Task 3. ✓
- Symmetric path unchanged → Tasks 3/4 are additive; index-0 change branches on `aspro`. ✓
- Critic-not-opponent-conditioned limitation → recorded in spec + `TODO(claude)` already in `ppo.py` (no code task needed). ✓

**Placeholder scan:** No TBD/TODO-in-plan; all steps carry runnable code or exact edits. The only `TODO(...)` string is the intentional pre-existing `ppo.py` comment. ✓

**Type consistency:** `make_bandit(name, num_arms, reward_range, seed)`, `AdversarialBandit.sample()->int` / `update(arm, reward)` / `distribution()->np.ndarray`, and `best_response_value(game, fixed_policy, fixed_player, br_player)->float` are used with identical signatures in Tasks 1/2/4. `E`/`X` are ints; `rewards[X]` indexes the `sample_episode` return (a 2-vector of per-player payoffs). ✓

## Notes for the executor

- `oracle.sample_episode(None, agents_pair, is_evaluation=False)` plays one episode, buffers PPO transitions, and returns a 2-element array of per-player cumulative payoffs. PPO learns in batches as the buffer fills; `_force_learn` flushes the remainder per policy. The bandit updates per episode; the two cadences are independent (spec §Algorithm).
- The NeuPL PPO agent policies implement OpenSpiel's `action_probabilities` (the symmetric path already feeds them to `PolicyAggregator`), so `best_response_value` works on `agents[E][i]` directly.
- Keep `num_iterations`/`num_pols_sampled`/`total_episodes_per_policy` defaults exactly `680`/`8`/`400` so the production run is unchanged; only tests pass smaller values.
