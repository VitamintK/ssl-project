# Asymmetric APSRO for `run_neupl_v2` — Design

## Summary

Add an `--aspro` mode to the custom NeuPL loop (`run_neupl_v2` in `psro.py`) that runs
**asymmetric Anytime-PSRO** instead of the current symmetric, Nash-over-payoff-matrix
training. One player is the **exploited** player (the one we optimize and measure); the
other is the **exploiter**. The exploited player's policies are trained against an
opponent mixture chosen online by an **abstract adversarial bandit** (Hedge, regret
matching, …) rather than an LP Nash. Measurement logs the **best-response value (BRV) of
each exploited policy**.

## Terminology

- **E** — the exploited player (chosen via `--exploited_player`, `p1` or `p2`).
- **X** — the exploiter (the other player).
- `E_i`, `X_i` — player E's / X's policy at index `i` (`i = 0 … N-1`).
- **BRV(π)** — best-response value against a fixed policy π: the payoff an exact
  best-responding opponent achieves. Lower ⇒ π is less exploitable.

## Goals

- `--aspro` selects asymmetric APSRO in `run_neupl_v2`; default (flag absent) is today's
  symmetric loop, unchanged.
- `--exploited_player={p1,p2}` names E; the other player is X.
- `--bandit={hedge,rm}` selects the adversarial bandit implementation.
- The adversarial bandit is abstract so new algorithms can be added without touching the
  training loop.

## Non-goals

- No change to the existing symmetric `run_neupl_v2` path.
- No payoff matrix / LP-Nash in the `--aspro` path.
- `--aspro` is incompatible with `use_randall_loss` and `gt_payoffs` (both assume the
  payoff matrix); combining them raises a clear error.

## Algorithm

Population, reusing `run_neupl_v2`'s existing ramp (each outer iteration `k` grows; each
iteration samples `num_pols_sampled` policy indices from the active range and trains each
for `total_episodes_per_policy` episodes, with PPO learning in batches as the rollout
buffer fills — **not** per episode):

- **`E_0` = uniform random** (fixed; never trained).
- **`X_i` (i ≥ 0)** = trained as a pure best response to `E_i`. In particular `X_0` is a
  trained BR to the uniform `E_0` (only `E_0` is fixed).
- **`E_i` (i ≥ 1)** = trained against a **bandit-controlled mixture over `{X_0 … X_{i-1}}`**.

Per-exploited-policy bandits:

- One bandit `bandit[i]` for each `i = 1 … N-1`, over exactly `i` arms `{0 … i-1}` (the X
  policies with index `< i`). Each bandit's arm set is fixed (determined by `i`), so there
  is no growing-arm problem.
- Bandits **persist across outer iterations** — `E_i` is sampled and trained repeatedly as
  `k` ramps, and its bandit keeps learning across those visits.

Training within an outer iteration (asymmetric):

- **Train E**: sample `num_pols_sampled` indices `i ∈ [1..k]`. For each, for
  `total_episodes_per_policy` episodes:
  1. `j = bandit[i].sample()`
  2. play one episode of `E_i` vs `X_j` via `oracle.sample_episode(...)` (buffers
     transitions for `E_i`'s PPO; returns both players' cumulative rewards)
  3. `bandit[i].update(j, reward = X's payoff)` (= −E's payoff)
  Then flush `E_i`'s PPO buffer via `_force_learn`.
- **Train X**: sample `num_pols_sampled` indices `i ∈ [0..k]`. For each, train `X_i` as a
  pure best response to `E_i` (single fixed opponent, mirroring today's `_train_player`
  but with a pure opponent), then flush.

The bandit maximizes reward = X's payoff, i.e. it seeks the opponent mixture that best
exploits the current `E_i`; `E_i`'s PPO learns to be robust to that mixture, driving
BRV(`E_i`) down over the run.

## Abstract adversarial bandit — new module `bandits.py`

```python
class AdversarialBandit(ABC):
    def __init__(self, num_arms: int, ...): ...
    @abstractmethod
    def sample(self) -> int: ...           # draw an arm from the current mixture
    @abstractmethod
    def update(self, arm: int, reward: float): ...  # partial (bandit) feedback
    def distribution(self) -> np.ndarray: ...       # current mixture, for logging
```

- **`Hedge`** — exponential-weights / Exp3-style: maintains per-arm weights; `sample`
  draws ∝ weights; `update` applies an importance-weighted exponential update (partial
  feedback).
- **`RegretMatching`** — maintains cumulative regrets; `distribution` ∝ positive regrets
  (uniform if none positive). Under bandit feedback it updates regrets from
  importance-weighted reward estimates. (Alternative full-info variant — evaluate all arms
  each step — is noted but not the default.)

Rewards are normalized from the game's `[min_utility, max_utility]` to `[0, 1]` inside the
bandit so step-size assumptions hold across games.

`--bandit` maps `"hedge" -> Hedge`, `"rm" -> RegretMatching`.

## Measurement / logging

Every `expl_check_episode_interval` episodes:

- For each exploited policy `E_i`, `i ∈ [0..k]`, compute **BRV(`E_i`)** exactly on the game
  tree (OpenSpiel best response for the exploiter player against the fixed `E_i`).
- Append a record to `stats.jsonl`: `{episodes, episodes_training, walltime,
  brv_per_policy: [BRV(E_0) … BRV(E_k)]}` (plus the existing episode counters).
- Plot BRV vs. episodes, **one line per policy index `i`**, saved in `experiment_dir`.

No payoff matrix, `_update_payoffs`, `_compute_nash`, or NashConv in this path.

## Known limitation — critic is not opponent-conditioned

The shared `PPOConditionedOnPolicyRepresentationAgent` critic is conditioned on the joint
observation plus the training agent's *own* policy embedding, but **not** on the opponent's
policy embedding. So when `E_i` trains against the bandit mixture of `X_j` opponents, its
value/advantage estimates are averaged over the mixture rather than opponent-specific. This
is a pre-existing property (the symmetric loop also trains against Nash mixtures), left
as-is here; a `TODO(claude)` comment marks the spot in `ppo.py`. Opponent-conditioning the
critic is a possible follow-up, out of scope for this spec.

## Code structure

Branch **inside** `run_neupl_v2` (recommended in brainstorming):

- New params: `run_neupl_v2(..., aspro=False, exploited_player="p1", bandit="hedge")`.
- Shared setup (agents, `oracle`, env, ramp schedule `k`, LR annealing, checkpointing,
  logger, `experiment_dir`) is reused unchanged.
- After setup: `if aspro:` run the APSRO loop (its own closures `_train_exploited`,
  `_train_exploiter`, `_measure_brv`, and the `bandit[i]` dict); `else:` today's symmetric
  loop, untouched.
- `X_0` initialization: in `--aspro`, `X_0` is a trainable PPO policy (BR to `E_0`), not the
  `UniformRandomAgentPolicy` the symmetric path installs at index 0 for the exploiter;
  `E_0` stays `UniformRandomAgentPolicy`.
- `__main__` argparse gains `--aspro`, `--exploited_player {p1,p2}`, `--bandit {hedge,rm}`,
  wired into the `run_neupl_v2(...)` call alongside the existing `--neupl_v2` flag.
- Guard: `aspro` together with `use_randall_loss` or `gt_payoffs` raises `ValueError`.

## Testing

- **Bandit unit tests** (`bandits.py`): against a stationary stochastic reward vector,
  both `Hedge` and `RegretMatching` concentrate on (near) the best arm; `distribution`
  stays a valid simplex; a 1-arm bandit is degenerate-safe.
- **BRV sanity**: BRV of the uniform `E_0` on Kuhn matches the analytic/`compute_nash`-free
  best-response value; BRV is within `[min_utility, max_utility]`.
- **Smoke test**: `run_neupl_v2(game_name="kuhn_poker", aspro=True, T=<small>)` runs a few
  iterations, writes `stats.jsonl` with `brv_per_policy`, checkpoints, and produces the BRV
  plot, for both `exploited_player` values and both bandits.
- **Guard test**: `aspro=True` with `use_randall_loss=True` (or `gt_payoffs=True`) raises.

## Open sub-decisions (defaults chosen; easily flipped)

- RM under bandit feedback uses importance-weighted reward estimates (default) vs. full-info
  all-arm evaluation each step.
- BRV plot = one line per policy index vs. episodes.
- `--aspro` leaves Randall loss / gt-payoffs off (guarded).
