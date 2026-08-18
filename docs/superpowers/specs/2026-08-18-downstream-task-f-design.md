# Downstream Task F: Finding Equilibria by Descent–Ascent in Embedding Space

**Date:** 2026-08-18
**Branch:** `claude/downstream-task-f` (based on `conditional-br-2`)
**Source roadmap:** `roadmap/downstream_task_f.md`

## Motivation

Tasks A–E measure how well policy embeddings *predict* game quantities. Task F asks a
different question: can embeddings be used to *find optimal strategies*? Concretely, if we
learn a value function over embedding pairs, can we recover an approximate Nash equilibrium
by optimizing directly in embedding space, then decode the result back into real policies?

## Summary of the approach

1. **Pretrain** a value function `V(e_p1, e_p2) ≈ E[payoff to player 0]` over embedding pairs.
   This is exactly the Task B `PayoffPredictor`, restricted to differentiable (mlp/linear) models.
2. **Solve** for an approximate equilibrium `(e_p1*, e_p2*)` by two-timescale gradient
   descent–ascent *in embedding space*, using gradients of `V` w.r.t. the embeddings.
3. **Decode** `e_p1*`, `e_p2*` back into real policies via per-source decoder callables.
4. **(Optional) Posttrain** `V` on freshly-computed value targets for the embedding pairs
   visited during descent–ascent, then re-solve.
5. **Evaluate** the recovered policies with **NashConv** in the real game.

Scope decisions (agreed):
- **Game:** `kuhn_poker` first, reusing existing `random` / `psro` / `neupl` sources and encoders.
- **Posttraining:** included as a config toggle.
- **Eval metric:** NashConv (exploitability) via OpenSpiel.
- **Deferred:** PBS-conditioned value function, zero-shot-BR alternative for P2, time-averaging
  variant, and matching-pennies / RPS games.

## Design constraints discovered in the codebase

- The value function must be **differentiable w.r.t. embeddings**, so Task F rejects
  `model_type == "random_forest"` and uses `mlp` or `linear` (the torch `PayoffModel`).
- Decoders do not exist yet; only encoders (policy → embedding) do. Task F adds them.
- Off-manifold embeddings decode poorly, so descent–ascent supports optional bounding.

---

## Component 1: Decoders (embedding → policy)

A **decoder** is a callable `decode(embedding) -> open_spiel Policy`. One is built per player,
per source, in `_run_experiment`, and passed into `run_task_f`. `run_task_f` stays
source-agnostic and never inspects an encoder object.

### 1a. Autoencoder decoders (`weight_autoencoder.py`, `functional_autoencoder.py`)

Add `get_decoder(...)` mirroring the existing `get_encoder(...)`:

```python
def get_decoder(self, game, player_id, template_agent, device="cpu") -> Callable[[np.ndarray | torch.Tensor], Policy]:
    """Return decode(embedding) -> PPOAgentPolicy.
    embedding -> autoencoder.decoder -> actor weight-vector -> params loaded into a
    clone of template_agent -> PPOAgentPolicy(game, agent, player_id, use_observation=False).
    """
```

- Weight AE: `autoencoder.decoder(z)` yields the reconstructed actor weight-vector; load it
  into a fresh clone of `template_agent.actor` via `nn.utils.vector_to_parameters`.
- Functional AE: same shape; reuse the existing `_actor_param_specs` / `_vector_to_param_dict`
  helpers to map the decoded vector into the actor's parameter dict, applied to a clone.
- Each `decode` call must produce an **independent** agent (deep-copy the template) so
  successive decodes do not alias weights.

### 1b. Identity decoder

No autoencoder. The embedding *is* the flattened actor weight-vector, so
`decode(embedding)` loads it straight into a template `PPOAgent` clone → `PPOAgentPolicy`.
This is the exact inverse of `ppo_agent_to_vector` (which vectorizes `actor.parameters()`).

### 1c. NeuPL decoder (`psro.py`)

Add a top-level function:

```python
def neupl_decoder(game, agent, embedding, player_id, use_observation=False) -> PPONeuplAgentPolicy:
    return PPONeuplAgentPolicy(game, agent, player_id,
                               use_observation=use_observation, embedding=embedding)
```

This matches the existing line-~1020 pattern in `make_ppo_policies_from_neupl_agents`. The
NeuPL "conditioned" agent (`PPOConditionedOnPolicyRepresentationAgent`) is already loaded in
`_run_experiment`; the decoder closes over it. NeuPL embeddings live in post-norm space, so
descent–ascent bounding uses the observed pool range for that space.

### 1d. Encoder objects returned from data loading

`get_policies_and_embeddings` / `get_policies_and_embeddings2` in `main.py` are extended to
also return the encoder object where one exists (weight/functional AE), or `None`
(identity/neupl). `_run_experiment` uses the returned encoder to build the decoder callable
via `get_decoder`, or builds the NeuPL/identity decoder directly.

---

## Component 2: Value-function pretraining

Reuse `PayoffPredictor` (Task B) unchanged for pretraining:
- `compute_ground_truth_payoffs()` over the P1×P2 pool (sampled via `get_expected_payoffs`,
  consistent with Task B).
- `train_with_agent_level_split(validation_split)`.
- The trained torch model (`predictor.trainer.model`, a `PayoffModel`) is the differentiable
  `V` used by the solver. Input layout matches Task B: `concat([p1_emb, p2_emb])`.

Task F validates that `config.model_config.model_type in {"mlp", "linear"}`.

---

## Component 3: Descent–ascent solver (`downstream.py`)

New class `EmbeddingEquilibriumSolver`:

```python
class EmbeddingEquilibriumSolver:
    def __init__(self, value_model, embedding_dim_p1, embedding_dim_p2,
                 pool_p1_embeddings, pool_p2_embeddings, config, device="cpu"): ...
    def solve(self, init_p1=None, init_p2=None) -> SolveResult: ...
```

Two-timescale gradient descent–ascent. Player 0 (P1) **maximizes** `V`; player 1 (P2)
**minimizes** `V` (zero-sum convention: `V` predicts P0's payoff).

```
e_p1, e_p2 <- init (default: a random pool embedding for each player)
for outer_step in range(outer_steps):
    # inner loop: P2 minimizes V (fast timescale)
    for _ in range(inner_steps):
        g2 = d V(e_p1, e_p2) / d e_p2
        e_p2 <- clamp( e_p2 - lr_p2 * g2 )       # descent
    # outer step: P1 maximizes V (slow timescale)
    g1 = d V(e_p1, e_p2) / d e_p1
    e_p1 <- clamp( e_p1 + lr_p1 * g1 )           # ascent (opposite sign)
    record (e_p1.detach(), e_p2.detach())
```

- **Two-timescale** realized by `inner_steps > 1` and/or `lr_p2 > lr_p1`.
- **Bounding** (`bound_embeddings=True` by default): `clamp` projects each coordinate into the
  per-dimension `[min, max]` range of the corresponding real-embedding pool, keeping iterates
  near the manifold the decoder was trained on. Configurable off.
- **Restarts** (`num_restarts`): run from several initializations; keep the run whose final
  point minimizes an exploitability proxy `max_e2 V - min_e1 V` estimated locally (or simply
  keep all and evaluate NashConv per restart, choosing the best).
- `value_model` is frozen (`requires_grad_(False)` on its params); only the embedding leaf
  tensors carry gradients.
- Returns `SolveResult(e_p1_star, e_p2_star, value_at_star, visited_pairs, trace)`.

---

## Component 4: Posttraining (config toggle)

When `config.posttrain=True`, after each solve:
1. Subsample up to `posttrain_budget` of the visited `(e_p1, e_p2)` pairs.
2. For each, decode both embeddings to policies and compute a value target via
   `get_expected_payoffs(game, p1_policy, p2_policy)` (sampled).
3. Append `(concat_embedding, target)` rows to the value-function training set and
   continue-train `V` for `posttrain_epochs`.
4. Re-run `solve`.

Repeat for `posttrain_rounds`. Terminology follows the roadmap: initial fit = "pretraining",
this loop = "posttraining". Note this requires decoding *intermediate* embeddings, so decode
quality on near-manifold points matters (hence bounding).

---

## Component 5: Evaluation — NashConv

After the final solve:
1. Decode `e_p1*` → `p1_policy` (player 0), `e_p2*` → `p2_policy` (player 1).
2. Compute **NashConv** of the recovered joint policy in the real game using OpenSpiel
   `best_response` / `TabularBestResponse` (the same machinery `ExploitabilityPredictor`
   uses in `downstream.py`): the sum over players of each player's gain from best-responding
   to the other's recovered policy. Lower = closer to equilibrium.
3. Report alongside:
   - `V(e_p1*, e_p2*)` (model's own value at the fixed point),
   - the sampled true payoff of the decoded policy pair (sanity vs the model),
   - a **random-embedding-pair** NashConv baseline (decode two random pool embeddings).

Primary metric: **NashConv of the recovered pair**. Secondary: value-consistency gap and the
baseline delta.

---

## Component 6: Wiring & configuration

### `config.py` — `TaskFConfig`

```python
@dataclass
class TaskFConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)   # mlp or linear only
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
    # eval
    nashconv_baseline_samples: int = 8

    def __post_init__(self):
        if self.model_config.model_type not in ("mlp", "linear"):
            raise ValueError("Task F requires a differentiable value function (mlp or linear).")
        # + range checks on the numeric fields
```

### `tasks.py` — `run_task_f`

```python
def run_task_f(game, p1_policies, p1_embeddings, p2_policies, p2_embeddings,
               p1_decoder, p2_decoder, config, experiment_info, device="cpu") -> dict:
    # 1. pretrain V via PayoffPredictor (Task B)
    # 2. build EmbeddingEquilibriumSolver over V + pool embeddings
    # 3. solve (+ optional posttraining loop)
    # 4. evaluate NashConv, assemble metrics dict, register_result
```

Mirrors `run_task_b`'s signature and result-registration, plus the two decoder callables.

### `main.py`

- `get_policies_and_embeddings` / `get_policies_and_embeddings2`: also return the encoder
  object (or `None`).
- `_run_experiment`: for `task == 'f'`, load both players' policies/embeddings (as `task == 'b'`
  already does), build `p1_decoder`/`p2_decoder` from the returned encoders (autoencoder →
  `get_decoder`, neupl → `neupl_decoder` bound to the loaded agent, identity → direct
  reconstruct), build `TaskFConfig`, and call `run_task_f`.
- `__main__`: add `RUN_TASK_F` flag; build specs for `neupl` / `psro` / `random` sources
  (both players, like Task B). PSRO/random use `mlp`; neupl uses `mlp` (never random_forest).

---

## Testing (TDD)

New `test_task_f.py` (plus small unit tests near the encoders):
1. **Decoder round-trip:** for a real PPO agent, `decode(encode(agent))` reconstructs a policy
   whose action distributions are close to the original (small reconstruction error for AE;
   near-exact for identity).
2. **`neupl_decoder`:** returns a `PPONeuplAgentPolicy` that yields valid action-probability
   dicts summing to 1 over legal actions.
3. **Solver:** on a controlled value function with a known saddle point (e.g. a bilinear
   `V(e1,e2)=e1^T A e2` toy), descent–ascent converges toward the saddle; and on kuhn,
   NashConv after solve is no worse than the random-embedding baseline.
4. **Config validation:** `TaskFConfig` rejects `random_forest` and out-of-range hyperparams.
5. **Smoke test:** `run_task_f` end-to-end on kuhn + `random` source with small `N`, small
   step counts, returns a metrics dict containing `nashconv`.

---

## Files touched

- `weight_autoencoder.py` — add `get_decoder`.
- `functional_autoencoder.py` — add `get_decoder`.
- `psro.py` — add `neupl_decoder`.
- `downstream.py` — add `EmbeddingEquilibriumSolver` (+ NashConv helper if not reusable).
- `config.py` — add `TaskFConfig`.
- `tasks.py` — add `run_task_f`.
- `main.py` — return encoders from loaders; handle `task == 'f'`; `RUN_TASK_F` wiring.
- `test_task_f.py` — new tests.

## Open questions / risks

- **Decode fidelity for autoencoders.** If reconstruction error is high, recovered policies
  will be poor regardless of the solver. The round-trip test surfaces this early; bounding
  mitigates off-manifold decode.
- **NashConv for neural policies.** `PPOAgentPolicy` / `PPONeuplAgentPolicy` compute action
  probabilities on the fly; confirm the OpenSpiel best-response path accepts them directly (it
  does for `ExploitabilityPredictor`) or wrap into a tabular policy if needed.
- **Two-timescale stability.** Descent–ascent can cycle; restarts + bounding + reporting the
  best-of-restarts NashConv is the mitigation for v1 (no convergence guarantee claimed).
