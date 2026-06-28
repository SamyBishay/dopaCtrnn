---
title: What To Do Now
updated: 2026-06-28
---

# What To Do Now

Concrete implementation plan derived from post-scrutiny fixes. Cross-reference `fixes.md` for full problem statements and tradeoffs. Things already implemented are noted and skipped.

---

## 1. Code changes

### Evaluation

- **Stochastic action selection.** In `evaluate_vec` (`analysis.py:109`), replace:
  ```python
  acts = out["combined"].argmax(-1).cpu().numpy()   # greedy [B]
  ```
  with:
  ```python
  acts = Categorical(logits=out["combined"]).sample().cpu().numpy()   # stochastic [B]
  ```
  Use `Categorical` (already imported and used in `rollout()`), not `torch.multinomial+softmax`.
  **Scope:** only `evaluate_vec`. Leave `_record_episode` in `train.py` greedy (trajectory recording).

- **Eval seed** — already done. `train.py:358` passes `ev_seed = total_episodes`; `analysis.py:87` seeds from it.

### Initialisation

- **Match supervisor's W_in scale (F13).** In `model.py:50`, change `GDNet`'s input weight initialisation from `std=0.1` to `std=1.0`:
  ```python
  self.W_in = nn.Parameter(torch.randn(n, cfg.obs_dim) * 1.0)
  ```
  **Scope:** only `GDNet` line 50. Do NOT change `HabNet.W_in` (line 169), which also uses `0.1` but should stay unchanged. This flips the dominance at init from recurrent-dominated (~13× recurrent over input) to input-dominated (~6× input over recurrent), consistent with the supervisor's PFC design. The recurrent init (`std≈0.9/√n`) does not need changing. **Implement together with the APE teaching signal change** — both touch training dynamics and should land as one baseline reset before any ladder experiments are rerun.

### Training dynamics

- **Noise injection.** Add `noise_std: float = 0.05` (tunable) to `config.py`. In `model.py`, insert noise **after the complete `if da_tau_on / else` block and before `if lesion:`** in both networks:

  GDNet (insert between lines 119 and 121):
  ```python
  # after: h = h + (self.cfg.dt / _tau(self.tau_p)) * dh  [else branch, line 119]
  if self.training:
      h = h + torch.randn_like(h) * self.cfg.noise_std
  # before: if lesion: [line 121]
  ```
  HabNet (insert between lines 198 and 199):
  ```python
  # after: h = h + (self.cfg.dt / _tau(self.tau_p)) * dh  [line 198]
  if self.training:
      h = h + torch.randn_like(h) * self.cfg.noise_std
  # before: if lesion: [line 199]
  ```

- **DA-modulated excitability.** Add `da_exc_base: float = 0.0` and `da_exc_gain: float = 0.0` to `config.py` (off by default; E12 sets them non-zero). In `GDNet.step()`, compute `exc_bias` from the **input `da_tonic` argument** (pre-update; do NOT use `da_ton_pre` or the `da_tonic` reassigned at line 125), then add it to `dh` **after the `if da_gain_mode == "recurrent" / else` block** (both branches must see the bias):
  ```python
  # after the if/else dh block (covers both recurrent and output gain modes):
  exc_bias = self.cfg.da_exc_base + self.cfg.da_exc_gain * da_tonic  # da_tonic = input arg
  dh = dh + exc_bias
  ```
  High DA → uniform push toward tanh saturation → attractor basins widen globally. Distinct from recurrent gain (scales connections) and tau modulation (changes integration speed).

- **APE teaching signal.** Keep teacher as the **combined policy** (do not change to GD-only). Replace argmax hard targets with soft KL targets over all 5 actions:
  ```python
  # per step, before opt_hab.step():
  kl = F.kl_div(
      F.log_softmax(pi_h, dim=-1),
      F.softmax(pi_combined.detach(), dim=-1),
      reduction='sum'
  )
  kl.backward()
  opt_hab.step(); opt_hab.zero_grad()
  ```
  APE loss is computed and backpropagated at **every single step** — many updates per episode. A2C loss (GAE) is computed at **episode end** only. A batch = one episode for the habitual update.

- **Habitual update frequency (Villet faithfulness).** Currently both GD and hab are updated from the same B=128 episode batch — hab's gradient is a 128-episode average, not an online signal. Fix: restructure `_train_batch` so the per-step inner loop calls `opt_hab.step()` after each step:
  - Inside the `for _ in range(cfg.max_episode_steps):` loop, after each `venv.step()`, compute the KL APE loss for active lanes (mask by `active`) and call `opt_hab.zero_grad(); kl.backward(); opt_hab.step()`.
  - GD keeps its batch accumulation (stacks [T,B] tensors, one backward at loop end).
  - Remove the batched `total_hab` backward and `opt_hab.step()` at `train.py:226-227`.

- **DA gate: replace penalty ramp with RPE-based signal.** Remove `da_pen` from the GD loss (`da_pen` is a local variable in `_train_batch`, not a config param). Remove config params `da_cost_lambda`, `da_warmup`, `da_ramp` from `config.py`. Add new config params: `rpe_alpha: float = 6.0`, `rpe_bias: float = -2.0`, `rpe_warmup_iters: int = 100`.

  Modify `_train_batch` to return `mean(|adv|)` as a fourth value:
  ```python
  e_rpe = float(adv[valid_t].abs().mean())   # add before return
  return float(total_gd.detach()), float(total_hab.detach()), n_correct, e_rpe
  ```

  In `train()`, maintain running stats and compute the gate externally, then pass it as `force_w` to `model.step()` (which already respects `force_w` at `model.py:269`):
  ```python
  # outer train() loop, after _train_batch returns e_rpe:
  mu  = 0.99 * mu  + 0.01 * e_rpe
  var = 0.99 * var + 0.01 * (e_rpe - mu)**2
  z   = (e_rpe - mu) / (var**0.5 + 1e-6)
  if z > 0:
      da_signal = float(torch.sigmoid(torch.tensor(z)))
  else:
      da_signal = 0.999 * da_signal
  # warmup + floor: gate stays open until hab is competent
  if it < cfg.rpe_warmup_iters or hab_solo_acc < cfg.early_stop_acc:
      current_w_gd = 1.0
  else:
      current_w_gd = float(torch.sigmoid(torch.tensor(cfg.rpe_alpha * da_signal + cfg.rpe_bias)))
  ```
  Pass `current_w_gd` as `force_w` into `_train_batch` so training uses the externally-computed gate. `da_request` inside `GDNet` is still learned and still drives expression gain (`W_eff`); only the arbitration gate (`w_gd`) switches to RPE. `wgd_alpha` and `wgd_bias` remain in config for `gate_mode="expression"` experiments.

  Biological grounding: Schultz et al. (1997) RPE; Pearce & Hall (1980) surprise-gated associability; Lloyd & Dayan (2015) fast/slow DA channels. See `fixes.md` F5 for full tradeoff analysis.

### Environment

- **Delay phase redesign** (`environment.py`). Three sub-changes:

  **a. PUSHBACK sub-state** (replaces teleport at `environment.py:267`): PUSHBACK is **not a new phase constant** — it is the early part of the DELAY phase, tracked via `delay_idx`. On reaching the sample arm end, instead of teleporting, programmatically walk the agent back step-by-step (arm → junction → stem → START) by overriding the agent's actions. `delay_idx` starts counting from the first PUSHBACK step. No step cost during PUSHBACK (already covered by the `in_delay` mask).

  **b. CONFINED sub-state** (mouse arrives at START during PUSHBACK): set `obs[2] = 1.0`. This slot (`obs[2]`) is always 0 and never written — safe to repurpose, no OBS_DIM change. (`_sig[2]` → `obs[5]` is the choice signal, a separate slot — no conflict.) Block any action that would move the mouse off START. `delay_idx` continues counting.

  **c. CHOICE transition** (when `delay_idx >= current_delay`): clear `obs[2] = 0.0`, set `sig_choice` as now. For the default config (len_edge=7, difficulty=2), the pushback path is **6 steps** (arm end → junction → 3 stem rows → START). Assert at env init: `cfg.delay >= 6`. Net hold at START = `40 − 6 = 34` steps.

  **Fixed delay**: add `delay: int = 40` to `config.py` (supervisor's hard `MAX_DELAY`). Remove `delay_start`, `delay_max`, `delay_advance_acc`, `delay_advance_evals` from `config.py`. Remove the delay curriculum block in `train()` (`train.py:416-426`). `current_delay` becomes fixed at `cfg.delay`.

### Analysis

- **hab_onset criterion**: in `h2_handoff` (`analysis.py`), find `cross = np.where(hab >= 0.8)[0]` and change `0.8` to `0.99`. Consistent with the 99% early-stopping threshold. The H2 combined-accuracy pass threshold (`comb[-1] >= 0.8`) stays at 80% — different question.
- **Rolling accuracy / early stopping** — already done. `rolling_buf`, `rolling_window = 20_000` in `config.py`, early stop at `early_stop_acc = 0.99` in `train.py:339-344`. The training loop already stops once the rolling accuracy threshold is met — no change needed.
- **Steps to goal logging.** Add `steps_to_goal` to the `logs` dict in `train()`. In `_train_batch`, track mean steps per completed episode in the batch and return it as a fifth value:
  ```python
  # before return in _train_batch: collect step counts for done episodes
  # (step_count is already tracked in venv.step_count; read it when done[b] fires)
  mean_steps = float(np.mean([venv.step_count[b] for b in range(B) if newly_done[b]])) if newly_done.any() else float('nan')
  ```
  Log it at each eval window alongside `combined_acc`.

---

## 2. Experiment protocol

Rerun everything with **≥5 seeds** under the corrected codebase. Single-seed results are uninterpretable given the 37% CV in `hab_onset` from E0's 5-seed spread.

### Experiment DAG

```
E0 → E1 → E2
           └─→ E6  DA uniform tau shortening
                   ├─→ E3   DA output gain (gain_only / weights_only / combined ["both" in code])
                   ├─→ E4   value-free vs value-coupled APE
                   ├─→ E7   DA tau widen/deepen
                   ├─→ E10  DA recurrent gain (NEW)
                   │        ├─→ E11  recurrent gain + DA plasticity
                   │        └─→ E9   recurrent gain + dual tau
                   │                 (parents: E10, E7)
                   └─→ E12  DA excitability (additive bias, NEW)
                            └─→ E13  excitability + plasticity + dual tau
                                     (parents: E12, E11, E7)
```

Note: E6→E3 and E6→E4 are **experimental ordering dependencies only** (run E6 as timing baseline before adding mechanisms) — not code dependencies.

### Priority table under a 1-hour grid5000 budget

All six must-survive hypotheses (H1–H3, H5–H7) are measured on the **E2 model itself**, not on the ladder. The entire H_tau ladder (E3–E13) should be **deferred to Future Work** and named as such in the mémoire. Concentrate all statistical power on E2.

| Priority | Run | Seeds | Yields | Notes |
|---|---|---|---|---|
| **Must** | E2 main (ego/allo + all §1 fixes) | 5–8 | H1, H2, H3, H5, H6 | entire mémoire rests here |
| **Free** | H7 untrained control | — | H7 | eval-only at init; fold into E2 analysis, zero cost |
| If time | E2 freeze-hab ablation | 3 | strengthens H2/H5 | same code, one extra arm |
| If time | E0/E1 baselines | 1–2 each | Discussion contrast | not hypotheses |
| **Cut → Future Work** | E3, E4, E6, E7, E9, E10, E11, E12, E13 | 0 | H_tau ladder | name in Future Research, do not run |

A clean, well-seeded E2 reporting an honest partial H5 is more credible than a sprawling tree of n=1 mechanism runs that are "uninterpretable" by the standards already set in CLAUDE.md §5.

**E2 arms:** ego/allo split (main) · freeze-hab ablation. Drop the expression/scheduled arm.
- Freeze-hab ablation: add `freeze_hab: bool = False` to `config.py`. When `True`, skip the per-step `opt_hab` update entirely in `_train_batch`. Tests that GD reaches criterion without hab gradients.

**E3**: compare `gain_only`, `weights_only`, `"both"` (= "combined" in prose, **"both" in config and model code**). Do not use the string `"combined"` in any config or model check.

**E4:** value-coupled fails where value-free succeeds — reward-insensitivity falsification, not sensitivity analysis.

**E6**: uniform tau shortening. Prerequisite ordering before E7/E9.

**E7**: asymmetric tau — fast tau for widen (decision), slow tau for deepen (maintenance).

**E9**: recurrent gain (E10) + widen/deepen tau (E7). Tests synergy.

**E10** (NEW): recurrent gain only (`da_gain_mode="recurrent"`, no tau).

**E11** (NEW): recurrent gain + DA-modulated plasticity. New config params: `da_plasticity: bool = False`, `da_plasticity_base: float = 0.1`, `da_plasticity_gain: float = 0.9`.
```python
opt_gd.param_groups[0]['lr'] = cfg.lr_gd * (cfg.da_plasticity_base + cfg.da_plasticity_gain * da_tonic)
```

**E12** (NEW): DA excitability alone. New config params: `da_exc_base: float = 0.0`, `da_exc_gain: float = 0.1` (non-zero for E12 arm). No recurrent gain, no tau.

**E13** (NEW): E12 + E11 + E7 combined. No new config params.

---

## 3. Results streaming and monitoring

Results must be accessible as experiments complete, without waiting for all seeds:

- **Incremental batch viz**: `build_batch_viz.py` must be re-runnable at any point during a sweep and produce a valid `batch_visualiser.html` from however many seed directories are present. It should not error on missing seeds — just skip and note the count.
- **Live watch script**: add `watch_results.py` to `Code/main/`. Polls `results/<exp>/` every 30 s and prints a plain-text table: experiment / arm / seeds-complete / mean-acc ± std / hab_onset mean / H1–H6 pass-fail. Works over SSH, no GUI required.
- **Drag-and-drop**: already works in `visualiser.html` — no change needed.

---

## 4. Per-experiment graphs

Each visualiser must show:

1. **Smoothed accuracy** (combined, hab-solo, GD-solo) — already in `visualiser.html`. Add a smoothing toggle (exponential α=0.05) so raw and smoothed curves are both available.
2. **Steps to goal** over training — requires `steps_to_goal` in logs (§1). Add as a secondary panel in the handoff chart.
3. **DA recruitment** (`da_recruit`) — already logged and plotted.
4. **Hypothesis pass/fail pills** (H1–H6) — already partially present. Ensure all hypotheses are visible without scrolling and are derived from `results.json`.

**Visualiser feature status:**
- **Agent on maze map**: present in `visualiser.html` (SVG `layer-agent`, lines 573-577). **Not present in `batch_visualiser.html`** — add maze+agent replay panel to batch viz.
- **Legend hover → highlight series**: present in `visualiser.html` (lines 694-702). **Not present in `batch_visualiser.html`** — add same `mouseenter`/`mouseleave` logic to batch viz.

---

## 5. Environment signal magnitudes

In `environment.py`:

- **Choice-onset signal** (`environment.py:280`): replace `self._sig[to_choice, 2] = self.cfg.sig_val` with:
  ```python
  choice_sig = (1.0 / (self.w - 1)) * 0.1
  self._sig[to_choice, 2] = choice_sig
  ```
- **Sample signal** (`environment.py:252-253`): replace `self.cfg.sig_val` with `1.0 / (self.w - 1)`. For `len_edge=7`, `w=9`, this gives `0.125`.
- Remove `sig_val` from `config.py` or repurpose as a multiplier (value 1.0) to avoid breaking any remaining read sites until explicitly cleaned up.

---

## 6. Runtime optimisations (grid5000, 1-hour budget)

All changes in this section target wall-clock time. Apply them for any timed run; they do not affect correctness of the hypotheses.

### Network sizes

- **Set `n_hab = 128`** (down from 512). The habitual net is a 4-dim phase-signal Go/NoGo imitation policy — 512 units is gross overkill. This shrinks the recurrent matmul from `[128×512]@[512×512]` to `[128×128]@[128×128]` (~16× fewer FLOPs) and matters most now that hab gets per-step backward passes. Keep `n_gd = 512` for the headline runs (H6 WM-attractor capacity needs it); use `n_gd = 256` for smoke runs only.
- **Alternative to shrinking n_hab**: set `hab_rank = 4` in `config.py` (low-rank path already in `model.py:170`). Keeps n_hab=512 but makes "habit is low-dimensional" structurally true rather than something recovered post-hoc. Pick one approach; don't do both.

### Training loop

- **`max_episode_steps`: 200 → 90.** With fixed `delay=40`, pushback=6 steps, and ~15 navigation steps, ~60 steps suffices for most episodes. 90 gives headroom. The delay phase (held at START) is what makes the current cap wasteful.
- **`rolling_window`: 20_000 → 5_000.** The current value means early stopping cannot fire until `len(rolling_buf) >= 20_000` — a hard floor of ~157 iterations (~20k episodes) regardless of accuracy. At 5_000, a fast learner stops much earlier. 99% accuracy over 5k trials is still a strong criterion.
- **`eval_every`: 100 → 500.** Each eval window runs 3 passes (combined/hab/gd) + fixed-delay; at every-100-episodes this is a significant fraction of wall time.
- **`eval_trials`: 200 → 100.** Halves eval cost; still sufficient for pass/fail detection during training.
- **`traj_log_every = 0`** for timed runs. The `_record_episode` call inside the training loop (every 50 episodes by default) adds overhead and produces output not needed for hypothesis testing.
- **`ckpt_every = 0`** for timed runs unless resume is needed.

### Hab per-step update — optimizer call batching

The per-step `opt_hab.step()` introduces ~50–100 optimizer calls per training iteration. If this dominates, accumulate the KL loss over **K=5 steps** before stepping:
```python
if step_count % 5 == 0 or not active.any():
    opt_hab.zero_grad(); kl_accum.backward(); opt_hab.step(); kl_accum = None
else:
    kl_accum = kl_accum + kl if kl_accum is not None else kl
```
This reduces optimizer overhead 5× while remaining effectively online. K is a config knob: `hab_update_freq: int = 1` (1 = every step, original intent; 5 = batched).

### Seed parallelism on grid5000

BLAS scales poorly past ~4 threads on these workloads; more processes beat more threads per process. Recommended launch:

```bash
# 40-core node, DOPA_NUM_THREADS=4 → 9 concurrent seeds
for s in $(seq 0 7); do
    DOPA_NUM_THREADS=4 python run_experiment.py --exp e2 --seed $s &
done
wait
```

Or use `DOPA_NUM_THREADS=2` for ~18 concurrent seeds (likely higher total throughput — confirm with the smoke run). Leave 2–4 cores headroom for the OS.

**No mixed precision.** float16 has no BLAS acceleration on CPU and is often slower; stay float32.

### Hab/GD parallelism — correct architecture

**Keep single-process, sequential, two optimisers — exactly as `train.py` is now.** Do not split GD and hab into separate processes or threads.

Reasoning:
- **Hard step-level data dependency.** `pi_combined = w·pi_gd + (1−w)·pi_h`, and hab's per-step KL target is `softmax(pi_combined.detach())` — which requires `pi_gd` first. GD must run before hab within each step; no genuine concurrency is available.
- **Autograd graphs are already cleanly separable.** GD uses `pi_h.detach()`, hab uses `pi_combined.detach()` — they don't fight over gradients. Two optimisers in one process is correct and sufficient.
- **Process separation would be slower.** Serialising `pi_gd`/`pi_combined` (128×5 tensors) across a process boundary every step dwarfs the tens-of-µs matmuls being "parallelised".
- **The parallelism that pays is across seeds** — embarrassingly parallel, zero inter-process communication, perfectly matched to a 40-core node (see seed parallelism above).

The levers that actually reduce wall-clock time are algorithmic: shrink n_hab (16× fewer FLOPs), cap max_episode_steps, reduce rolling_window. Fix those; don't re-architect.

### Smoke run before the parallel wave

Before launching the full seed wave, run a single cheap smoke run to measure ep/s and confirm the corrected codebase still learns:
- `n_gd=256, n_hab=128, delay=20, episodes=30_000, rolling_window=2_000`
- If it doesn't reach 80% combined accuracy by ~20k episodes, the code changes broke something — do not launch the full wave blind.
