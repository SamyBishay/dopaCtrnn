---
title: What To Do Now
updated: 2026-06-28
---

# What To Do Now

Concrete implementation plan derived from post-scrutiny fixes. Cross-reference `fixes.md` for full problem statements and tradeoffs. Things already implemented are noted and skipped.

---

## 1. Code changes

### Evaluation

- **Stochastic action selection.** In `evaluate_vec` (`analysis.py`), replace `argmax` on action logits with `torch.multinomial(torch.softmax(pi, -1), 1)`. Note: the supervisor also uses sampled actions (not greedy).
- **Eval seed** — already done. `train.py:358` passes `ev_seed = total_episodes`; `analysis.py:87` seeds from it.

### Training dynamics

- **Noise injection.** Add `self.noise_std: float = 0.05` (tunable) to `config.py`. In `model.py`, after each `h = h + (dt/tau) * dh` (GDNet lines 117–119, HabNet line 198), before the `lesion` check:
  ```python
  if self.training:
      h = h + torch.randn_like(h) * self.cfg.noise_std
  ```

- **DA-modulated excitability.** In `GDNet.step()`, inject a DA-dependent additive bias before the nonlinearity:
  ```python
  exc_bias = cfg.da_exc_base + cfg.da_exc_gain * da_tonic
  dh = -h + rec_gain * (r @ self.W.T) + x @ self.W_in.T + self.b + exc_bias
  ```
  High DA → uniform push toward tanh saturation → attractor basins widen globally. Distinct from recurrent gain (which scales connections proportionally) and tau modulation (which changes integration speed). Two config params: `da_exc_base`, `da_exc_gain`.

- **APE teaching signal.** Currently hab imitates the *combined* policy's argmax. Change to use **GD's own output** as teacher. Rule per timestep in `train.py` (find `ape_loss`):
  - If `argmax(pi_gd) == WAIT`: one-hot WAIT target, cross-entropy.
  - Else: `softmax(pi_gd[N,S,E,W])` — soft target over the 4 movement directions, KL divergence or cross-entropy with soft targets.

- **DA gate: replace penalty ramp with RPE-based signal.** Remove `da_pen` from the loss entirely. Replace with:

  Each training iteration, compute `e = mean(|adv|)` for the current batch. Maintain:
  ```python
  mu  = 0.99 * mu  + 0.01 * e
  var = 0.99 * var + 0.01 * (e - mu)**2
  z   = (e - mu) / (var**0.5 + 1e-6)

  if z > 0:
      da_signal = sigmoid(z)           # instant rise on surprise
  else:
      da_signal = 0.999 * da_signal    # slow DAT-mediated decay (~1000 iters to zero)

  w_gd = sigmoid(alpha * da_signal + bias)
  ```
  Add a 100-iteration warmup with `w_gd` pinned to 1.0 while mu/var accumulate. Add a floor condition: `w_gd` cannot drop until `hab_solo_acc > criterion`. Parameters (`alpha`, `bias`, warmup length) go in `config.py`.

  This makes DA track genuine learning dynamics (RPE shrinks as task is mastered → gate closes naturally) rather than a fixed schedule. Timing is emergent, not imposed. Biological grounding: Schultz et al. (1997) RPE; Pearce & Hall (1980) surprise-gated associability; Lloyd & Dayan (2015) fast/slow DA channels. See `fixes.md` F5 for full tradeoff analysis.

### Environment

- **Delay phase redesign** (`environment.py`). Three sub-changes:

  **a. PUSHBACK phase** (replaces teleport at `environment.py:267`): on reaching the sample arm end, override agent actions and walk the mouse back step-by-step (arm → junction → stem → START). `delay_idx` starts from the first PUSHBACK step. No step cost during PUSHBACK (extend the `~in_delay` mask).

  **b. CONFINED sub-state** (mouse arrives at START during PUSHBACK): set `obs[2] = 1.0`. Block any action that would move the mouse off START. `delay_idx` continues counting.

  **c. CHOICE transition** (when `delay_idx >= current_delay`): clear `obs[2] = 0.0`, set `sig_choice` as now. Assert at env init: `cfg.delay_start >= pushback_path_length`.

  `obs[2]` is currently always 0 and never written; repurposing it as the confinement bit is safe, no OBS_DIM change.

### Analysis

- **hab_onset criterion**: `analysis.py:152` — change `hab >= 0.8` to `hab >= 0.99`. Consistent with the 99% early-stopping threshold. The H2 combined-accuracy pass threshold (`comb[-1] >= 0.8`, line 162) stays at 80% — different question.
- **Rolling accuracy** — already done. `rolling_buf`, `rolling_window = 20_000` in `config.py`, applied in `train.py:339`.

---

## 2. Experiment protocol

Rerun everything with **≥5 seeds** under the corrected codebase. Single-seed results are uninterpretable given the 37% CV in `hab_onset` from E0's 5-seed spread.

### Experiment DAG

```
E0 → E1 → E2
           ├─→ E3   DA output gain variants (gain_only / weights_only / combined)
           ├─→ E4   value-free vs value-coupled APE
           ├─→ E7   DA tau widen/deepen ──────────────────────────────────────┐
           ├─→ E10  DA recurrent gain (NEW)                                   │
           │        ├─→ E11  recurrent gain + DA plasticity ──────────────────┤
           │        └─→ E9 ◄─────────────────────────────────────────────────-┤
           │                 recurrent gain + dual tau                         │
           └─→ E12  DA excitability (additive bias, NEW) ──────────────────────┤
                    └─→ E13  excitability + plasticity + dual tau (NEW) ◄──────┘
                             (parents: E12, E11, E7)
```

**E2 arms:** ego/allo split (main) · freeze-hab ablation (skip `opt_hab.step()` — tests that GD reaches criterion without hab gradients). Drop the expression/scheduled arm — not load-bearing for any core claim.

**E4:** reframe in the write-up as the reward-insensitivity falsification (value-coupled fails where value-free succeeds), not a sensitivity analysis.

**E9** (renamed from "full Naudé"): two parents — adds recurrent gain (E10) to the widen/deepen tau (E7). Tests whether combining both DA mechanisms improves on either alone.

**E10** (NEW): recurrent gain only (`da_gain_mode="recurrent"`, no tau modulation). Needed to isolate the recurrent gain effect before combining or adding plasticity.

**E11** (NEW): recurrent gain + DA-modulated plasticity. In `train.py`, before `opt_gd.step()`, scale the GD learning rate by DA level:
```python
opt_gd.param_groups[0]['lr'] = cfg.lr_gd * (da_plasticity_base + da_plasticity_gain * da_tonic)
```
High DA early → higher effective LR → faster GD consolidation; low DA post-handoff → near-zero LR → frozen policy. Two new config params: `da_plasticity_base`, `da_plasticity_gain`. Compare `da_plasticity=True` vs `False` arms. Note: gradient scaling would be cancelled by Adam's normalisation — LR scaling is the correct approach.

**E12** (NEW): DA excitability alone (additive bias in GDNet.step(), no recurrent gain, no tau). Tests whether the basin-widening mechanism works independently. Expected signature: PR decreases during high-DA phases, delay-period decoder holds longer, fixed-point stability radius increases — all computable from existing `analysis.py`.

**E13** (NEW): excitability + plasticity + dual tau. Three parents: E12, E11, E7. Most faithful implementation of Naudé's complete mechanism. Tests synergy: GD consolidates faster (plasticity) AND maintains deeper WM attractors (excitability + deepen tau) AND hands off more cleanly than any single mechanism.

---

## 3. Cluster setup

- **Check Grid5000 availability** before launching: `ssh <login>@access.grid5000.fr "oarstat -s"` or check the Grid5000 status page.
- **Adapt the experiment launcher** (`submit_oar.sh`, `batch_runner.py`) to Grid5000: one OAR job per seed, 1 core + ~4 GB RAM per job, no GPU needed (model is small).
- **Target batch_size=8** — gives meaningful per-trial dynamics (16× more online than current 128), 3 seeds fit in 3 hours on Grid5000 CPU nodes running in parallel. Do a timing benchmark on one 200k run at batch_size=8 before committing to the full sweep.
- **Fixed delay** (confirmed from Villet): Villet uses a fixed 90 s delay throughout. Drop the delay curriculum entirely — set one fixed `delay` value in config, remove `delay_start`/`delay_end`/`delay_step`.

---

## 4. Environment signal magnitudes (F14)

In `environment.py`:

- **Choice-onset signal**: replace the full reset (`_sig[2] = cfg.sig_val`) with a value consistent with one additional decay step:
  ```python
  choice_sig = (1.0 / (self.w - 1)) * 0.1
  self._sig[to_choice, 2] = choice_sig
  ```
- **Sample signal**: replace `cfg.sig_val` with `1.0 / (self.w - 1)` — normalised to maze width so the signal scales with geometry. For `len_edge=7`, `w=9`, this gives `0.125`.
- Remove `sig_val` from `config.py` or repurpose it as a multiplier after normalisation.

---

## 5. Writing notes

- **H5 / Discussion**: note that greedy eval made H5 a mathematical identity — silencing the habitual network collapsed to `argmax(w_gd * pi_gd)`, making accuracy recovery a consequence of the decoding rule, not the mechanism. With stochastic eval the test is genuine. State this plainly; do not frame the old result as a partial confirmation.
