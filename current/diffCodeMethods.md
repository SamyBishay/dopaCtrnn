---
title: Code vs Methods — Discrepancies
created: 2026-06-20
type: working-note
---

# Code vs Methods — Discrepancies

This file catalogs every place where `Code/main/` diverges from what `Mémoire/memoire_method_section_draft.md` (and the introduction) claims. Items are ordered roughly by severity. Fix each before submitting the mémoire.

---

## D1 — Observation dimensions and allocentric/egocentric split (CRITICAL)

**Methods §2.3.1 (GD):** "22-dimensional allocentric observation (an allocentric position map, a blocked-arm indicator, and a phase indicator)"

**Methods §2.3.2 (Hab):** "14-dimensional egocentric observation (wall/proximity sensors, local landmarks, and the previous action)"

**Methods §2.3.2:** explicitly calls this "a deliberate modelling assumption" and the reason the systems are "functionally distinct."

**Code (config.py):** `obs_dim = 6` for both systems.
**Code (environment.py):** single 6D observation: `[col/(COLS−1), row/(ROWS−1), 0, sig_L, sig_R, sig_choice]`
**Code (model.py DualSystemModel.step):** `model.step(obs_t, obs_t)` — both GDNet and HabNet receive **the same 6D allocentric observation**.

**Consequence:** The entire allocentric/egocentric asymmetry described as the reason the habitual system "cannot plan the non-match rule" does not exist in the implementation. Both systems see identical, globally-referenced position coordinates. The 22D position map, wall sensors, and previous-action channel are absent.

**Action required:** Either (a) implement the split as described, or (b) rewrite §2.3 to accurately describe the 6D shared observation and explain why the systems remain functionally distinct despite receiving the same input (the answer is the learning rule and DA modulation, not sensory access).

---

## D2 — Task: three phases described, four phases implemented

**Methods §2.2:** "Each trial comprises three phases: **sample**, **delay**, **test**."

**Code (environment.py):** four phases: `pre_sample → sample → delay → choice`.

The `pre_sample` phase (agent walks from START at (2,2) to JUNCTION at (0,2)) exists in the code and generates its own shaped reward (`junction_bonus = 0.30`), but is invisible in the methods.

**Action required:** Add `pre_sample` to the task description in §2.2. Mention the `junction_bonus` shaping reward.

---

## D3 — Delay phase: agent not held in start box

**Methods §2.2:** "In the delay phase, the agent is held in the start box for a fixed number of timesteps."

**Code (environment.py `step`):**
```python
elif self.phase == "delay":
    self._phase_signal *= 0.1   # exponential decay
    self.delay_idx += 1
    if self.delay_idx >= self.current_delay:
        self.phase = "choice"
```
The delay is simply a step counter (`delay_idx`); the agent continues to move freely anywhere in the maze. There is no confinement to a start box.

**Consequence:** The working-memory demand is real (phase signal decays to ~3×10⁻⁵ by step 5), but the agent navigates freely throughout — a meaningfully different setup from a fixed holding period.

**Action required:** Replace "held in the start box" with an accurate description: "the delay phase is a fixed number of free-navigation timesteps during which the phase signal decays exponentially; the agent can move anywhere but receives no new cue."

---

## D4 — Sample phase: agent is not "forced"; no reward at sample arm

**Methods §2.2:** "one arm is physically blocked and the agent is **forced** into the open arm, **where it receives reward**."

**Code:** The blocked arm cells become impassable, but the agent still makes movement decisions each step under the policy (it can WAIT, oscillate, take sub-optimal paths). "Forced" overstates the constraint. More importantly, **no reward** is given for reaching the sample arm end; the code gives a shaping bonus (`arm_end_bonus = 0.20`) added to `task_r` for dense credit, but this is a training shaping signal, not a reward in the Villet (2025) sense (where actual sucrose reward is delivered).

**Action required:** Clarify (a) that the arm is blocked but the agent navigates freely, and (b) that the arm_end_bonus is a reward-shaping term rather than a biological reward, so the analogy to Villet's sample-phase sucrose reward should be flagged.

---

## D5 — w_GD uses da_request, not "DA_total"

**Methods §2.3.3:**
> `w_GD = σ(α · DA_total + bias)`

**Code (model.py DualSystemModel.step):**
```python
w_gd = torch.sigmoid(self.alpha * da_request + self.bias)
```
The mixing weight uses `da_request` (the phasic, instantaneous signal emitted by GDNet), **not** `da_tonic` (the slow low-pass) and not any "DA_total." The `da_tonic` variable is only used inside `GDNet.step()` for the gain modulation on D2-like units.

```
Signal routing:
  da_request (phasic)  ──► D1-like gain modulation (GDNet)
                       ──► w_GD mixing weight (DualSystemModel)
  da_tonic   (tonic)   ──► D2-like gain modulation (GDNet)
```

**Action required:** Replace "DA_total" with "da_request" (or "phasic DA signal") in the formula and explain that the mixing weight tracks the fast phasic signal, not the tonic baseline.

---

## D6 — DA gain applied to GD system only; methods implies both systems

**Methods §2.3.3:** "Dopamine sets a multiplicative gain on **each system's** effective weights, W_eff = f(DA) · W"

**Code (model.py):**
- `GDNet.step()` applies `gain = gain_base + gain_da * da_comp` to its output.
- `HabNet.step()` has **no** DA gain term.

```
GDNet:   r_out = (gain_base + gain_da · da_comp) · tanh(h)   ← DA-gated
HabNet:  r_out = tanh(h)                                      ← no DA gating
```

**Action required:** Remove "each system's" — dopamine expression-level gating is a property of the goal-directed system only. The introduction and §2.3.3 should be narrowed to "the goal-directed system's expression-level gain."

---

## D7 — DA gain is activation-level, not weight-level (implementation note)

**Methods §2.3.3:** "W_eff = f(DA) · W … scaling how strongly the intact learned weights are expressed"

**Code:**
```python
r_out = gain * torch.tanh(h)    # gain ∈ ℝ^n (per-neuron)
pi    = r_out @ self.W_out.T    # effective output = gain·tanh(h)·W_out
```
The gain multiplies the **activation vector** `tanh(h)`, not the weight matrix W itself. Because `gain` is per-neuron (not a scalar), the effective output transformation is `diag(gain) · tanh(h) · W_out.T`, which is mathematically equivalent to a modified weight `W_eff = diag(gain) · W_out` applied to `tanh(h)` — but the recurrent W is unchanged, and the effect on the hidden dynamics is different from a literal per-weight modulation.

**Action required:** Be precise: "dopamine multiplies each unit's output activation, r_out = gain(DA) · tanh(h), which scales the effective contribution of each unit to the policy." This is cleaner than the W_eff = f(DA)·W shorthand, which implies the weight matrix itself is scaled.

---

## D8 — "mot" (motivation) factor absent from methods formula

**Methods §2.3.3:**
> `combined logits = w_GD · π_GD + (1 − w_GD) · π_H`

**Code (model.py DualSystemModel.step):**
```python
combined = w * mot * pi_gd + (1.0 - w) * pi_h
```
`mot` (default 1.0) is set to 0.0 in devaluation trials (H3, H5). It acts as a multiplicative gate on the goal-directed policy contribution before mixing — the implemented mechanism for devaluation, not mentioned in the mixing formula.

**Diagram of actual devaluation mechanism:**
```
            mot=1.0 (normal)        mot=0.0 (devalued)
            ─────────────────        ─────────────────
combined =  w_GD · π_GD + π_H·(1−w_GD)     0 · π_GD + π_H·(1−w_GD)
                                             = (1−w_GD)·π_H   (hab only)
```

**Action required:** Add `mot` to the mixing formula in §2.3.3 and explain it in §2.5.1 (Devaluation): "for devaluation trials, `mot` is set to 0, zeroing the goal-directed system's contribution to the policy while leaving its weights intact."

---

## D9 — Learning phase checkpoint criterion is 60%, not 80%

**Methods §2.4:** "learning criterion … ≥ 80% correct"  
**Methods H1 (§2.7):** "Pass: ≥ .80, p < .001"

**Code (config.py):**
```python
learn_combined_min: float = 0.60  # ckpt_learn saved when combined >= 0.60
learn_habsolo_max:  float = 0.60  # AND hab-solo still <= 0.60
maint_solo_min:     float = 0.80  # ckpt_maint saved when hab-solo >= 0.80
```
The **learning-phase checkpoint** (ckpt_learn) is captured at 60% combined accuracy (early in training when the habitual system is not yet competent). The 80% threshold applies to H1 (final evaluation, ckpt_maint) and to the **maintenance-phase checkpoint**. The methods conflate the two.

**Action required:** Distinguish the checkpoint criteria from the H1 pass criterion. In §2.4: "the learning-phase checkpoint is saved when combined accuracy first exceeds 60% while habitual-solo accuracy is still below 60%, capturing the period when the goal-directed system is the primary controller; the maintenance checkpoint is saved when habitual-solo accuracy reaches 80%."

---

## D10 — expr_gate scaling of GD loss not mentioned

**Code (train.py):**
```python
expr_gate = w_b.mean().detach().clamp(0, 1)
gd_b = expr_gate * (policy_l + entropy_l) + critic_l + da_pen
```
The GD policy and entropy losses are scaled by `expr_gate` (the current mean mixing weight w_GD). When the habitual system has taken over and w_GD is near zero, the GD system's policy gradient is nearly zeroed out — it can maintain its value estimate and pay the DA penalty, but its policy is no longer updated. This is an important training detail with implications for why the GD weights remain intact (they stop receiving policy gradient once they're dormant).

**Action required:** Add a sentence in §2.4 or §2.3.3: "the goal-directed policy gradient is scaled by the current mixing weight w_GD, so that once the handoff occurs the GD system's policy weights receive negligible gradient — preserving the learned solution in a dormant but intact state."

---

## D11 — Habitual training does not use "APE" in the standard sense

**Methods §2.3.2:** "its teaching signal is an action prediction error — the mismatch between the expected and the executed action (Greenstreet et al., 2025)"

**Code (train.py):**
```python
ce_b = F.cross_entropy(pi_h_b, acts_b, reduction="sum")          # "APE"
logp_h = Categorical(logits=pi_h_b).log_prob(acts_b)
hab_b = cfg.ape_weight * ce_b + cfg.eff_weight * (-(adv_int.detach() * logp_h).sum())
```
The "APE" is implemented as **cross-entropy of the habitual policy against the action taken** (which was sampled from the combined policy, not the habitual policy alone). This is an imitation / behavioral cloning loss on the combined policy's actions, not a biological APE signal. The action `acts_b` was chosen by `Categorical(logits=combined)`, so the habitual system is learning to predict what the combined system (partly goal-directed) chose — not a pure action prediction error on its own policy.

**Action required:** Clarify the training signal precisely: "the habitual system is trained by cross-entropy against the actions selected by the combined policy (behavioral cloning), plus an efficiency term from intrinsic returns. This approximates an action prediction error but differs from a true local APE in that the target actions are influenced by the goal-directed system."

---

## D12 — Software list in §2.9 is wrong

**Methods §2.9:** "model.py, environment.py, train.py, **params.py**, **visualize.py**"

**Code (actual files):**
- `config.py` (not `params.py`)
- `analysis.py` (not listed)
- `figures.py` (not listed)
- `run_experiment.py` (not listed)
- `aggregate.py` (not listed)
- No file named `params.py` or `visualize.py` in `Code/main/`
  (the visualizer is `Code/visualisations/make_viz.py`)

**Action required:** Update §2.9 to list the actual files.

---

## Summary table

| # | Severity | Topic | Methods says | Code does |
|---|----------|-------|-------------|-----------|
| D1 | **CRITICAL** | Obs dimensions & split | 22D alloc (GD) / 14D ego (Hab) | 6D identical for both |
| D2 | **HIGH** | Number of task phases | 3 (sample, delay, test) | 4 (pre_sample, sample, delay, choice) |
| D3 | **HIGH** | Delay phase movement | Agent held in start box | Agent moves freely |
| D4 | **HIGH** | Sample phase | Agent forced + receives reward | Navigates freely, arm_end_bonus is shaping only |
| D5 | **HIGH** | w_GD formula | σ(α · DA_total + bias) | σ(α · da_request + bias) |
| D6 | **HIGH** | DA gain scope | Both systems | GD system only |
| D7 | MEDIUM | DA gain level | W_eff = f(DA)·W (weight) | gain · tanh(h) (activation) |
| D8 | MEDIUM | Devaluation mechanism | Not in mixing formula | mot factor in combined = w·mot·π_GD + (1−w)·π_H |
| D9 | MEDIUM | Learning checkpoint | 80% criterion | 60% checkpoint, 80% for H1 pass |
| D10 | LOW | expr_gate training | Not mentioned | GD policy loss scaled by w_GD |
| D11 | MEDIUM | Hab training signal | APE = action prediction error | Cross-entropy against combined policy actions |
| D12 | LOW | Software list §2.9 | params.py, visualize.py | config.py, analysis.py, figures.py, run_experiment.py |
