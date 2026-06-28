# Mémoire: Method Section

## 2.1 Overview

This study is a computational reproduction of Villet et al. (2025). We ask whether four biologically motivated design choices — an allocentric/egocentric observational asymmetry between the two systems, a low-rank recurrent structure in the habitual system, a value-free action-prediction-error learning rule for the habitual system, and a dopamine-gated expression gain on the goal-directed system — are jointly sufficient to reproduce the full Villet lesion, devaluation, and reactivation pattern. Each design choice is tested by an ablation in which it is removed while the others are held fixed; the model with all four active is reported as the main result. The biological motivation for each choice is given in §1; the present section states the choices and their operationalisations without re-arguing them.

---

## 2.2 Hypotheses

The following hypotheses are stated here, before the architecture, because each design choice in §2.4 is motivated by the hypothesis it is intended to support. All are derived from Villet et al. (2025) and translated into computationally testable predictions.

**H1 — Learning (prerequisite).** The full model reaches ≥ 80% correct on the non-match task at a non-trivial delay. This is a gate: if the model does not learn the task, no subsequent result is interpretable.

**H2 — Emergent, not scheduled, handoff.** Control transfers from the goal-directed to the habitual system as, and only as, the habitual system becomes competent. The onset of the handoff (defined by the drop in w_GD) should follow the onset of habitual competence (habitual-solo accuracy ≥ 99%) across seeds, and the inter-seed variance in handoff timing should track the inter-seed variance in learning speed rather than a fixed episode count. This is the primary evidence that the transfer is emergent rather than built into the training schedule.

**H3 — Devaluation dissociation.** Setting motivation to zero (simulating reward devaluation; see §2.6) impairs accuracy at the learning-phase checkpoint, when the goal-directed system is dominant, but not at the maintenance-phase checkpoint, when the habitual system is dominant.

**H4 — Lesion × phase dissociation.** Silencing the goal-directed system (zeroing its hidden state) impairs performance at the learning checkpoint but not at maintenance; silencing the habitual system leaves maintenance accuracy intact; silencing both collapses performance. This reproduces Villet's chemogenetic inhibition pattern.

**H5 — Reactivation and emergence falsification.** At the maintenance checkpoint, silencing the habitual system (a) preserves accuracy, (b) causes the goal-directed system's DA-request signal to rise back toward its learning-phase level, and (c) restores devaluation sensitivity. This is the mechanistic signature of a genuinely dormant-but-intact goal-directed solution. It is treated explicitly as a falsification: if the DA-request does not rise when the habitual system is removed, the handoff was a function of episode count rather than habitual competence, and this outcome is reported as a negative result without qualification.

**H6 — Working-memory attractor.** During the delay period, the habitual system's hidden states form distinct, arm-specific clusters that are linearly decodable with high accuracy, despite the phase signal having decayed to near zero. This confirms that the system maintains a learned internal working-memory representation across the delay — a genuine attractor, not a persisting input.

**H7 — Untrained negative control.** Untrained networks (random initialisation, ≥ 5 seeds) show chance accuracy, no decodable delay representation, and no handoff timecourse, confirming that the results in H1–H6 reflect learned rather than architectural structure.

---

## 2.3 Task: delayed non-match-to-place T-maze

We simulate the DNMTP T-maze of Villet et al. (2025) on a parametric grid (width 9, height 6; `len_edge = 7`, `difficulty = 2`). Each trial comprises four phases.

**Pre-sample.** The agent starts at the bottom of a central stem and navigates freely upward to a junction at the top. Reaching the junction triggers the sample phase and delivers a shaping bonus (`junction_bonus = 0.30`) to provide dense credit signal during early training.

**Sample.** One arm is physically blocked (the corresponding cells become impassable). The agent navigates freely to the end of the open arm; reaching the arm end ends the sample phase and delivers a second shaping bonus (`arm_end_bonus = 0.20`). The arm is blocked but not scripted: the agent makes a movement decision at every step.

**Delay.** The agent is pushed back along the maze (arm end → junction → stem → start position) by programmatic action overrides, then held at the start position for the remainder of the delay period. No step cost is charged during the pushback or hold. A phase signal that was present during sample decays by a factor of 0.1 per step, so that by the time the choice phase begins the signal is negligible (< 10⁻³ of its peak value); correct choice therefore requires retaining "which arm was visited" in working memory across the delay. The delay length is fixed at evaluation (`fixed_eval_delay = 40` steps); during training a curriculum advances the delay from `delay_start = 8` to the evaluation ceiling as habitual-solo accuracy exceeds a threshold.

**Choice.** The agent is released from the start position. Both arms are accessible. A test reward (`test_reward = 1.0`) is delivered for entering the arm that was blocked during sample (the non-match rule). A small per-step cost (`step_cost = wait_cost = 0.02`) is charged for all actions including WAIT, pressuring the agent toward efficient trajectories. The action set is {N, S, E, W, WAIT} (five actions).

---

## 2.4 Model architecture

The agent is a dual-system model in which two continuous-time recurrent networks (CTRNNs; Beer, 1995) contribute to a combined policy at every timestep. Each network's hidden state evolves as:

  **τ dh/dt = −h + W h + W_in x + b**

with a pointwise tanh nonlinearity on the recurrent term and learnable time constants τ per unit, implemented in discrete time as `h ← h + (dt / τ) · dh`. Both networks inject Gaussian noise into the hidden state during training (`noise_std = 0.05`) to regularise the dynamics.

### 2.4.1 Goal-directed system (mPFC / DMS analogue)

The goal-directed system receives a **6-dimensional allocentric observation**: `[x/(w−1), y/(h−1), 0, sig_L, sig_R, sig_choice]`, where x and y are the agent's normalised grid coordinates, sig_L and sig_R are arm-identity signals active during the sample phase, and sig_choice is a low-amplitude phase indicator active at choice onset. The spatial coordinates give the system access to the agent's full location in the maze, enabling location-based planning.

The network has `n_gd = 512` units, split into two subpopulations (Naudé et al., 2024): a **widen** subpopulation (first 256 units, initialised with short time constants `τ_fast = 5`) operating at the action and decision timescale, and a **deepen** subpopulation (remaining 256 units, `τ_slow = 25`) operating at the maintenance timescale. Input weights are initialised with standard deviation 1.0 (input-dominated initialisation); recurrent weights with `0.9 / √n`.

Dopamine modulates the **output expression** of all units: before the output readout, each unit's activation is scaled by a per-unit gain,

  **r_out = (gain_base + gain_DA · da_comp) · tanh(h)**,

where `da_comp` is `da_request` (phasic) for widen units and `da_tonic` (a slow low-pass of `da_request`, decay rate κ = 0.10) for deepen units. `gain_base = 0.5` sets a non-zero floor; `gain_DA = 0.5` determines the range of DA modulation. This gain scales the effective contribution of intact learned activations to the policy without altering the weight matrices — the property that makes rapid reactivation possible (§1.3).

The network emits action logits, a scalar value estimate, and a scalar **DA-request** signal:

  **da_request = σ(h · w_da + b_da)**

The DA-request feeds back to set the system's own expression gain and the policy mixing weight (§2.4.3). An L2 penalty on `da_request²` in the training objective (§2.5) pressures the goal-directed system to minimise its own dopamine demand. It can do so without sacrificing reward only once the habitual system's policy is competent — the mechanism for the emergent handoff (H2).

The goal-directed system is trained by **advantage actor-critic with generalised advantage estimation** (A2C-GAE; Mnih et al., 2016; `γ = 0.99`, `λ_GAE = 0.95`) on the task reward, with return normalisation (`ret_norm_window = 10,000`). The policy and entropy losses are additionally scaled at each batch by the current mean mixing weight w_GD (see §2.4.3), so that once the handoff occurs and w_GD is near zero, the goal-directed policy gradient is nearly zeroed — preserving the learned solution in a dormant but intact state.

### 2.4.2 Habitual system (DLS analogue)

The habitual system receives a **4-dimensional position-free observation**: `[0, sig_L, sig_R, sig_choice]` — the three phase signals only, with the spatial coordinates withheld. By construction the system has no knowledge of its location in the maze and cannot represent or plan a spatial rule such as non-match-to-place.

The network has `n_hab = 128` units. When `hab_rank > 0`, the recurrent weight matrix is **low-rank**, parameterised as `W_rec = (M · N^T) / n_hab` with M and N of shape `[n_hab, rank]`, initialised with standard deviation 0.1. This makes the claim that habitual motor sequences are low-dimensional structural rather than something to be recovered post-hoc, and substantially reduces the parameter count of the system most expected to be stereotyped. In the full-rank ablation (§2.6, Ablation 4), the constraint is removed.

The habitual system has two opponent output readouts following the OpAL\* architecture (Jaskir & Frank, 2023):

  **π_H = r_out · W_Go^T − r_out · W_NoGo^T**

Crucially, the habitual system is trained **entirely without task reward**. Its learning signal is an action prediction error: a per-step KL divergence of the habitual policy against the combined policy's action distribution (a soft behavioural cloning target), plus a small intrinsic completion bonus for reaching arm ends (`eff_weight = 0.10`). Because reward never enters the habitual objective, the system cannot, in principle, change its behaviour when reward is devalued — the structural source of the devaluation insensitivity observed in H3.

### 2.4.3 DA-gated mixing

The DA-request signal emitted by the goal-directed system drives the policy mixing weight:

  **w_GD = σ(α · da_request + bias)**

with `α = 6.0` and `bias = −2.0`, so that w_GD ≈ 0 when da_request ≈ 0 (habitual system dominant) and w_GD ≈ 1 when da_request is high (goal-directed system dominant). The combined policy is:

  **combined = w_GD · mot · π_GD + (1 − w_GD) · π_H**

where `mot` is a motivational scaling factor, equal to 1 during normal trials. For devaluation evaluation trials, `mot = 0`, zeroing the goal-directed system's contribution to the combined policy without altering any weights. This implements Villet's reward devaluation manipulation: a value-sensitive system (goal-directed) will show reduced accuracy; a system that never represented reward (habitual) is unaffected.

---

## 2.5 Training procedure

A single network is trained for up to 200,000 episodes (batches of `B = 128` parallel environments), with early stopping when the rolling combined accuracy over the last 5,000 episodes exceeds 99%. The habitual system is updated at every timestep (per-step KL backward pass, one optimiser step per step); the goal-directed system is updated at episode end (A2C-GAE over the full episode trajectory). Separate optimisers and learning rates are used: `lr_GD = 10⁻⁴` (goal-directed, Adam), `lr_hab = 10⁻³` (habitual, Adam). Gradient norms are clipped to 0.5. The DA-recruitment penalty (`da_cost_lambda · da_request²`) is phased in after a warmup of 100 training iterations.

Two checkpoints are saved during training for use in the phase-crossed manipulations (§2.6). The **learning-phase checkpoint** (`ckpt_learn`) is saved when combined accuracy first exceeds 70% while habitual-solo accuracy is still ≤ 70%, capturing the period when the goal-directed system is the primary controller and the habitual system has not yet become competent. The **maintenance-phase checkpoint** (`ckpt_maint`) is saved at the last evaluation where habitual-solo accuracy reaches ≥ 80%, capturing the overtrained state.

All reported results use ≥ 5 independently seeded runs. Single-seed results are not interpreted given the high cross-seed variance in handoff timing (coefficient of variation ≈ 37% in pilot runs).

---

## 2.6 Experimental conditions

### Main model

The main model activates all four design choices: ego/allo observational split, low-rank habitual network (`hab_rank = 4`), value-free APE training rule, and DA output gain (`gain_DA = 0.5`). This is the model evaluated against H1–H7.

### Ablation 1 — Remove the observational asymmetry

The habitual system receives the same 6-dimensional allocentric observation as the goal-directed system (`habit_obs = "allocentric"`), including the spatial coordinates. This tests whether positional exclusion is necessary for the functional distinction: specifically, whether a habitual system with full positional access learns a more flexible (non-match) rule, breaks the devaluation dissociation, or shows a different handoff timecourse.

### Ablation 2 — Remove the value-free learning rule

The habitual system is trained with an additional A2C term on task reward (`habit_rule = "value_coupled"`), so that reward enters the habitual objective. The prediction is that devaluation sensitivity is restored at maintenance (H3 fails), because the habitual system now represents the value of outcomes and can, in principle, adjust its policy when they change.

### Ablation 3 — Remove the DA output gain

The expression gain on the goal-directed output is fixed at its baseline value (`gain_DA = 0`), so that dopamine has no effect on the strength of the goal-directed policy's expression. The weights, the mixing weight w_GD, and the DA-request signal are all unchanged. The prediction is that reactivation fails (H5 fails): the dormant goal-directed solution exists in the weights but cannot be reinstated by a rise in DA-request because the gain mechanism that would amplify its expression has been removed.

### Ablation 4 — Remove the low-rank constraint

The habitual system uses a full-rank recurrent weight matrix (`hab_rank = 0`). This tests whether the structural low-dimensionality constraint matters for the quality of habitual performance, the speed of the handoff, or the attractor geometry of the delay representation.

---

## 2.7 Analysis pipeline

### Behavioural readouts

The primary measure is proportion correct on frozen-policy test trials, evaluated for the combined system, the goal-directed system alone (w_GD forced to 1), and the habitual system alone (w_GD forced to 0). The w_GD and da_request timecourses across training are logged every 500 episodes.

### Handoff timing (H2)

The habitual-onset episode (`hab_onset`) is the first evaluation at which habitual-solo accuracy reaches ≥ 99%. The handoff-onset episode is the first evaluation at which w_GD has dropped by ≥ 50% of its peak value. The cross-correlation and lag between the two timecourses, and the across-seed spread of both onset episodes, are compared: an emergent handoff should show a consistent lag (handoff follows competence) with variance tracking learning speed; a scheduled handoff would show a fixed onset episode regardless of habitual competence.

### Devaluation dissociation (H3)

With `mot = 0` (and the policy otherwise frozen), accuracy is measured at `ckpt_learn` and `ckpt_maint`. A phase × devaluation interaction is tested; the prediction is a significant drop at the learning checkpoint and no significant drop at maintenance.

### Lesion × phase dissociation (H4)

The goal-directed hidden state, habitual hidden state, or both are zeroed at evaluation, crossed with `ckpt_learn` and `ckpt_maint`. Accuracy under each silencing condition reproduces Villet's inhibition pattern.

### Reactivation probe (H5)

At `ckpt_maint`, the habitual system is silenced (h_hab zeroed). Before and after silencing, we measure (a) combined accuracy, (b) the goal-directed DA-request signal, and (c) devaluation sensitivity (`mot = 0` after silencing). Pass: accuracy is preserved, DA-request rises, devaluation sensitivity is restored. Fail (reported without qualification): DA-request does not rise — the handoff was a schedule, not a competence-locked transfer.

### Working-memory representation (H6)

Delay-period hidden states of the habitual system are projected with PCA and visualised with one point per trial coloured by the sampled arm. A logistic regression decoder (5-fold cross-validated) reads arm identity from delay-period hidden states; accuracy is compared against the 50% chance baseline. Because the phase signal decays to near zero during the delay, any decodable structure must reflect a learned internal representation rather than a persisting input — a genuine working-memory attractor. Pass: visible arm-specific clustering and decoder accuracy ≥ 90%, p < .001 (binomial).

### Untrained negative control (H7)

The full analysis pipeline (H1–H6 measures) is re-run on untrained networks (random initialisation, ≥ 5 seeds) at `delay = 40`. The prediction is chance accuracy, no decodable delay representation, and no handoff timecourse.

### Statistical reporting

α = .05 throughout; tests are one-tailed where theory fixes the direction. H1 is tested with a one-sample binomial against p₀ = .50 on N = 1,000 frozen-policy trials, reporting proportion correct, exact p, and 95% CI. H6 decoder accuracy is reported with 95% CI from the cross-validated fold distribution.

---

## 2.8 Software and reproducibility

The model, environment, training loop, and analysis pipeline are implemented in Python (PyTorch) across the following files: `model.py`, `environment.py`, `train.py`, `config.py`, `analysis.py`, `figures.py`, `run_experiment.py`. All reported results use fixed random seeds; full code and configuration are provided in the Annex to permit replication.

---

## 2.9 Ethics and data statement

This work is purely computational; no new human or animal data were collected. The biological paradigm reproduced in simulation is taken from published work (Villet et al., 2025), conducted under that study's institutional approvals.
