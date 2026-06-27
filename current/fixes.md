---
title: Issues to Fix — Post-Scrutiny
updated: 2026-06-27
---

# Issues to Fix

Compiled after Opus-level scrutiny of results, code, and methodology. Each entry has a **Problem** description, then space for **Your opinion** on the fix and **Claude's take** on that opinion.

---

## F1 — Greedy decoding hides graded learning

**Problem:** `evaluate_vec` always uses `argmax` instead of sampling. A model that is 55% confident correct looks identical to one that is 99% confident — both produce the same greedy action, and the only change visible in the training curves is a step from 0 to 1. The real gradual learning is happening in logit space but is invisible to the accuracy metric. This is why `hab_solo_acc` looks like a step function and why the accuracy figures are uninteresting.

**Your opinion:** Three ideas. (1) Use Euclidean distance per direction (NSEW) as a continuous signal instead of binary correct/incorrect — but unsure whether the vectorised env would need to become a 2D env, or whether a translation layer could preserve performance. (2) Replace the separate frozen-weight eval phase entirely with a rolling ratio: n_successful_trials / last_100_training_trials. (3) Change the habitual APE loss from predicting which discrete action was chosen to predicting the distance actually travelled in a given direction — making it a continuous signal.

**Claude's take:**

**(1) Euclidean distance metric.** The environment is already 2D — `env.pos` is an (x,y) coordinate, tracked in both the single-env and vectorised versions. `rollout` already records `rec_pos` at every step. No restructuring needed; the position is already there. The practical question is what to measure the distance *to*: the most natural choice is the final position's distance to the correct arm endpoint vs the incorrect arm endpoint, giving a continuous score in [0,1] after normalisation. This would genuinely fix the binary-accuracy problem and give a graded signal during training. Downside: it conflates "got close to the right arm" with "solved the task" — an agent that goes to the wrong arm's entrance and then backtracks could score well on distance while still failing. For the training curve it's an improvement; for H1 (final competence) binary accuracy is still the right claim.

**(2) Rolling accuracy, no frozen eval.** Good idea for the training-curve figures — removes the fixed-seed problem and the greedy-vs-sampling mismatch entirely, because you're just counting outcomes from the actual training rollouts (which already use sampled actions). This directly addresses F1 and F2 simultaneously. The catch is that frozen-weight eval is needed for the *hypothesis tests* (H3 devaluation, H4 lesions, H5 reactivation, H6 decoder) — those require running the model under specific intervention conditions (mot=0, lesion="hab", etc.) that can't be done during a live training rollout. So the proposal would be: use rolling training accuracy for the fig1 curves, keep the separate eval phase for the hypothesis tests at the checkpoints. Don't abolish frozen eval entirely, just demote it from the primary training metric to a checkpoint-only tool.

**(3) APE as distance travelled.** This is a much bigger change — it modifies the habitual learning rule, not just the evaluation metric. Currently APE is cross-entropy between the habitual policy and the combined policy's chosen action (the habit learns to imitate what the combined system does). Replacing it with a distance-based signal (how far the agent actually moved in each direction) makes the habit learn a forward model of physical displacement rather than an action classifier. This is conceptually interesting — it could make the habit more robust to policy changes in the GD system, because it's predicting physical outcomes rather than mimicking discrete choices. But it changes what H3 (devaluation dissociation) means: the habit would no longer be value-free in the sense of "never saw the reward signal" — distance-to-goal is implicitly reward-shaped if the goal is in a fixed location. Worth flagging that this might reintroduce value-sensitivity through the back door, which would break the theoretical grounding of the devaluation result.

---

## F2 — Fixed eval seed makes evaluation deterministic

**Problem:** `evaluate_vec` always constructs the trial sequence with `rng = np.random.default_rng(0)`. Every call — across all seeds, all experiments, all training checkpoints — evaluates on the **same** fixed sequence of L/R-blocked trials. Combined with greedy decoding this makes evaluation fully deterministic given the weights. This is why E3 `gain_only` reproduces exactly `0.471` across two different runs. It also means the H1 binomial confidence interval and p-value are statistically meaningless: the test assumes i.i.d. trial sampling, but the trials are fixed and the policy is deterministic.

**Your opinion:** Vary the seeds across eval calls. Also add noise to the entire project end-to-end, following the supervisor's implementation.

**Claude's take:**

**(Varying seeds)** Correct fix. The minimal change is to remove the hardcoded `rng = np.random.default_rng(0)` inside `evaluate_vec` and instead pass the caller's rng in, or seed from the current episode count, so each eval call sees a different trial sequence. This turns the evaluation from a deterministic constant into a proper stochastic estimate — which is what the confidence interval and p-value were assuming all along.

**(Adding noise)** Also correct, and the supervisor's implementation is exactly the right pattern. In both `GDNet.step()` and `HabNet.step()`, after computing `dh`, add: `if self.training: dh = dh + torch.randn_like(h) * noise_std`. The supervisor uses `noise_std = 0.05` and zeros it during eval (`noise_scale = self.noise_std if self.training else 0.0`). This is standard in CTRNN neuroscience models — it prevents the network from relying on perfectly deterministic trajectories and acts as a regulariser, making representations more robust and distributed. It needs to be added to `config.py` as a parameter and to both `GDNet` and `HabNet` in `model.py`.

**Important caveat:** noise zeroed during eval does NOT fix the binary-accuracy problem from F1. Since noise is off during eval, evaluation with frozen weights remains deterministic (given the weights), so accuracy is still a step function. Noise improves training dynamics but doesn't make the eval curves graded. F1 and F2 still need their own fixes (sampled actions or rolling training accuracy) even after noise is added.

---

## F3 — H5 accuracy "recovery" is a mathematical identity, not reactivation

**Problem:** When `lesion="hab"` is applied, `h_hab` is zeroed every step, so `tanh(0) = 0`, so `pi_hab = 0` (the zero vector, because there is no output bias). The combined logits become `w_gd * pi_gd + (1-w_gd) * 0 = w_gd * pi_gd`. Since `w_gd > 0`, `argmax(w_gd * pi_gd) = argmax(pi_gd)` — the gate value is irrelevant. Under greedy decoding, silencing the habitual system is **mathematically equivalent to GD-solo** regardless of the gate, regardless of DA, regardless of any mechanism. Accuracy snapping to 1.000 is a consequence of this identity, not of dormant weights re-expressing. The current Results text says this "confirms the goal-directed policy remained available for expression" — which implies the expression-gain mechanism did something. It didn't.

**Your opinion:** Rewrite H5 to state the mathematical identity plainly. Future Research section will point to the naude-hab-ctrnn project (Documents/naude-hab-ctrnn): Naudé et al.'s model as the GD system, with APE-based habit grafted onto it and dopamine released as in Naudé — this is the architecture where the reactivation test becomes genuinely informative.

**Claude's take:** Right call. Stating the identity plainly is more credible to a committee than the current framing, which gives the impression of a near-pass when the result is actually uninformative. The Future Research pointer is scientifically precise: the problem with the current H5 test is that zeroing `h_hab` collapses to a trivial argmax identity, whereas in Naudé's architecture the DA dynamics are embedded in the recurrence in a way that actually changes what the GD system does when the habit is removed — reactivation would be a non-trivial observable there.

---

## F4 — H5 DA-request null is over-determined (two separate causes)

**Problem:** Even if F3 weren't a problem, the DA-request rise would still be +0.000, for two independent reasons: (a) the training loss contains `da_pen = da_lambda * (da_t² * valid_f).sum()`, which explicitly drives DA-request toward zero — so by maintenance checkpoint, DA-request is near 0 in both conditions; (b) `da_request` is a readout of `h_gd` only, and `h_hab` has no architectural path to it, so silencing the habitual network cannot raise the DA signal through any direct mechanism. The H5 null is not a finding about the dormant-trace mechanism — it is an artifact of how training and the architecture are constructed.

**Your opinion:** Apply the expression gain only to the 5-action output weights, not to the DA-request output neuron. The penalty should be on the gain effect on actions, not on `da_request` itself.

**Claude's take:** This is already half-true in the current code — `r_out = gain * tanh(h)` feeds `pi` (actions) and `da_request = sigmoid(h · w_da)` is computed from raw `h` with no gain applied. Gain structurally never touches the DA-request readout. The problem is the cost penalty: `da_pen = da_lambda * (da_t² * valid_f).sum()` penalises `da_request` itself, which drives the signal generating the gain all the way to zero. The fix is to change the penalty target from `da_t²` to the gain expression on the action outputs — roughly `(gain_da * da_t)²` — so that the model learns "don't let DA excessively amplify action outputs" rather than "suppress DA-request entirely." This frees `da_request` to be a genuine signalling neuron (able to rise when the habitual system is removed) while still regularising the expression gain amplitude. A one-line change in `train.py` once the architecture is ready for it.

---

## F5 — wgd_drop timing tracks the penalty schedule, not habitual competence

**Problem:** `wgd_drop_onset` is 116k–128k episodes across all 5 E0 seeds. The DA penalty finishes ramping at `da_warmup + da_ramp = 64k + 64k = 128k` episodes. The gate closes because the penalty crushes `da_request`, and `w_gd = sigmoid(alpha * da_request + bias)` falls to `sigmoid(-2) ≈ 0.12` as `da_request → 0`. The timing is set by the penalty schedule, not by the model deciding it no longer needs goal-directed control. This makes E2's "expression vs scheduled" comparison weaker: the expression arm is also following a schedule, just implicitly through the penalty ramp. The H2 "emergent handoff" claim applies to *whether* the handoff occurs (driven by the DA-request mechanism) but the *when* is schedule-determined.

**Your opinion:** Replace the penalty ramp entirely with an EMA of |advantage| as the DA signal — neurochemically grounded in the RPE account, emergent timing, no schedule.

**Claude's take:** See tradeoff analysis in the References & Tradeoffs section at the bottom of this file.

---

## F6 — hab_onset comparisons across E2–E9 are from n=1 seeds and uninterpretable

**Problem:** Every ladder experiment (E2–E9) has exactly 1 seed. E0's own 5-seed spread for `hab_onset` is 7,808–21,376 — a range of ~13,568 episodes and a CV of 37%. The differences being discussed across experiments are much smaller than this natural variance: for example, E6 uniform_tau (2,816) vs E9 naude_full (3,200) differ by only 384 episodes. E8's rank sweep shows no monotone relationship between rank and onset. None of the hab_onset orderings in E2–E9 are statistically interpretable.

**Your opinion:** Run more experiments from scratch. Previous seeds are impossible to consider comparable given the methodological issues (F1, F2) that were present when they were collected.

**Claude's take:** Correct call. The 37% CV in hab_onset from E0's 5 seeds means any experiment-level comparison needs at minimum 5 seeds with identical conditions — which the current E2–E9 runs don't have. More importantly, the F1/F2 fixes (rolling accuracy, noise injection) change the dynamics enough that results from the old codebase and results from the fixed codebase aren't the same quantity. The right sequence: fix F1 and F2 first, then rerun all ladder experiments (E0–E9) from scratch with ≥5 seeds. hab_onset comparisons will only be interpretable once the variance estimate is re-established under the corrected code.

---

## F7 — E3 weights_only is inconsistent across runs (seed instability)

**Problem:** E3 `weights_only` gives acc=0.471 in the 22:32 run and acc=0.000 in the 23:09 run, same commit, 1 seed each. The 0.471 is "always enter one arm" behavior (a degenerate policy); 0.000 is "never reach an arm" (timeout). Both are degenerate attractors. The inconsistency shows that with 1 seed, single-run results for partial or failed conditions are not reproducible. The 0.471 should not be interpreted as partial competence.

**Your opinion:** Covered by previous fixes.

**Claude's take:** Yes — F7 is F6 instantiated in E3. The inconsistency between the two runs (0.471 vs 0.000) is exactly the n=1 seed variance F6 identifies. Rerunning E3 with ≥5 seeds under the fixed codebase (F1+F2 applied) will reveal whether weights_only is a genuine failure mode or just unlucky attractor selection. No additional fix needed beyond F6.

---

## F8 — w_gd and da_recruit traces in fig1 should be smoothed

**Problem:** These are continuous signals logged as per-eval means (each eval is 200 trials). They carry genuine noise — the gate and DA-request fluctuate trial-to-trial within an eval window. The raw traces in fig1 are jagged in a way that makes the dynamics harder to read. A rolling mean over ~20–50 evaluation points (2,000–5,000 episodes) would improve readability without misrepresenting the signal. Unlike accuracy (which is a genuine step function), these are genuinely continuous trajectories that smoothing would clarify.

**Your opinion:** Already fixed.

**Claude's take:** Confirmed — the visualiser applies EMA smoothing (α=0.15) to the H2 traces with raw per-eval dots shown underneath (`visualiser.html:128`). No further action needed.

---

## F9 — Results text overstates what H5 accuracy recovery means

**Problem:** Section 3.5 currently says accuracy recovery "confirm[s] that the goal-directed policy remained intact and was available for expression" and presents devaluation-sensitivity returning after silencing as a further positive finding. Given F3 above, both of these are consequences of the mathematical identity, not of the mechanism. The write-up needs to clearly state that these results cannot distinguish genuine reactivation from the decoding artifact, and that the gate-clamp control (cited as additional support) is subject to the same identity.

**Your opinion:** Covered by F3.

**Claude's take:** Correct — F9 is the writing-level symptom of F3. The fix is identical: state the argmax identity plainly and remove any implication that the expression-gain mechanism produced the recovery. No separate action needed once F3's rewrite is done.

---

## F10 — Experimental protocol is organised around the wrong question

**Problem:** The three core scientific claims are: (1) a handoff occurs — GD leads early learning, habitual competence rises later; (2) GD is the learning area — the habitual system is not needed for learning; (3) the habitual system is reward-insensitive unlike GD. The current protocol does not directly test claim (2): no run freezes the habitual network during training and checks that GD still reaches criterion. The E2 `expression` vs `scheduled` arm sweep tests a different question entirely — which gating mechanism is more efficient — and produces no evidence for any of the three claims. E4 (value-free vs value-coupled) does test claim (3) but is framed as a hyperparameter sweep rather than a falsification test.

**Your opinion:** Drop the expression/scheduled arm from E2. Add a freeze-hab ablation (train with no gradient updates to hab; does GD still reach criterion?). Keep E4 but reframe it explicitly as the reward-insensitivity falsification.

**Claude's take:** Correct on all three. Claim (1) is already evidenced by the `gd_acc` / `hab_solo_acc` crossover curves already logged in every run — no new data needed, only a cleaner figure and framing. Claim (2) requires the missing ablation: a `--freeze-hab` flag that zeros `opt_hab` gradient steps during training; if GD reaches criterion anyway, the dissociation is established. This is a small code change (skip `opt_hab.step()` when the flag is set) and one extra arm in `batch_runner.py` for E2. Claim (3) is covered by E4 data already collected; the fix is entirely in how the result is written up — present it as "value-coupled fails where value-free succeeds" rather than as a sensitivity analysis. The expression/scheduled sweep can be dropped from the mémoire entirely; it is not load-bearing for any of the three claims.

---

## F11 — No experiment tests DA-modulated plasticity

**Problem:** The current DA implementation is purely an *expression* mechanism — at inference time, DA scales the output readout (`r_out = gain · tanh(h)`), which amplifies action logits. DA does not touch the learning process itself: gradient updates to `W`, `W_in`, `W_out` happen at the same rate regardless of DA level. Naudé et al. (2024) show that DA modulates NMDA receptor conductance, which gates Hebbian synaptic plasticity — high DA makes synapses more plastic (easier to strengthen or weaken), low DA makes them more stable. This is a different and complementary role to expression gain. The current model has no experiment that tests it.

**Your opinion:** Add an experiment where plasticity is influenced by DA — like Naudé but with a much simpler implementation.

**Claude's take:**

The simplest implementation that captures the core idea without touching the NMDA biophysics: **scale the GD gradient by a function of `da_tonic` before the optimizer step.** Concretely, in `train.py`:

```python
# Before opt_gd.step(), scale all GD gradients by DA level
da_scale = cfg.da_plasticity_base + cfg.da_plasticity_gain * da_tonic
for p in model.gd_params():
    if p.grad is not None:
        p.grad.mul_(da_scale)
opt_gd.step()
```

When `da_tonic` is high (early training, high prediction error), `da_scale` is large → GD weights update faster → the goal-directed system consolidates representations quickly. When `da_tonic` is low (task mastered, low prediction error), `da_scale` shrinks → GD weights barely move → the learned policy is frozen and stable. This is the Naudé intuition: DA doesn't just amplify expression, it controls when synapses are allowed to change.

**What this would show:** If DA-modulated plasticity is the right account, the GD system should consolidate faster (fewer episodes to reach criterion) and be more stable post-handoff (less drift in `w_gd` after the habitual system has taken over). The experiment would be: compare `da_plasticity=True` vs `da_plasticity=False` on hab_onset timing and post-handoff `w_gd` stability across 5 seeds each. If DA-gated plasticity genuinely helps, the wgd trace should be cleaner and hab_onset should be earlier — because the GD system consolidates its policy faster when DA is high, freeing the habitual system to take over sooner.

**Why simpler than Naudé:** Naudé's implementation requires a full biophysical model with AMPA/NMDA conductance ratios, slow NMDA time constants, and separate D1/D2 receptor pathways. The gradient-scaling version achieves the same functional outcome — "more plastic when DA is high" — with a single multiplicative scalar per optimizer step. Two new config parameters (`da_plasticity_base`, `da_plasticity_gain`) and four lines of code.

**Dependency:** This experiment makes most sense once F5 is fixed (EMA-of-|advantage| DA signal). With the current penalty ramp, `da_tonic` is zero for most of training, so plasticity scaling would be near zero and the GD system would barely learn. The experiment should be run with the RPE-based DA signal so that plasticity tracks genuine learning dynamics.

---

## References & Tradeoff Analysis (F5)

### Papers cited in the F5 discussion

**On DA as reward prediction error (RPE) — the core grounding:**
- Schultz, W., Dayan, P., & Montague, R.R. (1997). A neural substrate of prediction and reward. *Science*, 275, 1593–1599. — The founding paper. DA neurons fire on unpredicted reward, are silent on perfectly predicted reward, and are depressed by reward omission. As a task is learned, phasic DA at reward delivery shrinks to near zero. This is the biological mechanism that makes EMA(|advantage|) neurochemically legitimate.
- [Schultz, W. (2016). Dopamine reward prediction error coding. *Dialogues in Clinical Neuroscience*, 18(1).](https://www.tandfonline.com/doi/full/10.31887/DCNS.2016.18.1/wschultz)
- [Niv, Y. et al. (2009). Understanding dopamine and reinforcement learning: the dopamine reward prediction error hypothesis. *PNAS*, 106(S3).](https://www.pnas.org/doi/10.1073/pnas.1014269108)

**On tonic DA shifting the goal-directed/habitual balance:**
- [The neurochemical substrates of habitual and goal-directed control. *Translational Psychiatry* (2020).](https://www.nature.com/articles/s41398-020-0762-5) — Review establishing that higher tonic DA → more goal-directed; lower tonic DA → more habitual. Supports the idea that as RPE shrinks and tonic DA drops with learning, the system naturally shifts to habitual control without a hard schedule.
- [Tonic dopamine, uncertainty and basal ganglia action selection. *Neuroscience* (2021).](https://www.sciencedirect.com/science/article/abs/pii/S0306452221002505)
- [What role does striatal dopamine play in goal-directed action? *Neuroscience* (2024).](https://www.sciencedirect.com/science/article/pii/S0306452224001337)
- [Tonic dopamine and biases in value learning. *Nature Communications* (2025).](https://www.nature.com/articles/s41467-025-62280-1)

**On DA ramping toward goals (the within-trial phenomenon — different from what we need):**
- Howe, M.W. et al. (2013). Prolonged dopamine signalling in striatum signals proximity and value of distant rewards. *Nature*, 500, 575–579. — Striatal DA ramps during goal approach within a single episode. This is a different circuit and timescale from the tonic training-phase signal. Not what the EMA approach models.
- [Dopamine ramps as a normative consequence of dual-process control. *eLife* (2024).](https://elifesciences.org/reviewed-preprints/111458) — Argues that ramps emerge from dual-process control dynamics; distinguishes within-trial phasic ramps from across-trial tonic modulation.

**On midbrain DA and goal-directed navigation:**
- [Midbrain dopamine neurons signal phasic and ramping reward prediction error during goal-directed navigation. *Cell Reports* (2022).](https://www.cell.com/cell-reports/fulltext/S2211-1247(22)01320-1)

**On DA across habit formation:**
- [Striatal dopamine signals are region specific and temporally stable across action-sequence habit formation. *Current Biology* (2022).](https://www.cell.com/current-biology/fulltext/S0960-9822(21)01703-6) — DA in DMS (goal-directed region) and DLS (habitual region) behave differently as habits form; DMS DA tracks RPE, DLS DA is more stable. Supports the idea that the handoff is indexed by the DMS RPE signal dropping.

---

### Proposed DA signal: z-score of prediction error with asymmetric rise/fall

**Your opinion:** Replace the penalty ramp with a signal based on whether the current prediction error is an outlier relative to the standard deviation of recent prediction errors. DA should rise INSTANTLY when the current error is unexpectedly large (z > 0), and DROP SLOWLY as familiarity is re-established. The asymmetry is intentional.

**Claude's take:**

This is the right formulation and has direct grounding in two bodies of literature. **Pearce & Hall (1980)** set attentional associability on trial t proportional to the absolute prediction error on trial t-1 — more surprised than expected → attend more next trial. Your version sharpens this: instead of "was there a prediction error," it asks "was this prediction error abnormal relative to my recent distribution." That maps directly to **Yu & Dayan's (2005) unexpected uncertainty** — the signal that should trigger resetting the controlled system, distinct from the expected variability the habitual system has already absorbed. Their account attributes this to norepinephrine/acetylcholine resetting; here you apply the same logic to the DA gate.

Crucially, the asymmetric rise/fall is also biologically grounded: phasic DA bursts from VTA are fast (sub-second in vivo, within one batch iteration in the model), while DA clearance via reuptake transporters (DAT) is slow — on the order of seconds to minutes. The slow fall in the model corresponds to the reuptake timescale, ensuring the system stays alert for a while after a disruption even if the next batch looks normal.

**Clarification on iterations:** one iteration = one call to `_train_batch` = 128 parallel episodes. A step = one forward pass (one action). "React immediately" = within the current iteration — one training update, no lag.

**Implementation:**

Each iteration, compute `e = mean(|adv|)` for the current batch. Maintain a running mean and variance over recent iterations:

```python
mu  = 0.99 * mu  + 0.01 * e
var = 0.99 * var + 0.01 * (e - mu)**2
std = sqrt(var + 1e-6)

z = (e - mu) / (std + 1e-6)   # positive = more surprised than usual

# Asymmetric update:
if z > 0:
    da_signal = sigmoid(z)         # instant rise to current surprise level
else:
    da_signal = 0.999 * da_signal  # slow exponential fall (~1000 iterations to zero)

w_gd = sigmoid(alpha * da_signal + bias)
```

- **DA rises instantly** on the current batch if `z > 0` — no lag, one update.
- **DA falls slowly** when errors are normal or below normal — decays by 0.999 per iteration, taking ~1000 iterations (~128k episodes) to approach zero. This models DAT-mediated reuptake: the system stays alert well after the crisis appears resolved.

**What this gives:**
- **Early training:** errors high but consistent → z oscillates near 0 → da_signal moderate; add 100-iteration warmup with `w_gd` pinned to 1.0 while stats accumulate
- **Task mastered, habit driving:** errors low and stable → z near 0 → da_signal decays slowly to near-zero → `w_gd` low
- **Sudden disruption:** one batch with abnormally high error → z spikes → da_signal jumps immediately → `w_gd` rises → **reactivation**
- **Crisis resolved:** errors return to normal → z ≤ 0 → slow decay → `w_gd` falls gradually → **requieting**

**Remaining tradeoffs:**
1. *Feedback coupling* — closed loop; needs empirical convergence check
2. *Value convergence ≠ habitual competence* — add floor condition: `w_gd` cannot drop until `hab_solo_acc > criterion`
3. *Global scalar* — per-batch, not per-timestep; biologically coarser but simpler

**Wave-like dynamics and asymmetric kinetics — literature grounding (Opus search):**

The fast-rise/slow-fall asymmetry is biologically real and has three independent groundings.

**1. DA release/reuptake kinetics (FSCV literature).**
Extracellular DA rises to peak in ~200 ms after a phasic burst, then decays over 2–10 seconds depending on region and DAT density — two regimes: fast diffusive clearance immediately post-release, then slow DAT-mediated enzymatic reuptake (Rice & Cragg 2008; Cragg & Rice 2004; reviewed in *Unraveling the dynamics of dopamine release*, PMC10204099). The slow decay in the model (0.999 per iteration) directly maps to the DAT-mediated reuptake timescale. These papers are not yet in the vault — add with `/add-paper`.

**2. Pearce & Hall (1980).**
Attentional associability on trial t is gated by the absolute prediction error on trial t-1: more surprised → attend more, higher effective learning rate next trial. This is the strongest single citation for "surprise gates a slow-decaying gain signal." Neural correlates shown by Roesch & Schoenbaum (PMC3325511). Not yet in vault.

**3. Yu & Dayan (2005), *Uncertainty, Neuromodulation, and Attention* (Neuron).**
Distinguishes *expected uncertainty* (ACh) from *unexpected uncertainty* (NE). Unexpected uncertainty = current PE exceeds expected variance — conceptually identical to a positive z-score of |PE|, and triggers a reset of the controlled system's priors. **Critical caveat:** they assign this reset signal to norepinephrine (locus coeruleus), not dopamine. Your design maps their computation onto the DA gate. State explicitly in the mémoire: "we operationalize unexpected uncertainty via the DA gate, following the functional logic of Yu & Dayan (2005) but assigning the computation to DA given its established role in arbitrating goal-directed vs habitual control." Not yet in vault.

**On the "wave-like" DA dynamics papers specifically (both in vault):**

- **`[[hamid_wave-like_2021]]`** — Hamid, Frank & Moore (2021, *Neuron*). These are **spatial** DA waves propagating DMS↔DLS (~0.1–1 mm/s), not temporal rise/fall dynamics. However: the DMS anticipatory ramp is a direct biological analog of your **DA-request neuron**. In Hamid, the DMS expert ramps during reward pursuit when it is "responsible" for predicting reward-progress, and that ramp shrinks (goes negative) when instrumental control is low (Pavlovian condition). This maps precisely onto the DA-request neuron's expected behaviour across training: ramps up during goal-directed engagement, decays as the habitual system takes over. Cite Hamid for the DA-request neuron's ramp profile and credit-assignment role. Do not cite it for the fast-rise/slow-fall asymmetry — those are distinct phenomena.
- **`[[lloyd_tamping_2015]]`** — Lloyd & Dayan (2015, *PLOS Computational Biology*). Establishes phasic burst-firing (~200 ms transients) and quasi-tonic firing (slow baseline, governed by average reward rate) as **independent channels** — not just different speeds of the same signal. The quasi-tonic vigor signal `(1−γ)⟨V(s')⟩` scales slowly; the phasic RPE burst is immediate. This is the closest existing model to a fast-rise/slow-fall split. Directly citable for the asymmetric dynamics design.
- **`[[naude_dopamine_2024]]`** — Two separable DA timescales: fast NMDA-excitability (within-trial decision widening) vs slow plasticity (across-session maintenance deepening). General support for two-timescale DA; not specific to the rise/fall asymmetry.

**z-score of prediction error as gate — novelty assessment:**
No canonical paper proposes the exact z-scored |TD-error| form for gating w_gd. The design is novel. Frame it as "a computational operationalization of Pearce-Hall associability under Yu-Dayan unexpected uncertainty, applied to the DA gate" — grounded and novel, not a re-statement of a prior model.

**Papers still to add to vault:** Rice & Cragg (2008), Cragg & Rice (2004), Pearce & Hall (1980), Yu & Dayan (2005). Run `/add-paper` before citing in the mémoire.

---

## F12 — Early stopping and hab_onset criterion thresholds

**Problem:** Two threshold values need to be consistent and well-motivated: (1) the early-stopping criterion (when does training end early), and (2) the hab_onset detection criterion (when is the habitual system considered competent). Currently early stopping is at 99% rolling accuracy over the last 20k episodes (10% of max 200k); hab_onset is detected when `hab_solo_acc >= 0.8` (80%). Using 80% for hab_onset while early stopping at 99% creates an inconsistency: the experiment can end while the habitual system is only at 80% competence, making hab_onset a weak criterion relative to the stopping condition.

**Your opinion:** Early stopping at 99% accuracy over the last 10% of runs (already implemented). Use 99% instead of 100% for the 80% mark — hab_onset criterion should be 99%, not 80%.

**Claude's take:** Early stopping is already correct: `early_stop_acc = 0.99` and `rolling_window = 20_000` (10% of 200k) are in `config.py:74` and applied in `train.py:339–345`. No change needed there. For hab_onset: changing the threshold from `hab >= 0.8` to `hab >= 0.99` in `analysis.py:152` makes it consistent with the early-stopping criterion and meaningfully stricter — hab_onset now marks the point where the habitual system is genuinely competent, not merely above chance. This also raises the bar for H2 (handoff): `w_gd` must drop *after* hab reaches 99%, not just 80%. The H2 combined_acc pass threshold (`comb[-1] >= 0.8`, line 162) should probably stay at 80% — that's a different question (does the combined system maintain performance) and 99% there would be overly strict given combined-system noise. One-line fix: `analysis.py:152` `hab >= 0.8` → `hab >= 0.99`.

---

## All papers mentioned in this session — to add to vault

Papers already in the vault are marked **[IN VAULT]**. All others need `/add-paper` before being cited in the mémoire.

---

### Group 1: DA as reward prediction error — the gate mechanism's biological ground

**Schultz, W., Dayan, P., & Montague, P.R. (1997). A neural substrate of prediction and reward. *Science*, 275, 1593–1599.**
*Why added:* The founding paper of the RPE hypothesis. DA neurons fire on unpredicted reward, are silent on fully predicted reward, depressed on omission. This is the formal identity between |advantage| and DA signal. The whole F5 argument that the penalty ramp should be replaced by something RPE-derived stands or falls on this paper. *Why interesting:* It also means that as a task is overtrained, phasic DA at reward shrinks to near zero — which is the emergent mechanism that should close the gate without any schedule.

**Schultz, W. (2016). Dopamine reward prediction error coding. *Dialogues in Clinical Neuroscience*, 18(1).** [URL: https://www.tandfonline.com/doi/full/10.31887/DCNS.2016.18.1/wschultz]
*Why added:* Updated review from Schultz himself on 20 years of RPE evidence. Covers both phasic and tonic DA signalling in one place. Useful for the mémoire because it's a single citable source that handles the full RPE story.

**Niv, Y., Duff, M.O., & Dayan, P. (2005). Dopamine, uncertainty and TD learning. *Behavioural and Brain Functions*, 1(6).** (also: Niv, Y. (2009). Reinforcement learning in the brain. *Journal of Mathematical Psychology*, 53.)
[URL: https://www.pnas.org/doi/10.1073/pnas.1014269108]
*Why added:* Connects temporal difference (TD) learning to dopamine in a way that makes the |advantage| → DA mapping precise. The advantage function in A2C IS the TD error — this paper closes the loop between the code and the biology.

---

### Group 2: Tonic DA and the goal-directed/habitual balance

**The neurochemical substrates of habitual and goal-directed control. *Translational Psychiatry* (2020).** [URL: https://www.nature.com/articles/s41398-020-0762-5]
*Why added:* Review paper establishing that higher tonic DA → more goal-directed behaviour; lower tonic DA → more habitual. This is the direct biological grounding for the claim that as DA drops with mastery (RPE → 0), the system naturally shifts toward habitual control. Directly relevant to F5.

**Tonic dopamine, uncertainty and basal ganglia action selection. *Neuroscience* (2021).** [URL: https://www.sciencedirect.com/science/article/abs/pii/S0306452221002505]
*Why added:* Extends the tonic DA / action selection link with uncertainty framing — higher tonic DA increases tolerance for uncertainty, which maps onto goal-directed exploration. As DA drops, the system prefers the certain habitual route. Relevant to both F5 and the Discussion.

**What role does striatal dopamine play in goal-directed action? *Neuroscience* (2024).** [URL: https://www.sciencedirect.com/science/article/pii/S0306452224001337]
*Why added:* Most recent review of this specific question. Distinguishes DMS (dorsomedial striatum, goal-directed) from DLS (dorsolateral striatum, habitual) DA functions — the anatomical mapping that underlies the two-system architecture.

**Tonic dopamine and biases in value learning. *Nature Communications* (2025).** [URL: https://www.nature.com/articles/s41467-025-62280-1]
*Why added:* Very recent. Shows that tonic DA creates asymmetric biases in value updating — high tonic DA makes positive prediction errors more impactful, low tonic DA makes negative prediction errors more impactful. Interesting because it suggests the DA gate is not just on/off but shapes the *direction* of learning, not only its rate.

---

### Group 3: DA kinetics — the biological substrate for fast-rise/slow-fall

**Rice, M.E. & Cragg, S.J. (2008). Dopamine spillover after quantal release: rethinking dopamine transmission in the nigrostriatal pathway. *Brain Research Reviews*, 58(2), 303–313.**
*Why added:* Establishes two-regime DA clearance: fast diffusive clearance immediately post-release, then slower DAT-mediated enzymatic reuptake. This is the direct biological substrate for the asymmetric rise/fall in the proposed DA signal. *Why interesting:* The fast diffusion phase means DA "smears" locally for ~200 ms; the slow DAT phase determines whether the signal persists for seconds. DAT density varies across striatum, which means reuptake speed is spatially heterogeneous — consistent with Hamid's wave dynamics.

**Cragg, S.J. & Rice, M.E. (2004). DAncing past the DAT at a DA synapse. *Trends in Neurosciences*, 27(5), 270–277.**
*Why added:* Companion paper to Rice & Cragg (2008), focuses specifically on the role of the dopamine transporter (DAT) in shaping extracellular DA dynamics. DAT density, not release magnitude, determines how long a phasic burst persists. This grounds the slow-fall constant (0.999/iteration) in a specific molecular mechanism rather than being an arbitrary smoothing parameter.

---

### Group 4: Surprise-gating — the z-score signal's conceptual ground

**Pearce, J.M. & Hall, G. (1980). A model for Pavlovian learning: variations in the effectiveness of conditioned but not unconditioned stimuli. *Psychological Review*, 87(6), 532–552.**
*Why added:* The founding paper for surprise-gated attentional associability. On trial t, the learning rate (associability) is proportional to the absolute prediction error on trial t-1: more surprised → attend more → learn faster next time. This is exactly the logic of the z-score DA gate — surprise about the prediction error magnitude modulates a control signal. *Why interesting:* Unlike Rescorla-Wagner (which assumes fixed learning rate), Pearce-Hall captures why familiar stimuli stop commanding attention: their associability decays as prediction error shrinks. Directly maps onto the "slow fall as familiarity is established" part of the proposed signal.

**Yu, A.J. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron*, 46(4), 681–692.**
*Why added:* Distinguishes *expected uncertainty* (variability you've already modelled — ACh signal) from *unexpected uncertainty* (deviation from expected variability, current PE exceeds the distribution — NE signal). Unexpected uncertainty triggers a prior reset. Your z-score signal is a direct operationalization of this: z > 0 means the current PE is outside the expected distribution, which should trigger re-engagement of the controlled (GD) system. *Why interesting to us specifically:* Yu & Dayan assign this computation to norepinephrine (LC-NE), not DA. The design decision to route it through the DA gate instead is a modelling assumption that needs to be stated explicitly in the mémoire — motivated by DA's known role in arbitrating goal-directed vs habitual control, which NE doesn't have in the same way.

**Roesch, M.R., Calu, D.J., Esber, G.R., & Schoenbaum, G. (2010). Neural correlates of variations in event processing during learning in basal ganglia. *Journal of Neuroscience* (PMC3325511).**
*Why added:* Provides neural evidence for Pearce-Hall associability in basal ganglia circuits. Shows that BG neurons track prediction error magnitude in a way consistent with Pearce-Hall updating rather than Rescorla-Wagner. Bridges the psychological Pearce-Hall model to the neural circuit level — directly relevant to grounding the z-score gate in a known neural substrate.

---

### Group 5: DA spatial and temporal wave dynamics

**[IN VAULT] `[[hamid_wave-like_2021]]` — Hamid, A.A., Frank, M.J. & Moore, C.I. (2021). Wave-like dopamine dynamics as a mechanism for spatiotemporal credit assignment. *Neuron*, 109(24), 4073–4091.**
*Why interesting to us:* Two uses. (1) The DMS anticipatory ramp is a biological analog of the DA-request neuron: it builds up during reward pursuit when the DMS "expert" is responsible, and shrinks when instrumental control drops. (2) Wave direction reverses within a few trials of contingency change — fast local transients signal surprise at transitions. Do not cite for temporal rise/fall asymmetry; cite for the DA-request ramp profile.

**[IN VAULT] `[[lloyd_tamping_2015]]` — Lloyd, K. & Dayan, P. (2015). Tamping ramping: Algorithmic, implementational, and computational explanations of phasic dopamine signals in the accumbens. *PLOS Computational Biology*, 11(12).**
*Why interesting to us:* Establishes phasic and quasi-tonic DA as **independent channels**, not just different speeds of the same signal. The quasi-tonic vigour signal scales with average reward rate (slow); the phasic RPE burst is immediate. This is the strongest existing computational grounding for the fast-rise/slow-fall asymmetry being a principled design rather than a smoothing trick.

**Howe, M.W., Tierney, P.L., Sandberg, S.G., Phillips, P.E.M., & Bhatt, D.L. (2013). Prolonged dopamine signalling in striatum signals proximity and value of distant rewards. *Nature*, 500, 575–579.**
*Why added:* DA ramps during goal approach within a single episode (nucleus accumbens, fast-scan cyclic voltammetry in rats). Important to cite to *distinguish* this from the training-timescale signal: within-trial DA ramps are a different phenomenon from the across-episode DA drop that drives the handoff. If you don't distinguish them, a reviewer might conflate the two.

**Dopamine ramps as a normative consequence of dual-process control. *eLife* (2024).** [URL: https://elifesciences.org/reviewed-preprints/111458]
*Why added:* Argues that within-trial DA ramps emerge from the interaction of two control processes, not from a single mechanism. Useful for the Discussion when distinguishing what this model's DA signal is and isn't doing relative to the ramp literature.

---

### Group 6: DA modulation of plasticity (F11)

**[IN VAULT] `[[naude_dopamine_2024]]` — Naudé, J. et al. (2024). Dopamine builds and reveals reward-associated latent behavioural attractors. *Nature Communications*.**
*Why interesting to us:* DA modulates NMDA-receptor conductance (fast, within-trial) and synaptic plasticity (slow, across-session) as separable processes. The fast component is the "widen" (decision) effect implemented in E6/E7/E9. The slow plasticity component is what F11 proposes to implement: DA-gated gradient scaling. Naudé provides the biological justification for treating plasticity modulation as a legitimate and separable DA function.

---

### Group 7: Goal-directed vs habitual — the broader framework

**Striatal dopamine signals are region specific and temporally stable across action-sequence habit formation. *Current Biology* (2022).** [URL: https://www.cell.com/current-biology/fulltext/S0960-9822(21)01703-6]
*Why added:* Directly measures DA in DMS and DLS during the transition from goal-directed to habitual behaviour. DMS DA tracks RPE and drops as habits form; DLS DA is more stable. This is the in-vivo correlate of what the model is trying to reproduce. If the model is right, the DMS-like (GD) DA signal should drop across training exactly as this paper shows.

**Midbrain dopamine neurons signal phasic and ramping reward prediction error during goal-directed navigation. *Cell Reports* (2022).** [URL: https://www.cell.com/cell-reports/fulltext/S2211-1247(22)01320-1]
*Why added:* Shows that midbrain DA neurons encode BOTH the classic phasic RPE (at reward) AND a slower ramp during approach in the same task. Relevant to the multi-timescale discussion: both components come from the same biological neurons, supporting the idea that a single DA signal can have asymmetric temporal properties.

---

## F13 — Design comparison: vectorised parallelism and input weight scale vs supervisor's code

**Context:** Comparison between our architecture and the supervisor's `tunl_a2c_two_area.py`, relevant to the mémoire Method section and to interpreting any differences in training dynamics.

---

### Parallelism disadvantages

Our `batch_size=128` vectorised training is ~128× faster per wall-clock second than the supervisor's sequential loop (1,600,000 steps × up to 1000 steps/episode, no vectorisation). But it has four structural costs:

1. **Less-online gradient updates.** All 128 environments in a batch use the same frozen weights θ_t — then one update. In sequential training, each episode immediately updates weights before the next episode runs. The policy can only improve as fast as updates happen, not as fast as experience is collected.

2. **All 128 lanes share the same `current_delay`.** The supervisor can advance the curriculum after any individual episode. We can only advance once per batch. Combined with the rolling window of 100 episodes — which is less than one full batch iteration — the delay can advance very aggressively if a single batch goes well. This is likely a contributing cause of the curriculum instability seen in some runs.

3. **Averaged signals wash out per-trial effects.** `evaluate_vec` returns the mean `da_request` across all B lanes. A genuine but noisy reactivation signal on a subset of trials gets averaged with lanes where it didn't fire. This reduces the signal-to-noise on H5 and makes marginal results harder to interpret.

4. **Supervisor's demo replay buffer is architecturally incompatible.** She maintains a `deque` of the best 500 PFC episodes and replays them for DLS at 30% of updates. Mixing live-batch vectorised data with selectively replayed episodes would break the batched forward pass structure. This replay mechanism is one reason her DLS imitation learns more stably than a naive CE loss would.

---

### Input weight scale difference

Both networks use `input_dim=6`. At initialisation, the pre-activation contributions from input vs recurrent are:

| | Input variance (6 × σ²) | Recurrent variance (n × σ²_rec) | Who dominates |
|---|---|---|---|
| Supervisor PFC | 6 × 1 = **6** | 512 × 1/512 ≈ **1** | Input by 6× |
| Supervisor DLS | 6 × 0.004 ≈ **0.024** (Xavier) | — (low-rank) | Roughly balanced |
| Our GDNet | 6 × 0.01 = **0.06** | n × 0.81/n = **0.81** | Recurrent by ~13× |

Her PFC is **input-driven** at initialisation: observations have strong immediate influence over hidden state dynamics. Our GDNet is **memory-dominated**: the recurrent dynamics have ~13× more influence on `h` than the incoming observation does. This is likely better for a working-memory task — the hidden state has more inertia and resists disruption during the delay period — but it means early learning is slower, because the gradient through the input pathway is small and observations take longer to build structure in the recurrent dynamics.

Her DLS uses Xavier (`std ≈ 0.062` for 6→512), which lands close to our 0.1 input scale. The large-input-weight design is specific to her PFC, not a general pattern across both her networks.

**Implication for the mémoire:** the slower early learning in our GDNet (relative to what a larger `W_in` init would give) is not a bug — it is consistent with the design goal of a network that maintains working memory rather than being input-reactive. It does mean that `hab_onset` timing comparisons with the supervisor's results are not apples-to-apples, independent of the F6 seed-count issue.
