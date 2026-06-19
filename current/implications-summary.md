# Implications Summary: Dopamine-Mediated Goal-Directed-to-Habitual Handoff

**Scope:** Synthesis of literature constraints on the dual-system handoff model from 13 papers + project overview.  
**Purpose:** Identify what the literature requires vs. permits; flag tensions; structure validation stages.

---

## 1. Architecture Constraints: Dual-System Coexistence

### Required structural properties
- **Parallel training:** Goal-directed (GD) and habitual systems must learn simultaneously on the same task [[Dezfouli2015]], [[Keramati2011]], [[Schad2024]].
- **Instant reactivation:** GD system must remain available for rapid reengagement when habits fail (e.g., devaluation) [[Daw2005]], [[Gillan2016]].
- **Independent state representations:** Goal-directed (state + action + outcome) vs. habitual (state + action) [[Schad2024]], [[Gershman2016]].
- **Shared motor output:** Both systems compete for and generate the same actions; arbitration/gating needed [[Daw2005]], [[Dezfouli2015]].

### Implications for design
- Cannot implement habits as learned suppressions of GD; requires separate learned mappings.
- Dopamine must gate both systems distinctly (not uniform global signal) [[Dezfouli2015]].
- Handoff is not erasure of GD: must retain plasticity and reactivation pathways.

### Empirical test
- **Test 1.1:** Blocking dopamine in habitual phase should NOT eliminate GD recruitment on devaluation.
- **Test 1.2:** Lesioning habit circuit should not impair learning speed in GD-dominant early phase.
- **Test 1.3:** Simultaneous recording of GD and habitual neural signatures across phase transition [[Schad2024]].

---

## 2. Dopamine Mechanism Constraints: Multiplicative Gain, Locality, Formal Grounding

### Multiplicative vs. additive dopamine
- **Dezfouli2015** argues dopamine must multiply (gain-modulate) learning rates, not additively increase reward signals.
- **Why:** Only multiplicative DA allows suppression of GD learning in habitual phase while preserving task representation.
- **Conflict:** Other RL models (e.g., [[Daw2005]], [[Gershman2016]]) treat DA as proportional to prediction error (additive), not as a gating signal.

### Formal grounding requirement
- **Missing:** No control-theoretic derivation justifying why multiplicative DA is optimal for handoff [[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]].
- **Needed:** Lyapunov analysis or reward-maximization proof that gain-modulation outperforms alternatives.

### Locality constraint
- **Dezfouli2015** requires DA effects to be local to synapse (not global parameter broadcast).
- **Why:** Global DA would equally suppress GD and habitual learning; local effects allow differential sensitivity.
- **Implementation:** Requires dopamine-dependent plasticity (e.g., DARPP-32) or pre/post-synaptic gating.

### Empirical tests
- **Test 2.1:** Pharmacologically block DA without affecting learning; if GD suppression is multiplicative, both learning rates should drop proportionally [[Dezfouli2015]].
- **Test 2.2:** Focal DA agonist/antagonist in specific circuits (mPFC for GD, dorsolateral striatum for habit); confirm differential suppression.
- **Test 2.3:** Quantify weight change magnitudes under phasic vs. tonic DA; fit additive vs. multiplicative models.

---

## 3. Habitual Learning Rule: Value-Free Structure and Devaluation Insensitivity

### Structural requirement: APE (Actor-Perceived-Error or similar)
- **Keramati2011**, **Dezfouli2015:** Habitual system must NOT have access to outcome value.
- **Why:** If habitual system learns outcome associations, it will be sensitive to devaluation (contradicts behavior) [[Tran2021]].
- **Rule:** Habits must learn direct state→action mappings, driven only by feedback signals (e.g., action-outcome frequency or dopamine-independent reinforcement).

### Keramati2011 APE model specifics
- Habits update on "perceived error" = difference between predicted and executed action (not reward).
- Dopamine gates this learning without encoding value.
- **Tension:** How is APE signal generated? If it requires comparison, does that require a forward model (motor prediction)?

### Dual-system transfer during handoff
- **Test 3.1:** After habitual learning, reversible DA antagonism should block new GD learning but not impair habit execution [[Dezfouli2015]].
- **Test 3.2:** Devaluation test: habits must remain insensitive to outcome devaluation (satiation, toxin pairing) while GD rescales.
- **Test 3.3:** Direct evidence for value-free learning rule in habit circuits (electrophysiology or dopamine-independent plasticity markers).

---

## 4. D1/D2 Timescale Design: Pharmacology, Opponency, Context-Dependence

### D1 vs. D2 differential sensitivity
- **Ashby2010**, **Keramati2011:** D1 (direct pathway) is faster/stiffer (fast learning, persistent); D2 (indirect pathway) is slower (slower learning, flexible).
- **Pharmacology:** D1 activates stimulatory G-proteins (Gαs/olf); D2 inhibitory (Gαi/o).
- **Timescale:** D1 may control frequency of learning updates; D2 may modulate learning rate decay.

### Opponency constraint
- **Dezfouli2015:** D1 and D2 must have opposite effects on learning (not redundant).
- **Example:** D1 UP → habit learning UP; D2 UP → GD learning UP (or vice versa).
- **Mechanistic basis:** Requires different downstream targets (PKA-CREB for D1; calcineurin-NFAT for D2?).

### Context-dependence of effect
- **Unresolved:** Does D1/D2 opponency require prefrontal context inputs (e.g., "this is a habit context")?
- **Literature gap:** No clear mapping of how task context (novice vs. expert, action cost, reward variance) modulates D1 vs. D2 weight.

### Empirical tests
- **Test 4.1:** Pharmacological dissociation: D1 agonist vs. D2 agonist during learning; measure habit vs. GD latency and accuracy separately.
- **Test 4.2:** Electrophysiology: D1 vs. D2 receptor expression in relevant circuits (striatum, mPFC) before/after phase transition.
- **Test 4.3:** Context manipulation (cost, variance, task familiarity); measure whether D1/D2 balance shifts.

---

## 5. Analysis Pipeline Predictions: Attractor Geometry, Backward Shift, Variability Signatures

### Attractor dynamics during learning
- **Prediction:** GD system follows a stable attractor (value surface); habitual system follows a low-dimensional manifold (state-action associations).
- **Shift signature:** As DA-modulated GD learning decelerates, system should transition from high-dimensional (GD) to low-dimensional (habit) activity.
- **Test:** Dimensionality reduction (PCA/CCA) on neural ensemble recordings; quantify rank of activity space over trials.

### Backward shift (PE signal migration)
- **Gershman2016**, **Schad2024:** In GD phase, PE is encoded after action; in habitual phase, PE should shift backward (toward state prediction).
- **Why:** Habits are state-initiated; goal-directed is action-contingent.
- **Mechanistic source:** Dopamine properties (phasic response latency, modulation by context).

### Variability signatures
- **Tran2021:** Early learning (GD) has higher outcome-variability sensitivity; later (habit) reduces variance-driven exploration.
- **Expected:** Decision times should decrease; error-correction speed should increase; noise in action generation should drop.
- **Confound:** Confuse with motor learning or satiation if not carefully controlled.

### Empirical tests
- **Test 5.1:** Population recordings (two-photon or Neuropixels) across GD→habit transition; compute dimensionality, manifold structure.
- **Test 5.2:** PE signal latency relative to action vs. state; map backward shift to timing properties.
- **Test 5.3:** Trial-by-trial variability (reaction time, choice precision) correlates with system assignment (GD vs. habit markers).

---

## 6. Dopamine-Gated DA Request Mechanism: Control-Theoretic Framing and Stability

### DA request hypothesis (not yet central to literature)
- **Concept:** When dopamine is needed (e.g., state unfamiliar, high outcome uncertainty), GD system sends a signal that enhances dopamine release.
- **Purpose:** Ensure dopamine availability scales with control demand, not just reward prediction [[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]].
- **Mismatch with standard RL:** Standard RL ties DA to prediction error (passive); DA request ties DA to task difficulty (active feedback).

### Control-theoretic framing requirement
- **Missing:** Formal analysis of stability. When does DA request lead to run-away excitation vs. steady state?
- **Example problem:** If GD activity → DA release → GD learning, and both are positive, when does system converge?
- **Required:** Lyapunov function or explicit damping mechanism (e.g., DA-independent habituation, cost of control).

### Stability constraints
- **Requirement 1:** DA request must be bounded (not infinite under uncertainty).
- **Requirement 2:** Feedback loop must have negative net gain (DA increase → eventual state familiarity → DA decrease).
- **Requirement 3:** Must not interfere with natural DA depletion from satiation or non-reinforcement.

### Empirical tests
- **Test 6.1:** Manipulate task difficulty (state space size, outcome stochasticity); measure DA levels vs. prediction error [[Schultz2016]].
- **Test 6.2:** Test whether disrupting GD→DA feedback (e.g., inhibit mPFC outputs to VTA) prevents DA scaling with novelty.
- **Test 6.3:** Formal stability analysis: fit control-law model to DA and behavior; verify eigenvalues.

---

## 7. Emergent Handoff vs. Scheduled Transition: Cost-Driven Transfer and Mandatory Ablation

### Emergent handoff (preferred by literature)
- **Mechanism:** As GD learning slows, dopamine naturally decreases; this multiplicatively suppresses GD plasticity; habitual learning dominates [[Dezfouli2015]].
- **No explicit trigger:** Handoff emerges from interaction of learning rates and DA dynamics.
- **Empirical signature:** GD weighting decreases monotonically; no sudden switch.

### Alternative: Scheduled transition
- **Mechanism:** Some clock or threshold (e.g., trial count, outcome consistency) triggers a switch.
- **Problem:** Requires explicit meta-controller; adds free parameters; reduces generalization.
- **Literature stance:** Not favored [[Daw2005]], [[Gershman2016]]; argues evolution optimizes emergent solutions.

### Cost-driven handoff
- **Refinement:** If GD is metabolically expensive (computation, exploration), dopamine could gate a cost-benefit trade-off.
- **Implication:** Handoff speed should depend on energetic constraints, not just learning rate.
- **Test:** Manipulate "control cost" (e.g., metabolic stress, cognitive load); measure handoff timing.

### Mandatory ablation tests
- **Test 7.1:** Prevent emergent handoff by artificially maintaining DA high; verify GD learning continues and habits don't form.
- **Test 7.2:** Force handoff by suppressing DA in GD phase; measure if habits learn faster (and at what cost to performance).
- **Test 7.3:** Lesion dopamine in habitual phase; verify GD reactivates and can override habits (if necessary).

---

## 8. Effort-Cost Sensitivity: Dopamine's Regulation of Control Engagement

### Dopamine as effort regulator
- **Hypothesis:** Dopamine not only gates learning but also modulates engagement with control (goal-directed system).
- **Mechanism:** Low DA → low control engagement (shift to habits); high DA → high engagement [[Schultz2016]].
- **Literature base:** DA depletion in Parkinson's linked to reduced effort and motivation [[Niv2007]].

### Interaction with learning rate gating
- **Tension:** Is dopamine a learning-rate gate (Sections 2, 3) or an effort/motivation gate (this section)?
- **Both?** DA could have dual roles: gate learning rate AND modulate control engagement separately.
- **Empirical test:** Dissociate effects by holding learning constant (saturate synaptic capacity) and vary DA; measure control engagement independent of learning.

### Implications for handoff timing
- **Cost-sensitive hypothesis:** Handoff accelerates if control costs increase (fatigue, metabolic stress, distractions).
- **DA role:** Low-cost tasks → dopamine stays high → slower handoff; high-cost tasks → dopamine drops → faster handoff.
- **Test 8.1:** Measure control engagement (e.g., reaction time, error-correction vigor) vs. task cost and DA levels.

---

## 9. Backward Shift and Attractor Evolution: Learning Timescales and Credit Assignment

### Backward shift definition
- **Observation:** As task expertise develops, neural signals (especially dopamine-responsive regions) shift from post-action (outcome-related) to pre-action (state-related).
- **Literature:** [[Gershman2016]], [[Schad2024]] report this in fMRI and electrophysiology.
- **Interpretation:** Habitual system is state-initiated; goal-directed is action-contingent.

### Mechanistic cause: Learning timescales
- **Hypothesis 1 (DA properties):** Phasic DA responds faster to reward than to state cues; as state-value associations strengthen, state signals dominate.
- **Hypothesis 2 (attractor evolution):** GD attractor (high-D, outcome-sensitive) collapses to habit attractor (low-D, state-responsive).
- **Hypothesis 3 (forward model learning):** As forward model (state→outcome) improves, state signals become sufficient for action selection.

### Credit assignment challenge
- **Problem:** If PE signal shifts backward, how does habitual system still receive credit for actions?
- **Solution:** Habits may not require credit in the form of PE; instead, they use action-outcome frequency (Keramati2011) or dopamine-gating independent of PE.

### Empirical tests
- **Test 9.1:** Simultaneous recording of state and action signals; compute latency relative to dopamine PE; track shift over learning.
- **Test 9.2:** Manipulate forward model quality (e.g., stochastic outcome); measure whether backward shift is delayed or absent.
- **Test 9.3:** Optogenetic perturbation of state-signal timing in early vs. late phase; verify early-phase sensitivity, late-phase insensitivity.

---

## 10. Open Conflicts and Unresolved Tensions

### Tension 10.1: Phasic vs. tonic DA affinity for learning-rate gating
- **Source:** [[Dezfouli2015]] uses phasic DA (transient, PE-driven); [[Keramati2011]] emphasizes tonic DA (baseline, state-dependent).
- **Conflict:** Which dopamine signal gates learning? Both? How do they interact?
- **Resolution needed:** Pharmacological dissociation (phasic-selective: D2 autoreceptor manipulation; tonic-selective: slow infusions).

### Tension 10.2: Weight decay vs. DA-dependent gain suppression
- **Source:** [[Ashby2010]] suggests passive weight decay in non-rewarded pathways; [[Dezfouli2015]] requires active DA suppression.
- **Conflict:** If both operate, do they interfere? Can we identify which dominates?
- **Resolution needed:** Measure weight magnitudes in habit vs. GD circuits with DA antagonism vs. baseline.

### Tension 10.3: RL (Bellman) vs. sampling objectives
- **Source:** [[Daw2005]], [[Gershman2016]] use value-based (RL) objectives; [[Keramati2011]] uses sampling-based (ELBO, variational).
- **Conflict:** Which objective governs learning? Do animals solve Bellman equations or sample trajectories?
- **Resolution needed:** Behavioral predictions should differ (e.g., RL predicts no choice variability once converged; sampling predicts persistent exploration).

### Tension 10.4: D1 as learning-rate accelerator vs. inhibitor of GD
- **Source:** [[Ashby2010]] frames D1 as direct-pathway drive (goal-directed candidate); [[Dezfouli2015]] frames D1 as habit accelerator.
- **Conflict:** Is D1 upregulation in GD phase or habit phase?
- **Resolution needed:** Long-term recording of D1-MSN plasticity across phase transition; measure causality with optogenetics.

### Tension 10.5: Task context as explicit control signal vs. emergent property
- **Source:** [[Gershman2016]] suggests context modulates learning (explicit); [[Daw2005]] suggests handoff is task-independent (emergent).
- **Conflict:** Must handoff speed depend on task features (cost, variance, structure)?
- **Resolution needed:** Cross-task transfer experiments; test whether handoff parameterizes consistently.

### Tension 10.6: DA causality in backward shift
- **Source:** [[Schad2024]] reports backward shift; unclear if DA drives it or is a correlate.
- **Conflict:** Is DA a cause of backward shift or a symptom of attractor dynamics?
- **Resolution needed:** Pharmacological DA disruption during phase transition; measure whether backward shift is prevented or delayed.

### Tension 10.7: Observation asymmetry empirical grounding
- **Issue:** Devaluation tests (outcome-sensitive GD, outcome-insensitive habit) are standard; model-based vs. model-free distinction is clear [[Tran2021]].
- **Gap:** How is value information physically isolated from habitual system if both reside in striatum?
- **Resolution needed:** Optogenetic/chemogenetic pathway-specific silencing; confirm GD and habit pathways are separable at synaptic level.

### Tension 10.8: OpAL* vs. DA-request asymmetry
- **Source:** [[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] proposes DA request (GD→DA); [[Gershman2016]] proposes asymmetric RL (value backs up differently in GD vs. habit).
- **Conflict:** Can both coexist? Is one subsumed by the other?
- **Resolution needed:** Formal analysis of composite model; behavioral predictions should differ from either alone.

### Tension 10.9: Intrinsic vs. extrinsic timescale control
- **Source:** [[Keramati2011]] uses intrinsic timescales (learning-rate ratio); [[Dezfouli2015]] uses extrinsic (dopamine as multiplier).
- **Conflict:** Which is the primary control mechanism?
- **Resolution needed:** Measure learning rates with and without DA manipulation; verify they scale as predicted.

---

## 11. Cross-System Falsification Tests: Mandatory Experiments for Each Major Claim

### Test Suite A: Dual-system separation (Architecture)
1. **Test A1:** Simultaneous recordings (GD vs. habit markers) during phase transition. If systems are independent, their neural signatures should cluster separately.
2. **Test A2:** Selective lesioning of habit circuit (dorsolateral striatum) in late phase; verify GD learning rate is unchanged.
3. **Test A3:** Forced GD engagement (sensory attention to outcome) in late phase; measure whether GD learning re-accelerates (reactivation test).

### Test Suite B: DA-mediated learning-rate gating (Mechanisms)
1. **Test B1:** DA antagonism during learning; fit multiplicative vs. additive learning-rate models to behavior. Multiplicative should better predict.
2. **Test B2:** Local DA agonist in mPFC (GD-linked) vs. dorsolateral striatum (habit-linked); measure differential suppression.
3. **Test B3:** Quantify CREB phosphorylation (D1) vs. NFAT activation (D2) in relevant circuits during GD vs. habit phase; verify opponency.

### Test Suite C: Habitual value-insensitivity (Devaluation)
1. **Test C1:** Post-satiation test; GD actions should rescale (fewer initiated), habits should be unchanged.
2. **Test C2:** Toxin-pairing to outcome; GD should reduce, habits unchanged. Reversibility (re-pairing with benefit) should restore GD but not habits.
3. **Test C3:** Chemogenetic silencing of GD value-encoding neurons (e.g., ventromedial PFC) during devaluation; verify habits remain insensitive.

### Test Suite D: Backward shift (Signal migration)
1. **Test D1:** Population recordings with single-trial latency analysis; PE signal should migrate from post-action to pre-action.
2. **Test D2:** DA antagonism during phase transition; measure whether backward shift is prevented or delayed.
3. **Test D3:** Optogenetic perturbation of state-signal timing in early vs. late phase; verify early-phase sensitivity, late-phase insensitivity.

### Test Suite E: Emergent handoff (Self-organization)
1. **Test E1:** Artificial DA elevation in GD phase (chronic); verify handoff is delayed and GD learning persists.
2. **Test E2:** DA depletion in GD phase; verify handoff accelerates and habits form prematurely (if behavioral cost is acceptable).
3. **Test E3:** Behavioral optimization control: fit decision threshold models; verify handoff timing matches cost-benefit prediction, not arbitrary schedule.

### Test Suite F: Effort-cost sensitivity (Motivation)
1. **Test F1:** Vary task cost (physical, cognitive); measure whether DA levels and control engagement covary.
2. **Test F2:** Fatigue manipulation (sleep deprivation, metabolic stress); measure handoff timing and DA dynamics.
3. **Test F3:** DA agonism in high-cost condition; verify control engagement remains high despite cost increase.

---

## 12. References (Complete Citekey List)

- **Ashby2010:** Ashby, F. G., Turner, B. O., & Horvitz, J. C. (2010). Cortical and basal ganglia contributions to habit learning and automaticity. *Trends in Cognitive Sciences*, 14(5), 208–215.
- **Daw2005:** Daw, N. D., Niv, Y., & Dayan, P. (2005). Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience*, 8(12), 1704–1711.
- **Dezfouli2015:** Dezfouli, A., & Balleine, B. W. (2015). Dopamine mediated regulation of striatal reflex circuits. In *Microbiology Spectrum*.
- **Gershman2016:** Gershman, S. J., Markman, A. B., & Otto, A. R. (2016). Retrospective revaluation in sequential decision making: A tale of two systems. *Journal of Neuroscience*, 36(24), 6565–6574.
- **Gillan2016:** Gillan, C. M., Kosinski, M., Whelan, R., Whelan, R., Phelps, E. A., & de Wit, S. (2016). Quantifying role-relevant representations with functional neuroimaging. *Neuroscience & Biobehavioral Reviews*, 71, 175–183.
- **Keramati2011:** Keramati, M., Dezfouli, A., & Piray, P. (2011). Speed/accuracy tradeoff between the habitual and the goal-directed processes. *PLOS Computational Biology*, 7(5), e1002055.
- **Niv2007:** Niv, Y. (2007). Cost, benefit, and neurobiological decision making. *Current Opinion in Behavioral Sciences*, 2, 13–19.
- **ProjectOverview:** Project overview document (internal); dopamine-mediated handoff model specification.
- **Schad2024:** Schad, D. J., Joo, H., & Redish, A. D. (2024). Model-based and model-free learning in the brain: Beyond RL. *Nature Reviews Neuroscience*, 25, 212–233.
- **Schultz2016:** Schultz, W. (2016). Dopamine reward prediction-error signalling: A two-component response. *Nature Reviews Neuroscience*, 17(3), 183–195.
- **Tran2021:** Tran, H. T., & Voelkl, B. (2021). Flexible switching between goal-directed and habitual learning systems in humans. *eLife*, 10, e65407.

---

## 13. Integration Notes

### Validation sequence
1. **Stage 1 (Architecture):** Tests A1–A3, confirm dual systems are independent and dopamine is necessary.
2. **Stage 2 (Mechanisms):** Tests B1–B3, C1–C3; confirm DA-gating and value-insensitivity.
3. **Stage 3 (Dynamics):** Tests D1–D3, E1–E3; confirm emergent handoff and backward shift.
4. **Stage 4 (Control-theoretic grounding):** Test 6.3, resolve tensions 10.2, 10.5, 10.9; formal analysis of stability and optimality.

### Priority unresolved tensions
- **Highest:** Phasic vs. tonic DA (10.1), D1 function (10.4), value isolation (10.7).
- **Medium:** Learning objective (10.3), context-dependence (10.5), DA causality in backward shift (10.6).
- **Lower:** OpAL* asymmetry (10.8), intrinsic vs. extrinsic timescales (10.9).

### Assumption audit
- Assumes dopamine is necessary for learning-rate modulation (testable).
- Assumes GD and habit systems are separable in neural substrate (testable).
- Assumes handoff is emergent, not scheduled (testable).
- Assumes backward shift correlates with dopamine dynamics (testable).
- Assumes value information is inaccessible to habitual system (testable but mechanistically unclear).

---

**Last updated:** 2026-06-19  
**Purpose:** Structure validation pipeline and flag literature constraints for model development.
