---
citekey: findling_neural_2025
type: paper-note
status: read
verify: pass
topics: [working-memory, mPFC, dopamine, CTRNN, RNN, credit-assignment, attractor]
pdf: "My Library/files/257/Findling et al. - 2025 - Neural variability in the medial prefrontal cortex contributes to efficient adaptive behavior.pdf"
full-text: "[[Findling Neural Variability in the Medial Prefrontal Cortex Contributes to Efficient Adaptive Behavior]]"
---

# findling_neural_2025

> Findling et al. (2025) — Neural variability in the mPFC implements adaptive behavior in volatile environments via stochastic fluctuations in belief representations, without explicit volatility inference.

## What the paper does

Human fMRI combined with computational modelling of a 2-armed bandit task (180 trials, volatility reversing at 5/7/10% rate) shows that human adaptive choices follow the optimal principle—discounting past information more in high-volatility episodes—but this efficiency does not rely on explicit volatility inference. Instead, a "Weber-variability model" explains the data: the dACC and pre-SMA encode first-order beliefs about action-outcome contingencies that undergo stochastic fluctuations scaling with belief magnitude (Weber's law), and these fluctuations alone produce near-optimal adaptive behavior without complex computations. At choice time, dmPFC activations correlate with Weber variability; at outcome time, Weber variability (not explicit volatility estimates) better accounts for dmPFC activity. The model decisively outfits alternatives including Rescorla-Wagner RL and higher-order volatility inference (Bayesian model comparison).

## Key claims relevant to this project

- **Neural variability is functional and load-bearing for adaptive control.** It is not noise to be minimized but a computational resource that enables rapid disengagement from obsolete beliefs and exploration in uncertain environments—directly paralleling the project's hypothesis that dopamine-gated variability drives the goal-to-habit transition. [fMRI data + BMC fits]

- **The dmPFC/dACC encodes simple, stationary world models (first-order beliefs) that are corrupted by variability**, not complex volatility-tracking models. This parsimony supports the dopaCTRNN principle of adding complexity only when justified: a single learned working-memory attractor may be sufficient if gain-modulation and state-dependent noise provide behavioral flexibility. [Whole-brain GLM and model comparison]

- **Weber variability scales with the magnitude of internal change** (belief update size), not with environmental statistics directly. This suggests dopamine modulation of noise magnitude proportional to prediction error could implement a similar principle—phasic dopamine (proportional to RPE magnitude) driving exploration when errors are large. [Neural variability model architecture]

- **mPFC is necessary for forming and updating internal representations** during decision-making, consistent with Villet's finding that early mPFC lesion blocks learning. The paper does not address dopamine explicitly, but its focus on mPFC representation dynamics provides a neural-level grounding for why prefrontal dopamine (targeted by the dopaCTRNN's goal-directed system) is critical during learning. [fMRI lesion logic]

## Mechanism / model details

The **Weber-variability model** assumes belief updating in a first-order Bayesian learner is corrupted by Gaussian noise that scales with belief magnitude (proportional to the variance of the quantity being estimated). Formally, updated belief entropy increases with this noise, and entropy-driven cost in decision-making produces exploratory switching behavior. Critically, no explicit inference of environment volatility occurs; the model learns action-outcome contingencies assuming stationarity, but variability corruption makes these beliefs "floppy" in proportion to how much they have recently changed. At choice time (after belief updating is complete), high entropy requires more neural computation to resolve the decision, explaining elevated dmPFC activations. Bayesian model comparison (uniform priors, exact posterior probabilities) shows Weber-variability decisively beats second-order volatility inference, third-order volatility inference, and noisy Rescorla-Wagner RL.

## Implications for our model

**Direct support for multiple design elements:**

1. **Stage 1 (single working-memory CTRNN) validation:** Findling et al. show that a simple, non-volatile-tracking learner (first-order inference) with neural variability suffices for near-optimal adaptive behavior. This supports the dopaCTRNN roadmap's strategy of starting with a baseline single CTRNN trained on ordinary RL (Stage 1) before introducing dual systems—the baseline network can encode the spatial working-memory task efficiently without explicit volatility or dopamine. The paper's analysis pipeline (fixed-point finding, PCA, dimensionality analysis on human and simulated data) directly parallels the dopaCTRNN's planned interpretability methods.

2. **Stage 5 (dopamine-gated emergent handoff) constraint:** The paper's finding that variability enables disengagement from an outdated belief without explicit recomputation of task structure strongly supports the dopaCTRNN's core mechanism: dopamine-modulated gain on intact weights (`W_eff = f(DA) · W`) coupled with state-dependent noise could allow rapid context-switching between goal-directed and habitual modes without weight decay or relearning. If phasic dopamine (high-magnitude, brief transients) drives Weber-type variability in belief updates, and tonic dopamine (baseline level) gates expression, the d1/D2 affinity split (Section 2.2 of project overview) becomes mechanistically grounded: phasic/D1 could drive exploration (via entropy increase), tonic/D2 could set the baseline expression gain.

3. **Habitual system's learned stimulus-response insensitivity:** Although Findling et al. study goal-directed learning exclusively, their result that first-order beliefs with appropriate variability achieve behavioral optimality without value-tracking is consistent with the dopaCTRNN's hypothesis (Stage 4) that a value-free APE rule in the habitual system, combined with restricted observations (no allocentric map), naturally produces devaluation-insensitivity. A habitual system locked into stimulus-response patterns by design and learning rule would show exactly the kind of "floppy belief" irrelevance to changing reward values that Findling et al. attribute to neural noise in the goal-directed system.

4. **Stage 7 (dimensionality characterization):** Findling et al. use participation ratio and effective dimensionality to characterize the stability and complexity of encoded beliefs. This is directly applicable to the dopaCTRNN's Stage 7 plan to measure whether the habitual system naturally collapses into a low-dimensional subspace despite the 128-unit substrate. The paper provides precedent for using neural-variability magnitude and belief entropy as interpretability measures.

## Open questions / caveats

- **Dopamine is absent from Findling et al.'s study.** The paper does not test whether variability is *controlled* by dopamine or merely correlated with task demands. The dopaCTRNN must independently specify how dopamine modulates noise magnitude (or equivalently, uncertainty/entropy in belief updates). The present work constrains the *functional role* of variability but not its neurochemical substrate.

- **Is Weber variability the right model for a two-system handoff?** Findling et al. study a single decision-making system learning under volatility. The dopaCTRNN has a goal-directed system that must *relinquish control* to a habitual system as learning progresses. It is unclear whether Weber variability in the goal-directed subsystem alone (without explicit comparison to habitual competence) produces the observed reactivation fallback in Villet's paradigm. The dopaCTRNN's DA-request mechanism adds an explicit cost-minimization layer absent in Findling et al.'s account.

- **Belief-updating timescale mismatch:** Findling et al. study updates across ~180 trials (seconds to minutes in real time). The dopaCTRNN's task involves delays of ~90 trial-steps within a single episode, and training across thousands of episodes. It is unclear whether Weber variability at the belief-updating (intra-trial) timescale translates to control-system selection at the episode or training timescale that the dopaCTRNN targets.

- **fMRI temporal resolution:** The paper acknowledges that fMRI cannot identify the precise neural origins of variability—only its aggregated impact on dmPFC activations. Confirming that striatal dopamine controls noise in a dual-system architecture would require electrophysiology or optogenetics, not fMRI, and Findling et al. do not provide that evidence. The dopaCTRNN's claim that dopamine gates belief-updating variability in the goal-directed system remains a post-hoc interpretation of Findling et al.'s findings, not a direct test.

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
