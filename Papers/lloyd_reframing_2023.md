---
citekey: lloyd_reframing_2023
type: paper-note
status: read
verify: pass
topics: [dopamine, striatum, credit-assignment, dual-system, APE, working-memory]
pdf: "My Library/files/131/Lloyd and Dayan - 2023 - Reframing dopamine A controlled controller at the limbic-motor interface.pdf"
full-text: "[[Lloyd Reframing Dopamine A Controlled Controller at the Limbic-Motor Interface]]"
---

# lloyd_reframing_2023

> Lloyd & Dayan (2023) — Dopamine release in nucleus accumbens is modulated by cognitive reframing of Pavlovian-instrumental conflicts, recasting punishment-avoidance as approach to safety and reward-avoidance as loss prevention, with model fits to rodent active avoidance and Go/No-Go data.

## What the paper does

Lloyd & Dayan propose that when instrumental actions conflict with Pavlovian responses (active avoidance vs. freezing; reward suppression vs. approach), the brain exerts cognitive control by manipulating dopamine release itself rather than suppressing the Pavlovian influence directly. They model this reframing as an internal control decision that substitutes a counterfactual baseline state, shifting the origin of a valence-action space and changing the sign of the temporal-difference (TD) error. The model is fit to two rodent experiments with nucleus accumbens dopamine measurements: Gentry et al. (mixed-valence active avoidance, where DA should be elevated on successful avoidance) and Syed et al. (Go/No-Go with DA suppression during successful No-Go trials for large reward). The model captures cue-evoked DA transient patterns across both paradigms by assuming control is deployed with a fixed probability on conflict trials, transforming the TD error from negative (punishment predictor) or positive (reward predictor) to opposite-signed transients that align with observed dopamine.

## Key claims relevant to this project

- **Dopamine is a control signal, not merely a value signal.** The model operationalizes dopamine release as an instrument of cognitive control itself—a quantity that can be actively manipulated to resolve Pavlovian-instrumental conflict, rather than a passive readout of learned values. This directly motivates the project's DA-request neuron (Section 2.3) as a learned, task-dependent control action.

- **Reframing operates via counterfactual baseline shifts, not suppression of the competing system.** Lloyd & Dayan show that the same dopamine signal can support opposite behavioral outcomes (approach on avoidance, inhibition on reward) depending on which reference state (s_fail vs. s_succ) the brain instantiates. This supports the project's separation between learning (W) and expression (W_eff), where dopamine modulates expression without erasing learned structure, consistent with Villet's instant fallback finding.

- **Pavlovian influences are channeled through striatal opponency, not removed.** The mechanism does not eliminate Pavlovian responses but redirects them via multiplicative interaction with expected value (state Pavlovian factor × Q-values). This aligns with the project's use of Go/No-Go-style opponency framing and justifies treating DA as a gain on intact goal-directed and habitual pathways rather than a switch between them.

- **Dopamine transients follow cue onset and reflect cognitive control deployment timing, not outcomes.** The model shows cue-evoked DA can differ from outcome-related DA and captures the temporal dynamics of DA during discriminative cues. This is load-bearing for interpreting the project's DA-request output as a learned, trial-type-dependent control signal rather than a monolithic reinforcement signal.

- **Control is expensive and its deployment reflects task structure and conflict.** The model assumes control is applied probabilistically on conflict trials (shock trials in avoidance; No-Go large-reward trials in Go/No-Go) but not on others. This operationalizes the project's cost-minimization principle for DA requests—requesting DA costs something, and the goal-directed system should minimize unnecessary requests as the habitual system becomes competent.

## Mechanism / model details

Lloyd & Dayan's core operation is a **counterfactual baseline shift**:

- **No-control case**: TD error δ = V(s_χ) - V(s_pre), where s_pre is the pre-trial state (~neutral baseline).
- **Control case**: Cognitive control instantiates a counterfactual state s_fail or s_succ (depending on task context) such that δ_control = V(s_χ) - V(s_counterfactual), changing the sign of the error and thus the dopamine transient direction.

For active avoidance with punishment predictor s_shk: V(s_shk) < 0 without control (δ < 0), but with control substituting s_fail ≪ 0, the error becomes δ > 0 (predicting safety/avoidance as a positive outcome). For No-Go suppression with reward predictor s_ngl: V(s_ngl) > 0 without control (δ > 0, promoting action), but with control substituting s_succ ≫ 0, the error becomes δ < 0 (predicting suppression as avoiding loss).

The model:
- Assumes fixed probability of control deployment on each conflict trial type.
- Computes expected state values using average-reward reinforcement learning.
- Models dopamine as a regressively weighted combination of positive and negative TD errors (greater weight on positive transients).
- Convolves the DA signal with an alpha function to match observed nucleus accumbens dopamine kinetics.
- Fits both behavior (success rates, reaction times) and dopamine measurements simultaneously.

The critical innovation is treating control itself as a discrete internal decision (apply reframing or not) rather than a continuous modulation, allowing control to be either on or off, with probabilistic deployment reflecting trial-type-dependent need.

## Implications for our model

**Direct architectural support:**

1. **Section 2.3 (DA-request mechanism) is operationalized by Lloyd & Dayan's counterfactual reframing.** The project's goal-directed DA-request neuron is a learned analog of the control decision Lloyd & Dayan model as fixed-probability deployment. If the DA request is trained to minimize its magnitude while still enabling task solution, it should spontaneously lower as the habitual system (which does not request DA) becomes competent—the emergent handoff. Lloyd & Dayan's framework shows this is neurobiologically plausible because dopamine is already used for exactly this kind of cognitive control operation.

2. **Dopamine as a control input, not a value signal, resolves the "dormancy without decay" problem.** Lloyd & Dayan's mechanism does not update weights; it modulates expression. This is the same logic as Section 2.1's W_eff = f(DA) · W. Weights trained under high DA-request remain intact when DA-request falls—they are not erased, only silenced, supporting instant reactivation (Villet's fallback).

3. **The model supports Stage 5's emergence control test.** Lloyd & Dayan show that dopamine deployment can be task-state-dependent (applied on conflict trials, not others). If the project's DA-request emerges as a learned consequence of habitual competence (as opposed to being scheduled by training time), then ablating the habitual system should cause DA-request to rise *immediately*—because the goal-directed system is suddenly needed again. Lloyd & Dayan's framework predicts exactly this kind of rapid re-engagement because the control mechanism is decision-driven, not gradient-driven.

**Supporting claims for the complexity roadmap:**

- **Stage 3 (allocentric/egocentric split):** Lloyd & Dayan do not model observation asymmetries, but their reframing mechanism is agnostic to the input space. The observation split is a project-specific design assumption; Lloyd & Dayan's dopamine control mechanism can operate on any state representation.

- **Stage 4 (value-free APE rule):** Lloyd & Dayan's model is explicitly reward-sensitive throughout; they do not directly model a value-free learning rule. However, their finding that dopamine can be controlled independently of outcome-contingent learning provides conceptual support for the idea that a system can be trained by one signal (reward) while being gated by another (dopamine), justifying the split between the goal-directed (RL-trained) and habitual (APE-trained) objectives.

- **Stages 6–7 (D1/D2 and effective rank):** Lloyd & Dayan do not differentiate D1/D2 pharmacology or measure attractor structure. These stages are independent extensions; Lloyd & Dayan's paper does not constrain them.

## Open questions / caveats

1. **Stability of reframing without explicit unlearning:** Lloyd & Dayan explicitly flag that if dopamine is positive but the TD error is counterfactual (not the "true" error), standard TD plasticity should drive the value function toward the counterfactual value, eroding the reframing over time. They propose three possible solutions (downstream gating of plasticity, temporal windows on learning, opponency mechanisms) but do not empirically resolve this. For the project, this means: before claiming the DA-request mechanism is stable across long training, we must either (a) verify that learning does not degrade the goal-directed system's weights when DA is low, or (b) implement an explicit stability mechanism (e.g., opponent prediction error). This is a hard prerequisite for Stage 5.

2. **Control cost is not explicitly modeled.** Lloyd & Dayan assume a fixed probability of control but do not model what it costs or how the cost is learned. The project's DA-request-minimization is a specific proposal for where that cost comes from (minimizing the request itself is optimised directly). Lloyd & Dayan's paper does not validate whether this specific cost structure is behaviorally or neurobiologically correct; that is empirical work for the project to do.

3. **Counterfactual states are not grounded in a circuit mechanism.** Lloyd & Dayan do not explain how the brain computes or instantiates s_fail or s_succ—they model them as black-box internal states. The project's dopaminergic implementation is more concrete (tonic/phasic signals, multiplicative gain), but remains silent on how s_fail and s_succ would be represented at the neuronal level. This is an open design problem if the goal is circuit-level mechanistic alignment.

4. **Dopamine is modeled as a scalar, not spatial or multiplexed.** Lloyd & Dayan treat dopamine as a single signal. The project uses D1/D2 affinity to generate two timescale channels (phasic vs. tonic); Lloyd & Dayan's framework does not address whether this decomposition preserves the reframing mechanism or creates new artifacts. The Stage 6 D1/D2 analysis will need to verify that D1- and D2-weighted gain functions can both implement the reframing without undermining it.

5. **Model fits to data are partial, not exhaustive.** Lloyd & Dayan note discrepancies between model and data (e.g., DA magnitude on poor-avoidance trials, prolonged DA on reward trials). While not falsifying the reframing hypothesis, these gaps mean the model is not capturing the full DA dynamics. Before using Lloyd & Dayan as a complete mechanistic template, the project should verify that emergent DA-request behavior in the CTRNN model aligns with the specific patterns Lloyd & Dayan do capture (cue-evoked sign-flip, conflict-trial deployment).

6. **Data are from nucleus accumbens core, not striatum broadly.** Lloyd & Dayan measure dopamine in NAc core specifically. The project models dopamine gating at both mPFC and DLS (functionally). Whether dopamine operates identically across these circuits, or whether there are specialized roles (e.g., NAc DA for motivation/conflict resolution, DLS DA for habitual performance control) that the simplified model misses, is unresolved. This is a detail for Stage 5 to test empirically via lesion-style ablations.

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
