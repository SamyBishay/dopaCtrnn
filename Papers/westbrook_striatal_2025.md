---
citekey: westbrook_striatal_2025
type: paper-note
status: read
verify: pass
topics: [dopamine, striatum, working-memory, RL, credit-assignment, APE]
pdf: "My Library/files/235/Westbrook et al. - 2025 - Striatal dopamine can enhance both fast working memory, and slow reinforcement learning, while reduc.pdf"
full-text: "[[Westbrook Striatal dopamine can enhance both fast working memory, and slow reinforcement learning, while reducing implicit effort cost sensitivity]]"
---

# westbrook_striatal_2025

> Westbrook et al. (2025) — Striatal dopamine simultaneously enhances fast working-memory deployment and slow reinforcement learning while reducing implicit discounting of effort costs, with these effects dissociable across dopamine synthesis capacity and pharmacological manipulations.

## What the paper does

Double-blind pharmacological-neuroimaging study (N=100) isolating the contributions of working memory (WM) and reinforcement learning (RL) to stimulus-response learning. Participants completed an RLWM task varying set size (2–5 items) across training blocks, with dopamine measured via [18F]-FDOPA PET imaging and manipulated via methylphenidate (dopamine reuptake inhibitor) and sulpiride (D2 antagonist). Computational modelling decomposed behaviour into a WM component (instantaneous, capacity-limited, decay-susceptible) and an RL component (incremental, capacity-unlimited). Key findings: higher dopamine synthesis capacity and methylphenidate increased reliance on WM and accuracy; sulpiride reduced WM contribution and performance; methylphenidate blunted implicit effort-cost discounting (the penalty subjects assign to rewards earned under high cognitive load).

## Key claims relevant to this project

- **Dopamine promotes both fast and slow learning pathways simultaneously**: methylphenidate boosts RL rates specifically in high-dopamine-synthesis individuals, while baseline dopamine synthesis capacity predicts baseline WM reliance. This directly supports the dual-timescale architecture (Stage 1 claim: both systems solve the task; Stage 5 claim: dopamine gates expression of both, not learning).

- **Dopamine modulates effort sensitivity independently of learning rates**: methylphenidate blunts the implicit effort-cost discount applied to rewards earned under high cognitive load [p. 39, 51]. This is mechanistically distinct from enhanced learning and maps to the project's hypothesis that DA regulates the cost of engaging deliberative (goal-directed) control.

- **WM reliance is a dopamine-controlled trait parameter**: individual differences in dorsal caudate dopamine synthesis capacity causally predict baseline propensity to rely on WM [p. 31], not transient task demand; dopamine antagonism (sulpiride) reduces WM reliance, suggesting dopamine maintains the *stability* of WM representations [p. 57–58], consistent with `W_eff = f(DA) · W` (dopamine as multiplicative gain on expression, not learning).

- **RL and WM systems must be disentangled**: the paper's core methodological contribution is a decomposition model showing that prior dopamine studies conflating the two have misestimated RL learning rates [p. 53]. Controlling for WM contribution reveals dopamine's effects on RL are real but smaller than apparent in aggregate behaviour.

- **Sulpiride's effects on WM are distinct from D1-mediated effects**: sulpiride reduces WM involvement and causes accuracy decline both early and late in blocks [p. 57–58], implying D2 (tonic) dopamine stabilises WM content over time; no direct D1 vs D2 comparison is provided, so the phasic/tonic distinction is not empirically tested.

## Mechanism / model details

Computational model with six free parameters: two for WM (reliance weight ρ, decay rate), two for RL (learning rate α, temperature τ), one for temperature across both systems. WM component uses instantaneous updating with capacity saturation; RL uses standard temporal-difference update. Model fit via maximum likelihood (mfit toolbox). Hierarchical regression of ρ (WM reliance) on dopamine synthesis capacity and drug condition shows dopamine as a main effect on trait WM preference, not task-state-dependent allocation. No explicit mechanistic model of *how* dopamine achieves this (e.g., via receptor subtypes, timescale filtering, or gain modulation); the paper is empirical, not computational, regarding dopamine's substrate.

## Implications for our model

- **Stage 1 validation**: Westbrook confirms that WM and RL are jointly operative in learning and can be behaviourally dissociated. The project's Stage 1 (single CTRNN solving the task via attractor dynamics) should produce both forms of learning if the network is to match human behaviour.

- **Stage 2–3 support for observation split**: the finding that dopamine synthesis capacity predicts baseline WM propensity (not task-determined allocation) supports the assumption that observation asymmetry (allocentric for goal-directed, egocentric for habitual) drives structural, dopamine-independent divergence initially, with dopamine then modulating the *relative expression* of both.

- **Core mechanism (`W_eff = f(DA) · W`) support**: the dissociation between dopamine synthesis capacity effects (predicting WM *reliance*, not learning rate) and methylphenidate effects (boosting RL *rate* in high-synthesis individuals) suggests dopamine has at least two independent roles — one related to maintaining expressed gain of a capability (consistent with multiplicative gain), another related to training plasticity (consistent with boosting learning rates). This is compatible with the project's separation of learning and expression.

- **Effort-cost sensitivity as a dopamine-regulated parameter**: methylphenidate's blunting of implicit effort-cost learning is the closest empirical correlate to the project's DA-request minimisation mechanism. If the goal-directed system optimises to minimise its own dopamine request, it is solving a cost-of-engagement problem; Westbrook provides evidence that dopamine regulates sensitivity to *effort* as a cost, not just performance per se. This supports the framing of the DA request as solving a real computational problem the brain faces.

- **Stage 5 milestone constraint**: the paper does not directly test handoff emergence or fallback recovery, and uses no dopamine-state manipulation during trained performance. However, the finding that dopamine synthesis capacity predicts stable trait differences in WM reliance — and that sulpiride *reduces* reliance rather than gradually shifting it — is consistent with dopamine controlling *whether* a system's weights are expressed (an all-or-nothing gate in the limit) rather than gradually shifting learning parameters.

- **Caution on D1/D2 story (Stage 6)**: the paper uses only PET imaging (dopamine synthesis) and non-selective dopamine manipulations (methylphenidate, sulpiride). No D1 vs D2 dissociation is provided. The project's hypothesis that D1/D2 affinity differences drive multiple timescales (Stage 2.2) cannot be directly tested against Westbrook's data.

## Open questions / caveats

- **Causality and direction of effect**: Westbrook measures dopamine synthesis capacity cross-sectionally and correlates it with WM reliance. The causal direction — does high dopamine capacity cause high WM reliance, or does high baseline WM reliance recruit more dopamine? — is not resolved. For the project, this matters for whether dopamine is the *driving* mechanism or merely a permissive/supporting signal.

- **Mechanism of effort-cost blunting**: Westbrook shows methylphenidate blunts implicit effort-cost discounting but does not explain *how*. Is it because dopamine reduces the *weight* assigned to effort as a cost variable? Or because it reduces *learning* about effort? Or because it alters reward valuation itself? The project's DA-request mechanism assumes dopamine directly regulates the decision to engage the goal-directed system; Westbrook is agnostic about the architecture.

- **Stability vs. gain**: the paper interprets sulpiride's reduction of WM reliance as indicating dopamine maintains WM *stability* (perhaps through D2 effects on PFC neurons). But the data are also consistent with dopamine controlling the *gain* at which WM information is read out. The project assumes the latter (`W_eff = f(DA) · W`); distinguishing these requires either formal modelling or direct manipulation of PFC dopamine (not done here).

- **No reversal/fallback test**: unlike the project's mandatory Stage 5 control test, Westbrook does not ask whether the handoff is reversed if one system is removed or disabled. The task structure (small set sizes, fast learning) may not support habit formation as strongly as the T-maze DNMTP with 90s delay, limiting generalisability to the Villet phenomenon.

- **D1 vs D2 untested**: the project requires a specific mapping of D1 (phasic, low-affinity, flexibility) vs D2 (tonic, high-affinity, stability) roles. Westbrook uses sulpiride (D2 antagonist) but no D1-selective agent, so the differential timescale claim cannot be evaluated directly against this paper.

---

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
