citekey: lloyd_tamping_2015
type: paper-note
status: read
verify: pass
topics: [dopamine, RL, APE, dual-system, working-memory, credit-assignment, attractor]
pdf: "My Library/files/264/Lloyd and Dayan - 2015 - Tamping Ramping Algorithmic, Implementational, and Computational Explanations of Phasic Dopamine Si.pdf"
full-text: "[[Lloyd Tamping Ramping; Algorithmic, Implementational, and Computational Explanations of Phasic Dopamine Signals in the Accumbens]]"
---

# lloyd_tamping_2015

> Lloyd & Dayan (2015) — Reconciles phasic dopamine TD-error theory with empirically observed ramping (gradual dopamine increase before action/reward), proposing three non-mutually-exclusive mechanisms: uncertainty-resolution in action timing, dopamine gain-control over value accumulation, and quasi-tonic discounted-vigor signals.

## What the paper does

Lloyd & Dayan tackle an apparent contradiction in dopamine theory: phasic dopamine neurons encode temporal-difference (TD) prediction errors (transient deviations from expected value), yet recent FSCV measurements show sustained dopamine *ramps* — gradual increases in extracellular dopamine in nucleus accumbens immediately preceding action or during approach to reward. Such ramps are predictable from preceding context, which should rule them out as TD errors. The paper explores three mechanisms that could generate ramping without abandoning TD theory: (1) TD errors arising from resolution of *uncertainty about when actions will execute*, communicated to the critic via efference copy; (2) dopamine's direct multiplicative effect on value-accumulation gain in a drift-diffusion decision process; and (3) a quasi-tonic signal `(1−γ)⟨V(s')⟩` arising from discounted (γ < 1) value functions, which acts as a long-run "discounted vigour" signal analogous to average reward rate.

## Key claims relevant to this project

- **Dopamine as gain-modulator**: Experimental evidence (Phillips et al.) shows dopamine transients increase probability of immediate action; a mechanistic decision-making model coupling dopamine directly to decision gain generates ramping signals and captures the negative correlation between phasic dopamine magnitude and decision latency [314–318].

- **Phasic vs. quasi-tonic dissociation**: Phasic dopamine (burst firing) and tonic dopamine (irregular basal pool firing) are independent control channels for fast transient signals and slow baseline levels respectively, controllable by different neural circuits [381–385]. This is the biophysical substrate for multi-timescale dopamine control.

- **Uncertainty resolution as TD mechanism**: When the critic receives (direct or indirect) information about actor's intention to act, and experiences timing uncertainty about when that act will execute, a TD error occurs at the moment of efference-copy confirmation — this can produce pre-response dopamine ramps without violating TD theory [253–275].

- **Value-dependent ramping under discounting**: Quasi-tonic dopamine proportional to `(1−γ)⟨V(s')⟩` explains the empirical scaling of ramps with reward magnitude and distance-to-goal in Howe et al.'s maze task; it is effectively a long-run vigor signal emerging naturally from discounted RL [329–339].

- **Dopamine release as local and modular**: Striatal dopamine release is regulated by local mechanisms (glutamate, GABA, acetylcholine) and independent of somatic dopamine-cell firing rates; spatiotemporal dopamine profile at receptors depends on release site, reuptake, and receptor target [97–101].

## Mechanism / model details

**TD-error account of pre-response ramping**: A discrete-state actor-critic model in which the actor selects action latency τ upon cue presentation. The critic receives indirect information about τ (via downstream efference copy delayed by a random time T ~ Gamma) or direct information (with added timing uncertainty). TD errors are computed under the average-reward formulation `δ = r + V(s') - V(s) - ρ` at each state transition. When the critic has direct knowledge of the actor's decision but uncertainty about execution timing, peak TD errors (and dopamine) increase with longer τ, matching Roitman et al.'s data. This mechanism requires no modification to standard TD theory.

**Dopamine gain-control in value accumulation**: A drift-diffusion decision process with time-varying gain `g(t)` proportional to dopamine concentration: `dx = g(t)[A dt + c dW]`. Tonic dopamine fluctuations (autocorrelated noise around a baseline) produce average ramping *solely through threshold-crossing statistics*: decisions are more likely when dopamine is high, creating an apparent ramp when trials are averaged backward-aligned to decision time. Phasic dopamine transients produce a stronger ramp and generate negative correlation between phasic-dopamine magnitude and latency [310–319].

**Discounted vigour**: In a discounted (γ < 1) setting, the quasi-tonic signal `(1−γ)⟨V(s')⟩` naturally emerges and maps to ramp-like sustained dopamine. This quantity changes slowly (values change modestly over time) and scales with both goal proximity and reward magnitude, explaining both the Howe et al. maze findings and their invariance to travel duration. The signal arises from the mathematical equivalence between average-reward and discounted-reward TD formulations under appropriate rescaling.

## Implications for our model

1. **DA as multiplicative gain is formally justified**: Lloyd & Dayan's drift-diffusion account (Section "A More Direct Role for Dopamine") provides a computational rationalization for dopamine's multiplicative effect on decision/expression gain (`W_eff = f(DA)·W`), grounding it in a principled decision-theoretic framework rather than architecture choice alone [Stage 5, DA-engine design].

2. **Phasic/tonic channel dissociation supports D1/D2 mapping**: The claimed independence of burst-firing (phasic) and basal-pool firing (tonic) as separate dopamine control channels directly justifies the project's use of D1/D2 receptor affinity to implement different timescales of network plasticity and gain modulation [Stage 6, D1/D2 analysis; Section 2.2 project overview].

3. **Uncertainty-resolution mechanism is orthogonal to DA-request**: Lloyd & Dayan's timing-uncertainty TD error mechanism operates in the critic's state representation (efference copy timing) and is fully compatible with the dopaCTRNN's goal-directed DA-request output. The request neuron can implement uncertainty-driven gain-modulation without conflict [Stage 5, integration point].

4. **Discounted vigour suggests quasi-tonic DA baseline**: The `(1−γ)V(s')` signal provides a theoretical account of why baseline dopamine in the model should reflect value-of-state (supporting the project's assumption that tonic DA depends on task state and reward devaluation status), not just generic motivation [Section 2.3 project overview, tonic baseline drivers].

5. **Multiple mechanisms likely coexist**: Lloyd & Dayan's conclusion that ramping may arise from multiple non-mutually-exclusive mechanisms suggests the dopaCTRNN should anticipate that pre-response dopamine fluctuations and goal-approach dopamine scaling both occur simultaneously, each driven by different circuit logic (uncertainty resolution in goal-directed, gain-modulation in both, vigor-scaling in extended tasks) [Stage 5–7 analysis plan].

## Open questions / caveats

- **Negative prediction errors asymmetry unresolved**: Lloyd & Dayan note that negative TD errors may be encoded differently (via pause in burst firing rather than hyperpolarization) in dopamine cells, but FSCV measures concentrations symmetrically. The dopaCTRNN assumes bidirectional DA modulation; if negative errors require a distinct circuit (pause-based inhibition), the model's scalar DA gain may be incomplete [verify: test whether DA suppression below baseline is behaviorally distinct from DA elevation above baseline in Stage 5 lesion tests].

- **Local release regulation independent of soma**: Because striatal dopamine release depends on local neuromodulators (glutamate, GABA, ACh), the model's assumption that a global DA signal directly multiplicatively gates all goal-directed outputs may underestimate spatial and synaptic specificity. This is a simplification acknowledged as deliberate in the project; Lloyd & Dayan flag it as a source of future complexity [design note: if Stage 6 D1/D2 analysis produces null results, consider local gating as unmodeled confound].

- **Discount factor γ and hyperbolic discounting**: Lloyd & Dayan show that ramp magnitude scales with γ but note that actual neural discounting is often hyperbolic (sum of exponentials) rather than purely exponential. The project's fixed γ is an abstraction; behavioral tests of sensitivity to delay should verify whether the learned DA-baseline adapts appropriately to task timescale (90s delay in DNMTP). [verify in Stage 5: test ramp structure across different delay lengths].

- **Efference copy assumptions critical but untestable in silico**: The timing-uncertainty account depends on efference copy being communicated to the critic just before action, yet the dopaCTRNN has no explicit anatomical separation between actor and critic. The model should verify that recurrent dynamics can implement the implicit communication this theory assumes [Stage 5 analysis: inspect goal-directed hidden-state trajectories for preparatory signals preceding action].

- **FSCV temporal resolution**: Lloyd & Dayan work with fast-scan cyclic voltammetry, which has ~100ms resolution; the dopaCTRNN operates on millisecond timescales (or configurable via learning rates). Ramping-signal structure may depend on integration timescales not directly comparable between theory and simulation [verify: conduct robustness check on DA gain time constant matching FSCV measurement timescale].

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
