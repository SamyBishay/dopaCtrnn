# Project overview: a dopamine-mediated mechanism for the goal-directed-to-habitual handoff

*Source document for drafting the mémoire / rapport de stage. This captures the current, settled
state of the project's vision, mechanism, protocol, and roadmap as of this stage of development.
It is a synthesis document, not the report itself — use it to extract sections, not to copy verbatim.*

---

## 1. Vision and motivation

The project proposes and tests, in silico, a **mechanism** for a well-documented but
mechanistically unexplained phenomenon in systems neuroscience: the transition from
goal-directed (deliberative, value-sensitive) to habitual (automatic, value-insensitive)
control over the course of learning, and the reversibility of that transition when the
habitual system fails.

The target phenomenon is established behaviourally and circuit-wise by **Villet et al.
(2025)**, a chemogenetic inhibition study showing that:
- medial prefrontal cortex (mPFC) is required for *learning* a spatial working-memory task
  but not for *maintaining* performance once it is overtrained;
- dorsolateral striatum (DLS) becomes necessary for maintenance after overtraining;
- behaviour shifts from reward-devaluation-sensitive (early) to devaluation-insensitive
  (late), the classic goal-directed/habitual behavioural signature;
- critically, the mPFC contribution does not disappear — it becomes **dormant** and is
  **instantly reactivated** if the DLS is silenced (a "fallback" engram), rather than being
  erased or slowly relearned.

**Villet's study contains no dopamine.** It is a lesion/inhibition study; it establishes the
phenomenon but offers no mechanism for *why* or *how* control transfers, nor for why the
fallback is instant rather than gradual. The project's central thesis is that **dopamine is
the unstated mechanistic substrate of this protocol**: reward devaluation operates through a
value signal that is canonically dopaminergic, prefrontal dopamine is independently implicated
in maintaining model-based, goal-directed representations (a link reinforced by the
ADHD/computational literature), and striatal dopamine is the principal target of midbrain
dopaminergic projections, making it a natural candidate substrate for habitual learning.
The project is explicitly framed as **proposing and testing a candidate mechanism for Villet's
phenomena**, not as a circuit-level reproduction of a paper that has no circuit model.

A second, more novel thesis runs alongside the first: that **reward prediction errors are not
a single global scalar signal but are better understood as local computational prediction
errors**, computed differently and at different timescales by different subsystems. This
connects to the Action Prediction Error (APE) framework (Greenstreet et al.) already used for
the habitual learning rule, and is intended as the project's most original contribution —
a claim the existing literature gestures toward but for which, to the author's knowledge, no
explicit computational model yet exists.

---

## 2. Core mechanism

### 2.1 Weights vs. expression: `W_eff = f(DA) · W`

The central architectural commitment, settled after several rounds of revision, is a
separation between **learning** and **expression**:

- Synaptic weights `W` are learned slowly via the ordinary training signal (reinforcement
  learning for the goal-directed system; a value-free local rule for the habitual system —
  see below). They are never directly modulated by dopamine.
- Dopamine acts as a **fast, multiplicative gain on the expression of those weights**:
  `W_eff = f(DA) · W`. It changes how strongly the (intact) weights are expressed in the
  network's dynamics, not what the weights are.

This separation was adopted specifically because it is the only substrate consistent with
Villet's reactivation finding. If the goal-directed system's "quieting" during habit
formation were implemented as slow synaptic decay, it could not explain *instant*
reactivation when the habitual system is silenced — decayed weights do not snap back.
Under `W_eff = f(DA) · W`, the goal-directed weights remain fully intact throughout
training; only their *expression* is suppressed by low DA. Removing the suppression
(e.g. by silencing the habitual system, which removes the basis for requesting low DA)
restores full expression of an already-competent network instantly. This was an explicit,
deliberate correction of an earlier design (a sigmoid arbitration gate, `w_GD = σ(α·DA +
bias)`) that mixed system *outputs* externally rather than modulating *expression*
internally — the earlier design could not, by construction, produce an emergent handoff,
only a scheduled or externally-imposed one.

### 2.2 D1/D2 affinity as the multiple-timescale engine

The model's claim to operate over **multiple timescales** is grounded in D1/D2 receptor
pharmacology: D1 receptors have **low affinity for dopamine** and are preferentially
recruited by **phasic** (fast, transient, burst) dopamine signals; D2 receptors have **high
affinity** and are preferentially engaged by **tonic** (slow, sustained, baseline) dopamine
levels. This is a real, citable pharmacological distinction (Grace 1991; Dreyer et al. 2010),
used functionally in basal ganglia models in the Frank/OpAL tradition (Frank 2005).

This is load-bearing for the project, not decorative: it is what allows a single scalar
dopamine signal to drive *two different timescales* of network behaviour through two
differently-tuned populations, rather than collapsing into a single-timescale gain control.

**A citation correction is explicitly recorded in the project's methods documentation:**
Kutter et al. (2026) — a paper on D1/D2 control of PFC decision codes — is cited *only* for
the claim that D1/D2 functional roles are learnable and context-dependent rather than a
fixed, hard-coded stability/flexibility split. It does **not** support the phasic/tonic
affinity mapping, which is cited separately to the receptor-pharmacology literature above.
An earlier draft of the model's documentation conflated these two claims under a single
citation to Kutter; this has been corrected.

### 2.3 The DA engine: what actually drives the handoff

Two dopamine signals jointly set the network's DA level:

- **Tonic baseline**, set by external task/state variables: reward devaluation, deprivation
  state (e.g. food/water restriction), and novelty.
- **Phasic/requested component**, set by a dedicated output neuron in the **goal-directed**
  system that *requests* additional dopamine to help achieve the current goal, while being
  simultaneously optimised to **minimise the magnitude of its own request**.

This second piece is the mechanism by which the handoff is intended to be **emergent rather
than scheduled**: early in training, the goal-directed system must request substantial DA
because it is the only system capable of solving the task; as the habitual system (trained
via the value-free rule below) becomes competent, task performance no longer depends on full
goal-directed engagement, and the optimisation pressure to minimise the DA request causes the
goal-directed system to request less. Falling DA reduces `f(DA)`, which reduces the expressed
gain of the goal-directed weights — quieting that system not by design but as a learned
consequence of no longer needing it.

**A mandatory falsification/control test is attached to this claim** (recorded as a hard gate
in the project's methods documentation): after training, the habitual system must be ablated.
If the mechanism is genuinely emergent, the goal-directed DA-request must rise back to (or
remain at) a high level, because the task can no longer be carried by the (now-disabled)
habitual system. If the DA-request falls regardless of whether the habitual system is
present — i.e. if it behaves as a function of training time/episode count rather than of
habitual competence — the mechanism is a disguised schedule, not an emergent handoff, and the
result must be reported as a failure of that design regardless of behavioural accuracy.

**One question is explicitly left open** in the project's documentation, to be resolved before
implementing this mechanism: is the goal-directed DA-request neuron trained by supervision
from the reward signal (which risks circularity, since the model is partly trying to explain
how reward-derived value signals come to be implemented) or trained purely from a local
prediction-error signal (consistent with the project's local-PE thesis, but requiring the
local error signal to be specified)? This is treated as unresolved, not as a detail to be
decided unilaterally by an implementing assistant.

### 2.4 The habitual learning rule

The habitual subsystem learns via a **value-free** rule: **Action Prediction Error (APE)**,
following Greenstreet et al. (2025), under which the habitual system learns to predict the
combined system's chosen *actions*, not their *value*, plus a small efficiency-oriented
reinforcement term (a step cost and a completion bonus, with no food/task reward in its
objective). Because reward never enters the habitual system's loss function, its learned
policy is structurally — not just empirically — insensitive to reward devaluation. This is
the mechanism's account of devaluation-insensitivity once a habit is formed, and it mirrors
(independently) the mechanism used in the supervisor's reference model, which achieves the
same reward-insensitivity through cross-entropy imitation of a frozen, reward-trained network
rather than through APE — see Section 5.

### 2.5 The observation asymmetry

The goal-directed system receives an **allocentric** observation stream (spatial position
map, blocked-arm identity, task-phase labels). The habitual system receives only an
**egocentric** stream (local wall/sensor information, landmark cues, previous action) and is
deliberately denied the spatial map. This asymmetry is the project's own design choice — it
is not drawn from any single cited paper, though it is consistent with the general
allocentric/egocentric dissociation in systems neuroscience — and is explicitly flagged in
project documentation as a modelling assumption that the habitual system's inability to learn
the task unaided (Villet's finding that early mPFC lesion blocks learning) and its later
reward-insensitivity both partly depend on.

---

## 3. Task and protocol

The task is a **T-maze delayed non-match-to-position (DNMTP)** paradigm, matching the
structure of Villet et al.: a sample phase in which the agent visits one of two arms, a delay
period (configurable length; reinforced by recorded results to require ~90s-equivalent delay
to produce the task-bracketing signature reported in Villet), and a test phase in which the
agent must choose the *non-matching* arm to receive reward. Step costs and an explicit wait
penalty shape efficient navigation. The environment and protocol are held **fixed** across
every stage of model development — only the network architecture varies — so that any
behavioural or dynamical difference between stages can be attributed to architecture, not to
a moving task target.

The environment is batched (vectorised over multiple parallel episodes) and exposes
configurable observation modes (a single shared stream, an identical-but-duplicated stream for
both subsystems, or the split allocentric/egocentric streams), allowing the same environment
code to serve every stage of the architecture-complexity roadmap (Section 4) without
modification.

---

## 4. Complexity roadmap: from single network to emergent handoff

The architecture is being developed as an explicit ladder, adding **exactly one new
component per stage**, each gated by a single configuration flag, with earlier stages'
behaviour required to be reproducible unchanged when later flags are off. This discipline
exists to satisfy the project's guiding principle — **add complexity only when it is
justified by a specific test** — and to make every later stage of the model a genuine,
falsifiable experiment rather than an assumed destination.

A second invariant runs through every stage: **total recurrent network size is fixed at 256
units** throughout the entire ladder (a single 256-unit network at the simplest stage; 128 +
128 once the network splits into two areas). This number was chosen because the task itself
is informationally trivial — one bit of working memory — so capacity is never the limiting
factor; the figure is instead sized for the **interpretability analysis** the project is built
around (fixed-point finding, PCA, participation-ratio dimensionality), which is most tractable
in the 100–256 unit range used in the analogous cognitive-RNN literature (e.g. Mante &
Sussillo). Holding total capacity fixed means that any change in learned dynamics across the
ladder can be attributed to architecture, not to a confound of network size.

The same analysis pipeline — a numerical fixed-point finder (in the tradition of Sussillo &
Barak, minimising state-update speed and classifying stability via the local Jacobian),
PCA-based trajectory visualisation, and participation-ratio dimensionality — is built once
and reused **unchanged** at every stage, so that the evolving attractor structure of the
network is comparable across the whole ladder.

The stages, in order:

1. **Single recurrent network (256 units).** One CTRNN, full task observation, trained by
   ordinary actor-critic reinforcement learning. No dual-system structure exists yet. Purpose:
   establish that the task supports a clean working-memory attractor at all (expected: two
   fixed points, or a one-dimensional line attractor, coding which arm was sampled, present
   throughout the delay) and validate the analysis pipeline before any complexity is added.

2. **Two symmetric areas (128 + 128 units), identical observations, fixed equal-weight output
   averaging.** A deliberate **null control**: nothing in this stage should produce
   functional differentiation between the two areas. Its purpose is to demonstrate that
   splitting the network in two does *not* by itself produce specialisation or a handoff —
   so that any differentiation appearing later can be attributed to the *specific* asymmetry
   introduced next, not to the mere existence of two areas.

3. **Split observations.** The only change from stage 2: one area now receives the
   allocentric stream, the other only the egocentric stream (Section 2.5). Purpose: show that
   the goal-directed (allocentric) area is necessary for *learning* the task — lesioning it
   during training should block acquisition, mirroring Villet's early-lesion finding — and that
   the two areas' internal dynamics begin to diverge structurally even though no dopamine or
   special learning rule yet exists.

4. **Value-free habitual learning rule.** The habitual area's training objective switches from
   ordinary reward-based learning to the value-free APE rule (Section 2.4). Purpose: produce
   the devaluation dissociation directly — goal-directed-driven behaviour should degrade under
   reward devaluation, habitual-driven behaviour should not — reproducing Villet's central
   behavioural signature, and confirm via the analysis pipeline that the habitual system's
   attractor has become a fixed, reward-invariant stimulus-response mapping.

5. **Emergent DA-mediated handoff (the project's central milestone).** Introduces the full
   mechanism of Section 2: `W_eff = f(DA) · W`, the tonic DA baseline driven by external state,
   and the goal-directed DA-request neuron optimised to minimise its own request. No external
   mixing weight, sigmoid gate, or hand-scheduled transition exists anywhere in this design.
   Two results are required: (a) the handoff appears with no clocked transition and the
   fallback experiment reproduces (silencing the habitual system causes the goal-directed
   system's expressed gain to recover and performance to be preserved, with devaluation
   sensitivity returning); and (b) the mandatory emergence control test of Section 2.3 passes.
   The headline analysis result at this stage is tracking, across training, which subsystem's
   attractor structure governs the network's output — directly visualising control migrating
   from the goal-directed subspace to the habitual subspace as a *consequence* of the dynamics,
   rather than as an externally imposed switch.

6. **Deepened D1/D2 analysis.** With the core mechanism in place, this stage tests specific
   dopamine-manipulation predictions following from the D1/D2 affinity story (Section 2.2) —
   for example, whether biasing toward phasic vs. tonic dopamine drive reshapes the
   working-memory attractor in the direction the affinity account predicts (more transient,
   dynamic coding under phasic/D1 drive; deeper, more sustained attractor structure under
   tonic/D2 drive). A null result here — no measurable reshaping — would itself be an
   informative finding about whether this level of pharmacological detail earns its place.

7. **Effective-rank characterisation of the habitual system.** Rather than imposing a
   low-rank constraint on the habitual network's recurrent weights, this stage first
   *measures* the effective dimensionality the habitual system settles into once fully trained
   (participation ratio of delay-period activity, singular-value spectrum of the learned
   recurrent matrix, dimensionality of the fixed-point set) — testing the hypothesis, drawn
   from the cortical neural-manifold literature (e.g. Churchland/Gallego-type findings that
   overtrained motor behaviour occupies a low-dimensional subspace despite a large anatomical
   substrate), that automaticity corresponds to a dimensionality collapse the network discovers
   on its own. An explicit low-rank constraint is then applied only as a **confirmatory
   minimality probe** — sweeping rank to find the smallest value that still solves the task at
   the measured delay — rather than as an assumption imported from elsewhere.

Stages 5 and 6 are explicitly optional refinements layered on top of stage 5's milestone;
stages 0–5 constitute the required path to the project's central claim.

---

## 5. Relationship to the supervisor's reference model

The supervisor independently provided a working reference implementation (`TwoAreaACNet`)
addressing the same target phenomenon (Villet) through a **different mechanism**:

- A full-rank prefrontal-cortex area (512 units) is trained by reinforcement learning on a
  related working-memory task (trial-unique non-match-to-location, TUNL, with a delay
  curriculum increasing across training).
- A **low-rank** dorsolateral-striatum area is then trained to **imitate** the frozen,
  competent prefrontal area via cross-entropy loss on its action distribution — the
  supervisor's mechanism for reward-insensitivity, achieved structurally (the striatal area
  never sees reward) rather than via a value-free local-error rule.
- The supervisor's central explanatory variable is **rank**: low-rank recurrent weight
  structure is hypothesised to produce line/plane-attractor dynamics that constitute
  automaticity, with rank itself the parameter swept across experiments (rank 1, 2, 4, 8) and
  characterised via singular-value spectra and participation ratio.
- In this model, the handoff is **staged and curriculum-driven** (train PFC → freeze → DLS
  imitates → read out from DLS as imitation quality improves) rather than continuously
  emergent from a cost or request mechanism.

This is a genuine scientific fork, not a stylistic difference: the supervisor's account of
*why* a habit is automatic is dynamical (rank-constrained attractor geometry); this project's
account is biophysical (dopamine-gated expression of intact, learned weights). The two are
not directly comparable result-for-result, and **which framing the project is meant to operate
within is an open question that has not yet been resolved with the supervisor** — specifically,
whether the brief was to extend the supervisor's low-rank approach or to develop an independent
mechanism, since the answer materially affects how the project's contribution should be framed
and which papers (notably the assigned-but-unread Yaghoubi & Echeveste paper, situated in the
same low-rank/dynamical-systems tradition) are central versus peripheral to it. Resolving this
is treated as a prerequisite to finalising the project's framing, not as a detail to settle
after the fact.

If the dopamine-mediated mechanism does **not** produce an emergent handoff, the supervisor's
broader scientific objective — independent of whose specific model is used — is read as
identifying the **minimal necessary and sufficient conditions** for a Villet-style handoff to
occur. Under that reading, a clean negative result (e.g., the handoff requires the low-rank
constraint and does not emerge without it) is itself a legitimate and arguably stronger
finding than an unconditioned success, since it would support the supervisor's rank-as-enabling-
condition hypothesis with an independent model.

---

## 6. Literature support summary

**Core, directly load-bearing:**
- Villet et al. (2025) — the target phenomenon in its entirety (handoff, devaluation
  dissociation, dormant/reactivatable trace, task-bracketing). Contains no dopamine or circuit
  mechanism.
- Greenstreet et al. — Action Prediction Error as a value-free teaching signal; directly
  implemented as the habitual learning rule.
- Kutter et al. (2026) — D1/D2 functional roles as learnable and context-dependent (cited
  narrowly; see Section 2.2 correction).

**Supporting:**
- Literature on striatal dopamine, opponency, and cost-benefit control (the "normative
  advantages of dopamine and striatal opponency," Westbrook on striatal DA/working
  memory/reinforcement learning, and adaptive cost-benefit control work) — jointly justify the
  Go/NoGo-style opponency framing and the cost-of-engaging-the-goal-directed-system intuition
  behind the DA-request-minimisation mechanism.
- Lloyd & Dayan, "Reframing dopamine" — the conceptual anchor for treating dopamine *level*
  itself as a quantity the system actively controls, which the DA-request mechanism
  operationalises directly.

**Peripheral / cite only if a corresponding analysis is added:**
- Hippocampal predictive-coding work (motivates an allocentric/predictive map feeding the
  goal-directed stream); prefrontal neural-variability work (relevant only if exploration/
  entropy is framed as productive variability); dopamine-built attractor work (relevant mainly
  to the supervisor's framing, useful if the project's PCA analysis shows comparable attractor
  structure).

**Currently in the project's library but not well-matched to this architecture** (flagged for
removal or re-justification): acetylcholine-dopamine demixing work (no acetylcholine in this
model); spatiotemporal dopamine *wave* dynamics (this model's dopamine is a single scalar, not
a spatial wave); cortical ramping/tamping work (no motor ramping is modelled or analysed).

**To acquire — currently load-bearing but absent from the project's library:**
- Frank et al., the OpAL / basal-ganglia D1-D2 opponency line of work — needed to properly
  justify the Go/NoGo opponency design and the D1/D2 affinity-based phasic/tonic mapping,
  currently supported only by general citation rather than by papers in hand.
- Yaghoubi & Echeveste — assigned directly by the supervisor; needed to determine the
  supervisor's intended dynamical-systems framing and to resolve the open question in Section
  5. (The Echeveste paper currently in the library, on sampling-based probabilistic inference,
  is a different paper by an overlapping author and does not itself justify anything in the
  current architecture.)

---

## 7. Practical and infrastructural notes

The project currently runs locally; access to the university's Grid5000 cluster exists but is
not yet in use. Environment vectorisation (removing per-batch-element Python loops from the
task environment) is required before cluster-scale runs are practical, and is treated as
infrastructure work separate from — and not to be conflated with — the scientific content of
the roadmap above.

## 8. Broader implications, if the central mechanism succeeds

Beyond the neuroscience reproduction, the mechanism — an autonomous, cost-driven transition
from an expensive, flexible controller to a cheap, rigid one, with the expensive controller
kept dormant and instantly recoverable as a fallback — has a conceptual parallel in
control-theoretic and robotics settings: hierarchical/amortised control architectures that
distil a deliberative policy into a fast reactive one while retaining the deliberative
controller for out-of-distribution recovery; cost-aware metareasoning about when to invoke
expensive computation; and the use of a controller's effective dimensionality as a runtime
signal that a learned skill has become rote and may fail silently outside its trained regime
without the fallback re-engaging. These are framed as conceptual contributions of interest to
those fields, not as a deployable robotics result — the task itself (a one-bit T-maze
working-memory problem) is a controlled neuroscience setting, not an engineering benchmark.

