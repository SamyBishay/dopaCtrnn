---
citekey: yaghoubi2026
type: paper-note
status: read
verify: pass
topics: [dopamine, RPE, credit-assignment, hippocampus, TD-learning, attractor, temporal-coding]
pdf: ""
full-text: "[[Yaghoubi Predictive coding of reward in the hippocampus]]"
---

# yaghoubi2026

> Yaghoubi et al. (2026) — Reward representation in hippocampal CA1 undergoes a weeks-long backward shift during learning, moving from direct reward encoding to prediction of reward-preceding cues, explainable by temporal difference reinforcement learning.

## What the paper does

Tracks calcium imaging of CA1 neurons across weeks as mice learn a delayed non-matching-to-location (DNMTP) task. Core findings: (1) reward-encoding cells decline from 8.5% of recorded neurons early to lower proportions with experience; (2) simultaneously, cells encoding task cues *before* the reward (screen choice, reward approach) increase; (3) individual reward-tuned neurons exhibit a "backward shift," with 21% transitioning from reward-encoding to pre-reward-encoding over sessions. A temporal-difference (TD) model with Gaussian place-cell basis functions reproduces this backward shift, showing that reward-predictability (discount factor >0.1) is necessary to drive the reorganization.

## Key claims relevant to this project

- **Backward temporal coding drives learning**: Across weeks, hippocampal populations restructure to encode reward *predictors* rather than reward itself—a circuit-level implementation of the credit-assignment principle central to TD learning. [Fig 3–5; population MI, single-cell tracking]

- **Reward representation decreases with experience, not performance**: The decline in reward-cell proportion is explained primarily by session number (days elapsed), weakly by task proficiency. This dissociation suggests learning-stage, not competence-stage, controls the reorganization. [Fig 2; linear model variance decomposition]

- **TD error as a basis for place-cell reorganization**: The paper's model shows place cells shift backward under TD error modulation proportional to (V(s') - V(s)), mirroring dopaminergic RPE dynamics. The backward shift is **contingent on a discount factor >0.1**—incorporating future state values is essential. [Fig 7; model ablation]

- **CA1 mirrors VTA dopamine dynamics**: The observed hippocampal patterns "resemble those of the dopaminergic output of the ventral tegmental area (VTA)... both gradual decrease in reward response coupled with gradual increase in response to reward-predicting cues during learning; both exhibit gradual backward temporal shift of the error signal." [Discussion; explicitly framed as circuit-level parallelism with dopamine]

## Mechanism / model details

The TD model treats the screen-to-reward sequence as a 1D MDP (8 states: choice at screen → reward port nose poke → reward delivery). Place cells (Gaussian basis functions) tile this space. A critic computes state values V(s) via TD; the TD error δ(t) = r + γV(s') - V(s) modulates the peak positions of place cells backward—reward-proximal cells move toward the screen state. Three patterns emerge: (1) reward-proximal cells shift monotonically backward; (2) reward-approach cells first move forward, then backward; (3) screen cells shift forward late. The backward shift is **contingent on discount factor γ > 0.1**; when γ < 0.1, cells remain reward-over-represented.

## Implications for our model

**Direct constraints:**

1. **Credit-assignment timescale and the DA-request mechanism (Stage 5, H3):** The paper shows that *learning-driven restructuring of neural codes happens over weeks*, not seconds. In the dopaCTRNN, the goal-directed DA-request neuron is trained to *minimize its own phasic dopamine request*. This request must settle on a timescale compatible with the habitual system's learning—the paper suggests this occurs gradually as the habitual system (trained on APE) accumulates competence. The backward shift observed here is the *encoding* analogue of what should happen in the dopaCTRNN's goal-directed subsystem's effective weights: as the habitual system becomes competent, the goal-directed DA request falls not instantly but over training (mirroring the weeks-long timescale).

2. **Attractor geometry and the allocentric/egocentric split (Stage 3, observation asymmetry):** The paper's reward-predictive cells form a **backward-shifted line attractor**, coding the path to reward not the reward itself. In the dopaCTRNN, the goal-directed subsystem receives allocentric observations (spatial map, arm identity) and should learn to form a similar *predictive* attractor—one that represents the path to the goal, not the goal itself. The hippocampal circuit's structure (allocentric cognitive map, reward prediction) is directly compatible with the goal-directed subsystem's role. The egocentric habitual system, by contrast, never sees this predictive structure and learns stimulus-response mappings (APE), consistent with reward-insensitivity.

3. **RPE as a local, compartment-specific signal (Section 1, local-PE thesis):** The paper's model shows TD errors *modulate place cells locally* (each cell's peak position is shifted by δ(t)). This is consistent with the project's claim that RPE is not a single global signal but computed locally by different subsystems. Here, CA1 computes a local TD error for spatial prediction; in the dopaCTRNN, the goal-directed subsystem should compute its own local prediction error for value, and the habitual system computes a local action-prediction error (APE). The paper supports the architectural claim that prediction errors can be subsystem-specific.

4. **Dopamine as a teaching signal operating over extended timescales (Section 2.2, D1/D2 affinity):** While the paper does not explicitly manipulate dopamine, it notes that hippocampal backward-shifting mirrors dopaminergic dynamics in VTA. The implication is that dopamine (via phasic RPE signals) drives this reorganization over weeks. In the dopaCTRNN, phasic dopamine (D1-recruited) should drive goal-directed learning via value-based RL, while tonic dopamine (D2-recruited) sets the baseline DA level that gates W_eff. The paper's timescale constraint (weeks for full restructuring) suggests phasic dopamine's learning signal must operate continuously; the model does not show what happens if that signal is suppressed prematurely, relevant to testing the DA-request-minimization hypothesis.

5. **Backward shift as the encoding basis for habit formation (Stages 4–5, habitual-learning rule):** The paper shows that reward-encoding cells *disappear* as the task is learned; in their place, pre-reward-encoding cells appear. This is the opposite of what happens in a naive animal: early reward-over-representation (cells cluster at reward sites), then backward shift to reward *predictors*. In the dopaCTRNN, the habitual system trained on APE should *never* show this backward-shift pattern—it learns stimulus-response, not stimulus-value-prediction. If, in the analysis at Stage 5, the goal-directed subsystem's attractor structure exhibits this backward shift while the habitual subsystem's does not, it would support the claim that the two systems learn fundamentally different codes. Conversely, if both subsystems show the same backward-shift pattern, it would suggest the dopamine-gating mechanism alone is insufficient to force a structural difference in learning.

## Open questions / caveats

- **Causality of DA in the backward shift:** The paper observes that hippocampal dynamics mirror VTA dopamine, but does not manipulate dopamine to confirm it drives the shift. In the dopaCTRNN, does blocking the goal-directed DA-request signal prevent the backward shift and block task learning? This is a critical test of whether dopamine is truly the substrate, or merely correlates with a schedule-driven reorganization.

- **Habitual system's access to hippocampal code:** The paper does not address whether downstream structures (e.g., striatum) can access or exploit the hippocampal backward-shifted code, or whether habitual learning in striatum operates independently. In the dopaCTRNN, the habitual subsystem is denied allocentric input by design; does this design choice *force* it to learn stimulus-response (reward-insensitive) mappings, or would it converge there naturally? The project's observation asymmetry is partly validated by the paper's hippocampal-circuit structure but not yet by a direct test.

- **Discount-factor dependency:** The model's backward shift is contingent on γ > 0.1. What is the effective discount factor in the dopaCTRNN's goal-directed subsystem? If it is <0.1 (short-horizon, myopic), the backward-shift mechanism may not operate, and the goal-directed attractor may remain reward-over-represented throughout training. This would affect whether the observed dynamics match the hippocampal pattern.

- **Single-system vs. dual-system learning timescales:** The paper tracks a *single* hippocampal system over weeks. In the dopaCTRNN, two systems are learning in parallel at different rates (goal-directed by supervised RL, habitual by APE). Do they cross at a critical competence threshold, or do they asymptotically converge? The paper's timescale (weeks) is much longer than typical RL training; whether the dopaCTRNN's handoff will exhibit similar extended timescales is untested.

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
