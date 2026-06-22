---
citekey: naude_dopamine_2024
type: paper-note
status: read
verify: pass
topics: [dopamine, attractor, RNN, D1/D2, working-memory, mPFC, goal-directed, RL]
pdf: "My Library/files/213/Naudé et al. - 2024 - Dopamine builds and reveals reward-associated latent behavioral attractors.pdf"
full-text: "[[Naudé Dopamine Builds and Reveals Reward-Associated Latent Behavioral Attractors]]"
---

# naude_dopamine_2024

> Naudé et al. (2024) — Dopamine exerts dual roles—building latent attractors through long-term synaptic plasticity and revealing them through instantaneous modulation of neural excitability—in a recurrent network-based decision architecture.

## What the paper does

The authors present the MAGNet (Motivational Attraction to Goals by Network dynamics) model, a biophysically-motivated recurrent network embodied in an action-perception loop, combined with optogenetic validation in mice. In an un-cued conditioning task with three rewarded locations, dopamine stimulation increases learned place preference through two separable mechanisms: (1) DA-plasticity, which builds goal-encoding attractors via long-term synaptic strengthening; and (2) DA-excitability, which transiently potentiates NMDA currents to reveal and deepen those attractors' basins of attraction. The key finding: DA-excitability alone does not produce directed behavior, nor does DA-plasticity alone; only their combination generates the observed directional and energizing effects, and only from distal positions when an attractor exists (context-dependent, not content-independent gating).

## Key claims relevant to this project

- **Latent attractors as separated learning/expression**: DA-plasticity builds attractors that do *not* systematically affect neural dynamics until revealed by DA-excitability. Learning and performance are decoupled in the dynamical sense—intact weights remain unexpressed until neuromodulation activates them. This directly validates the dopaCTRNN's central architectural commitment: `W_eff = f(DA) · W`.

- **DA as an online attractor revealer, not a decision threshold**: Rather than gating all actions equally (threshold model) or providing directional specification by itself, dopamine makes goal-encoding attractors accessible from distal starting positions by widening their basins. This is compatible with dopaCTRNN's vision of dopamine as tuning the *expressed gain* of an already-learned, intact goal-directed network.

- **Context-dependence is built into attractor geometry**: DA stimulation affects behavior *only* when an attractor exists (reward context), not in neutral contexts. No effect from proximal positions (where the animal is already near the goal). This predicts that dopaCTRNN's goal-directed system should show suppressed activity not through weight decay but through neuromodulatory gating of an intact network—corroborating Villet's instant reactivation finding.

- **D1/D2 and timescale separation**: The paper focuses on D1R and NMDA, but frames dopamine-receptor interactions as potentially synergistic and context-dependent, not fixed antagonism. The project's use of D1/D2 affinity to separate phasic and tonic timescales is consistent with the mechanistic sophistication Naudé demonstrates.

- **Dual directional + activational effects emerge from single mechanism**: The paper shows that both speed and direction changes arise from the same attractor-widening operation, not from separate gating and energization pathways. This suggests dopaCTRNN need not impose two independent dopamine effects; a unified DA-expression gate may suffice.

## Mechanism / model details

**The full model has three nested levels:**

1. **Biophysical RNN** (leaky integrate-and-fire neurons with mixed selectivity for space + reward): DA modulates two pathways in parallel on different timescales. Long-term pathway: DA-dependent STDP (DA as plasticity gate) consolidates Hebbian assemblies. Fast pathway: DA-dependent NMDA potentiation transiently increases synaptic efficacy across the network, with multiplicative interaction when both pathways are active.

2. **Behavioral potential-energy surface (BPE)**: A reduced theoretical model, derived by exploiting radial symmetry, reformulates the network dynamics as a one-dimensional potential landscape. Under DA-plasticity+excitability, phasic DA induces transient unfolding of a deep, wide basin of attraction, refocusing the internal goal toward the rewarded location. DA-plasticity alone leaves a shallow basin (weak local convergence only); DA-excitability alone has no effect on goal-directed dynamics.

3. **Action-perception loop**: The network's output drives the e-mouse toward the encoded goal, which feeds spatial feedback back into the recurrent layer, creating circular causality between agent and task space. This embodied coupling is essential: without it, the attractor structure alone would not generate approach behavior.

**Key architectural features:**
- Topological neuron organization: receptive fields for position, tuned to encode current and desired locations.
- Softmax decoding of internal goal from recurrent activity (basal ganglia analogue).
- Default behavior (wall-following) in absence of strong goal-encoding bump of activity.

## Implications for our model

**Direct support for design choices (Stages 1–5):**

- **Stage 1 (single CTRNN)**: Naudé validates that recurrent networks with mixed selectivity and embodied action-perception loops do support goal-encoding attractors. The project's hypothesis that a 256-unit CTRNN will exhibit working-memory attractors (Stages 1–2) is well-grounded.

- **Stage 2–3 (observation split)**: MAGNet's use of allocentric (place-cell-like) inputs for goal-directed goal-encoding, vs. the project's allocentric/egocentric split, are structurally compatible. Naudé does not explicitly split observations, but the mixed selectivity for space + reward points toward a prefrontal (allocentric) role.

- **Stage 4–5 (DA-mediated handoff)**: The paper's core finding—that dopamine gates the *expression* of intact, learned attractors without reshaping the learned weights—is the exact substrate dopaCTRNN needs for Villet's instant reactivation. The project's hypothesis that suppression of goal-directed activity during habit formation is reversible without relearning is validated by Naudé's mechanism.

- **The DA-request neuron (Section 2.3 of project overview)**: The paper does not model a self-modulating dopamine-request output. However, Naudé's account of DA as "revealing" (making accessible) pre-learned attractors is compatible with a learned system that requests less DA as the habitual system becomes competent—the request neuron would modulate the same "reveal" function.

- **D1/D2 separation (Stage 6)**: Naudé uses D1R/NMDA as the focus, but the project's distinction between phasic (D1) and tonic (D2) timescales is consistent with the paper's two-pathway structure (fast NMDA modulation vs. slow plasticity). The project should verify whether tonic D2 tone can independently modulate the "depth" of the BPE basin separate from the phasic "widening."

- **DA-excitability = two NMDA effects, both now modelled:** Naudé's DA-excitability mechanism operates through NMDA potentiation, which simultaneously (1) increases effective recurrent synaptic gain and (2) alters integration dynamics (effective tau). The dopaCTRNN originally implemented only a readout-only proxy of effect (1): `r_out = gain·tanh(h)`, tested in E3 (da_components). The literal recurrent gain `W_eff = f(DA)·W` inside the dynamics is now implemented as `da_gain_mode="recurrent"` and tested in E9. Effect (2) (tau modulation) is tested in E6 (uniform tau shortening) and E7 (widen/deepen split). **E9 is therefore the most faithful implementation of Naudé's DA-excitability mechanism**; E3 tested a readout-only proxy. When citing Naudé in the mémoire as justification for the gain mechanism, note which experiment tests which implementation.

**Open design question (Section 2.3, project overview):** The paper does not address *how* phasic DA is triggered in self-paced actions—it treats DA as either reward-driven, spontaneous, or externally applied. The project's dopamine-request neuron is proposed as one answer, but Naudé's theory is agnostic. The paper's framework allows the request neuron to work: if the goal-directed system learns to request DA to reveal its own attractor, falling demand as the habitual system takes over would reduce the "reveal" function, quieting goal-directed output. Verification: does the DA-request neuron converge to a learned feedback law that depends on habitual competence, not training time?

## Open questions / caveats

- **Attention specification in the model**: MAGNet assumes a single rewarded location or a discrete set of known attractors. Villet's task involves learning which arm to choose on a trial-by-trial basis—the choice rule is not a fixed spatial goal but a working-memory rule (non-match). The dopaCTRNN must specify how dopamine gates the *selection* of which attractor from a working-memory set becomes expressed, not just the revealed-ness of a known attractor. Does the DA-excitability mechanism in Naudé generalize to rule-based (non-spatial) goal selection?

- **Timescale mismatch**: Naudé's fast DA-excitability (NMDA modulation on millisecond–second timescale) and DA-plasticity (STDP on learning-trial timescale) operate on different scales than the dopaCTRNN's proposed tonic/phasic separation. The project should verify whether D1/D2 affinity is sufficient to separate the same two roles at the behavioral timescale (tens of seconds for the delay period).

- **Embodiment requirement**: MAGNet requires the action-perception loop to produce goal-directed behavior; without it, the attractor structure does not drive approach. The dopaCTRNN's environment is fully embodied (the e-agent navigates the T-maze), but the project should verify that the working-memory attractor structure that emerges is robust to the specific movement ballistics and that Villet-level lesion effects (early mPFC lesion blocks learning; late DLS lesion preserves behavior) are reproduced by attractor-based rather than input-dependent effects.

- **Verification before citing in mémoire**: The instant reactivation finding (Villet) is the project's core claim. Naudé's latent-attractor mechanism is consistent with instant reactivation but does not directly demonstrate it in a dual-system switching context. Before citing Naudé as evidence for the dopaCTRNN's design, verify that the project's own Stage 5 (emergent handoff + fallback test) reproduces Naudé-compatible attractor dynamics where goal-directed weights remain intact but dopamine-suppressed during habit. Note: citing Naudé to justify the gain mechanism should distinguish E3 (readout-only proxy, already run) from E9 (recurrent gain + tau, most faithful Naudé implementation, planned) — the mémoire should not claim E3 implements Naudé's NMDA excitability in full.

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
