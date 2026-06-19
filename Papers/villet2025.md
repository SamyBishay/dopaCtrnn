---
citekey: villet2025
type: paper-note
status: read
verify: pass
topics: [dopamine, habit, working-memory, mPFC, DLS, dual-system, devaluation, attractor]
pdf: ""
full-text: "[[Villet Cortico-striatal dynamics across working memory stages]]"
---

# villet2025

> Villet et al. (2025) — A chemogenetic inhibition study showing that dorsolateral striatum mediates automatized maintenance of spatial working memory while medial prefrontal cortex governs learning, with behavior shifting from reward-sensitive to reward-insensitive and goal-directed engrams remaining dormant yet instantly reactivatable when the automatic system fails.

## What the paper does

Mice learned a delayed non-match-to-place (DNMTP) task in a T-maze over repeated days. The authors used chemogenetic inhibition (hM4Di + CNO) to selectively silence mPFC or DLS during learning vs. maintenance phases, and devaluation tests to measure whether behavior was goal-directed (reward-sensitive) or automatic (reward-insensitive). Over training, mice shifted from reward-devaluation-sensitive (learning) to devaluation-insensitive (maintenance), marking the goal-directed-to-habitual transition. mPFC was essential for learning but dispensable for maintenance; DLS inhibition during maintenance did not impair performance but *did* restore devaluation-sensitivity, indicating reactivation of goal-directed strategy. Simultaneous mPFC+DLS inhibition during maintenance impaired performance, showing both structures are necessary but in a contingent, fallback-based arrangement.

## Key claims relevant to this project

- **Behavioral transition is real and staged**: devaluation sensitivity drops sharply between learning (~70% success) and maintenance (~80% criterion), marking a clean goal-directed→habitual switch [devaluation protocol, Figs 1-2].
- **mPFC is necessary for learning, not maintenance**: mPFC inhibition during learning blocks acquisition; inhibition during maintenance has zero effect on performance [mPFC lesion results, Figs 3-4].
- **DLS is necessary for automaticity but not learning**: DLS inhibition does not affect learning rate; inhibition during maintenance does not impair raw accuracy, but reveals reactivation of goal-directed behavior via devaluation re-sensitivity [DLS lesion + devaluation, Fig 5].
- **Goal-directed engrams are dormant, not erased**: DLS-inhibited mice instantly switch back to goal-directed behavior (devaluation-sensitive), not gradually relearn. Simultaneous mPFC+DLS inhibition impairs performance, proving mPFC is not fully disengaged [dual-inhibition experiment, Fig 6].
- **Task-bracketing structure is critical**: fixed 90 s delay between sample and test phases may reinforce the DLS's temporal-bracketing signature, with implications for when the automatic system "locks in" [Discussion, Methods].

## Mechanism / model details

No computational model is provided; this is a purely experimental study. However, the paper proposes a conceptual mechanism: the memory trace (goal-directed rule) remains in a latent, low-energy state within the mPFC during maintenance, while the DLS maintains behavioral output via automatic stimulus-response. The dormancy is functional rather than structural—the weights/connections are intact and can be reactivated instantly if the DLS-driven pathway is disrupted. This is distinct from synaptic decay (which would require slow relearning) and from structural transfer (which would preclude instant fallback).

## Implications for our model

This paper is the **foundational target phenomenon** for the dopaCTRNN project and directly constrains multiple design elements:

- **`W_eff = f(DA) · W` architecture (Section 2.1)**: The instant reactivation finding rules out slow synaptic weakening as the mechanism of goal-directed quieting. Only multiplicative gain-modulation (dopamine-gated expression of intact weights) can explain reactivation without delay. This is the primary justification for the separation of learning (W) from expression (f(DA)·W).
- **Dual-system architecture (Stage 2–3, roadmap)**: Goal-directed area (mPFC analog) and habitual area (DLS analog) must coexist from Stage 2 onward. Stages 2–3 directly test the prediction that lesioning the goal-directed area during learning blocks task acquisition.
- **Value-free habitual learning rule (Stage 4)**: The devaluation dissociation is the core empirical signature. Stage 4's APE-based rule must produce devaluation-insensitive behavior on its own, demonstrating that reward-insensitivity is *structural* (reward never appears in the habitual loss function) not just empirical.
- **Emergent handoff via DA-request mechanism (Stage 5)**: The dormancy finding is critical here. The DA-request neuron must be designed such that as the habitual system becomes competent, the goal-directed system's request for dopamine falls *of its own accord*, suppressing goal-directed expression not by external schedule but because the goal-directed system has become redundant. The mandatory ablation-control test (silencing the habitual system should re-elevate DA-request) directly mirrors Villet's fallback experiment.
- **Working-memory attractor (Stages 1, 5)**: The 90 s delay and task-bracketing signature imply that the attractor structure during the delay period is a key observable. Early stages must reproduce a clean working-memory fixed-point or line-attractor; later stages must show migration of the dominant attractor from goal-directed to habitual subspace.

## Open questions / caveats

- **No dopamine data in the paper**: Villet et al. measure no dopamine levels, receptor dynamics, or DA-dependence. The dopamine hypothesis for why this handoff occurs is the project's own proposal; the paper establishes only the phenomenon, not its chemical substrate. The project must falsify or confirm the claim that dopamine is *the* mechanism.
- **90 s delay is task-specific**: The paper notes the delay may reinforce task-bracketing. The critical timescale for the dopaCTRNN's `f(DA)` function (phasic vs. tonic affinity mapping) is unspecified. Depends how delay affects which DA system dominates.
- **Observation split (allocentric vs. egocentric) is a project assumption, not Villet's**: The paper provides no evidence that DLS receives only egocentric information or that the habitual system's inability to learn unaided is due to lack of spatial map access. This is a design choice with downstream consequences; if habitual learning *can* occur with full allocentric observation in the actual brain, Stage 3's prediction (goal-directed lesion blocks learning) would fail.
- **Devaluation timing unclear**: The paper conducts devaluation during learning at 70% success (sub-criterion) and during maintenance at 80% criterion. The boundary between phases is operationally sharp but dynamically may be gradual. If the handoff occurs gradually over that 70–80% window, Stage 4–5 predictions about when devaluation-insensitivity emerges may not align.
- **No reversibility test during learning**: The paper does not test whether silencing mPFC during learning can be compensated by DLS (only tests during maintenance). Unknown whether DLS could solve the task with goal-directed system inactive from the start.

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
