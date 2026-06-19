---
citekey: greenstreet_dopaminergic_2025
type: paper-note
status: read
verify: pass
topics: [dopamine, APE, habit, RNN, DLS, credit-assignment, dual-system]
pdf: "My Library/files/212/Greenstreet et al. - 2025 - Dopaminergic action prediction errors serve as a value-free teaching signal.pdf"
full-text: "[[Greenstreet Dopaminergic Action Prediction Errors Serve as a Value-Free Teaching Signal]]"
---

# greenstreet_dopaminergic_2025

> Greenstreet et al. (2025) — Demonstrates that movement-related dopamine activity in the tail of striatum encodes action prediction error (APE), a value-free teaching signal that reinforces stimulus–action associations independent of reward, and shows that APE and reward prediction error (RPE) work in tandem across different striatal regions to support learning.

## What the paper does

Mice learn an auditory frequency discrimination (cloud-of-tones, COT) task while dopamine dynamics are recorded and manipulated. The tail of striatum (TS) releases dopamine in a contralateral-movement-locked, outcome-insensitive pattern that decreases with training—characteristic of an action prediction error. Optogenetic stimulation of TS dopamine at choice time drives learning of stimulus–action associations, reinforcing repeated actions. Computational modeling shows that a value-free controller updated by APE cannot learn the task alone but, paired with a reward-based RPE controller, learns faster and consolidates stable sound–action associations. The model also predicts and experiments confirm that control transfers from goal-directed (RPE-driven) to habitual (APE-driven) systems over training as the value-based system's weights decay with low RPE.

## Key claims relevant to this project

- **APE is structurally value-free**: TS dopamine response does not vary with reward size or predicted value; it encodes only the discrepancy between executed and predicted action in a given state. This guarantees devaluation-insensitivity once the habitual system is trained via APE [lines 109–112].

- **APE is movement-not-outcome encoded**: TS dopamine scales with contralateral movement and turn angle, is present in freely-moving open-field behavior, and persists when sound cues are omitted; VS dopamine shows opposite pattern (outcome-locked, cue-dependent) [lines 68–80, 77–78].

- **APE signal decreases over learning**: Movement-related TS dopamine is largest early and smaller as mice predict their own actions, consistent with error reduction; VS cue-related dopamine grows, consistent with RPE [lines 91–95].

- **Dual controllers with APE+RPE: faster learning, automatic transfer**: A model with only APE fails to learn; RPE alone learns the task; APE+RPE learns faster. Critically, if the goal-directed (RPE) weights decay when RPE signal is low, control naturally transfers to the habitual (APE) system without external gating [lines 147–162].

- **Behavioral instantiation of the transfer**: Early in training, inactivating TS SPNs has no effect; as training progresses (after >65% performance), inactivation disrupts behavior, showing that habitual control is not present from the start but emerges [lines 164–167].

- **APE reinforces state–action, not movement-modulation**: Optogenetic TS dopamine stimulation at choice time causes a learned bias toward the stimulated port over multiple trials, not immediate movement; it does not affect movement initiation or ongoing kinematics [lines 118–128].

- **Frequency-specific cortico-striatal plasticity supported by APE**: The dual model learns hemisphere-specific sound–action associations that match empirical plasticity patterns [lines 169–171].

## Mechanism / model details

**Dual value-based/value-free RL model**: An actor-critic learns a value-based (goal-directed) policy P_GD(a|s) and value V(s) via RPE; a separate value-free actor learns a habitual policy P_H(a|s) via APE, which predicts actions not values. APE = Pr[action_predicted | state] − 1{action_taken}, updated on each trial based on the recent history of sound–action pairings. The goal-directed actor's weights decay proportionally to RPE magnitude: if RPE ≈ 0 (task is learned, predictions are accurate), the weights decay, reducing the expressed contribution of that system. The habitual system, trained only to repeat successful actions and minimize step cost (no reward in its loss), is structurally insensitive to outcome changes. Over training, the goal-directed system solves the task (RPE-driven learning works); as it does, RPE shrinks, goal-directed weights decay, and the habitual system's learned associations take over control [lines 147–162]. This produces an *emergent* handoff: control transfers not by design but as a learned consequence of the goal-directed system no longer needing high expression.

## Implications for our model

**Stage 4 (Value-free habitual learning rule) is directly validated**: Greenstreet provides both the mechanistic justification and the experimental proof that APE is a real dopaminergic teaching signal operating in striatum. The project's use of APE as the habitual learning rule is not a theoretical assumption but a measured biological phenomenon. This strengthens the claim that devaluation-insensitivity in the dopaCTRNN is not an accident of the architecture but a direct consequence of the learning rule's structure [Section 2.4].

**Stage 5 (Emergent DA-mediated handoff) maps directly onto Greenstreet's dual-system transfer**: Greenstreet's finding that control transfers when goal-directed weights decay (lines 158–162) is mechanistically isomorphic to the dopaCTRNN's `W_eff = f(DA) · W` design, where low DA suppresses expression of intact goal-directed weights. Both are implementations of the same idea: the goal-directed system remains intact but dormant. However, **Greenstreet's mechanism (weight decay) differs from the project's (dopamine-modulated gain)**. Weight decay is a slow process; dopamine modulation is fast. The dopaCTRNN's design is more consistent with Villet's finding of *instant* reactivation upon habitual-system silencing [Project overview, Section 2.1]. This is not a contradiction—it is a mechanistic refinement. Greenstreet's model shows that *some* form of control transfer works; the dopaCTRNN specifies a faster, dopamine-gated implementation that better explains Villet's data. Both should predict the same final behavior (emergent handoff with fallback reactivation), but at different timescales.

**The DA-request mechanism (Stage 5, Section 2.3) is novel and untested in Greenstreet**: Greenstreet shows that dopamine drives learning via APE and RPE, and that low dopamine (from weight decay) can suppress the goal-directed system. The dopaCTRNN goes further by proposing that the goal-directed system *actively minimizes its own dopamine request* as a cost-control mechanism. This is not tested in Greenstreet's work and should be treated as a hypothesis to validate with the mandatory ablation experiment [Project overview, Section 2.3]: after training, ablate the habitual system and measure whether the goal-directed DA-request rises again, confirming that the request is sensitive to habitual competence rather than being a function of training time alone.

**D1/D2 affinity (Stage 6) is orthogonal to Greenstreet**: Greenstreet's dopamine signal is unsplit; the tail-striatum dopamine drives both APE and the general suppression of goal-directed expression. The dopaCTRNN's hypothesis that phasic (D1-recruited) and tonic (D2-recruited) dopamine drive different timescales within the goal-directed system [Project overview, Section 2.2] is a detail Greenstreet does not address and which would need to be tested independently.

**The observation asymmetry (Section 2.5) is unconstrained by Greenstreet**: Greenstreet does not manipulate observation streams; both systems in her model have full task information. The dopaCTRNN's choice to deny the habitual system the allocentric spatial map is an independent architectural assumption, justified by consistency with Villet's finding that early mPFC lesion blocks learning [Project overview, Section 2.5]. Greenstreet's data do not falsify this; they also do not require it.

## Open questions / caveats

- **Timescale of the transfer**: Greenstreet achieves transfer via weight decay, a slower process. The dopaCTRNN proposes dopamine-modulated gain suppression, which should be faster. The paper does not directly measure the speed of control transfer, so whether Greenstreet's mechanism alone could explain Villet's *instant* reactivation is unclear. This is a quantitative, not conceptual, gap.

- **Is APE sufficient for the handoff, or is weight decay required?** Greenstreet's model shows that both APE learning and weight decay together produce transfer; the paper does not isolate whether APE alone (without decay) would suffice. For the dopaCTRNN, this matters: if APE alone drives a handoff via some other mechanism (e.g., learned attention or gating within the habitual system), then dopamine-modulated gain may be redundant. This should be checked empirically.

- **Threat prediction error (TPE) vs. APE in the same region**: Greenstreet notes that TS dopamine also encodes threat-related transients in the same mice [lines 196–204]. Whether and how TPE and APE share a substrate, are multiplexed, or operate at different timescales is unresolved. For the dopaCTRNN, which models goal-directed and habitual learning in appetitive tasks, this is a second-order concern but should be noted: the model may not account for dopamine in aversive/avoidant contexts.

- **Value-free does not mean value-insensitive in the long term**: Greenstreet clarifies [lines 216–221] that APE-trained associations are "value-free" in their learning rule but not necessarily in their ultimate function: if the action was *chosen* (via RPE) in pursuit of value early in training, the APE system will have learned to repeat it, encoding value indirectly through repetition. The dopaCTRNN's claim that habitual behavior is *structurally* insensitive to outcome devaluation relies on this logic but does not distinguish empirically between "no reward in the loss" and "no value in the learned representation." Villet's devaluation test isolates this; replication with the dopaCTRNN model should confirm.

- **Why does APE exist if the system is not advantageous early on?** Greenstreet proposes several roles [lines 223–226]: forming a Bayesian prior for action under uncertainty, enabling off-policy learning. The dopaCTRNN does not currently model uncertainty-dependent arbitration or off-policy correction, so whether APE provides these benefits in the model is untested.

[[Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff]] [[PAPERS_INDEX]]
