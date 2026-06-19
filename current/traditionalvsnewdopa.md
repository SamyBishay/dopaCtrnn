# Traditional vs. New Dopamine Models: A Multi-Dimensional Framework

## Overview
The classical dopamine hypothesis—centering on scalar temporal-difference reward prediction error (TD-RPE) signaling—has provided powerful explanatory leverage in computational neuroscience. However, emerging empirical phenomena reveal dimensions the TD-RPE model cannot capture. This synthesis maps four orthogonal dimensions that distinguish classical from newer models, grounding the **dopaCTRNN** approach within this landscape.

---

## The Classical View: Scalar TD-RPE

### Dominance and Motivation
The [[schultz_predictive_2015]]-era model became canonical because it elegantly unified:
- Phasic dopamine responses to reward and punishment
- Learning of state values and successor representations
- A single scalar quantity—prediction error—driving synaptic weight updates

Anchored to [[yaghoubi2026]] (comprehensive review) and [[lloyd_tamping_2015]] (vigour and gain modulation), the TD-RPE framework assumed dopamine encodes a unitary teaching signal.

### Limitations: What It Cannot Explain

| Phenomenon | Failure Mode | References |
|---|---|---|
| **Reversibility & dormancy** | Classical learning is cumulative; cannot account for rapid extinction or dormant memory reactivation | [[villet2025]] |
| **Online cognitive effects** | Does not predict how dopamine modulates working memory, attention, or cognitive flexibility in real-time decision-making | [[westbrook_striatal_2025]], [[cools_2019]] |
| **Framing dependence** | Reward prediction error should be frame-invariant; empirically, dopamine responses shift with task framing and context | [[lloyd_reframing_2023]] |
| **Value-free teaching** | Cannot explain dopamine's role in driving behavior when no explicit value signal is present | [[greenstreet_dopaminergic_2025]] |

---

## Four Orthogonal Dimensions

### Dimension 1: Spatial Organization

**Classical view:** Dopamine is a scalar neuromodulator, uniform across targets.

**New view:** Dopamine encodes spatial patterns—waves, gradient flows, or topographic maps.

- [[hamid_wave-like_2021]]: dopamine propagates as traveling waves across cortex and striatum, carrying directional information
- [[frank_adaptive_2025]]: spatial heterogeneity in DA release correlates with flexible vs. habitual action selection  
- [[cools_2019]]: receptor density distributions (D1/D2) create spatial specificity for stability-flexibility tradeoffs

**dopaCTRNN engagement:** Explicitly represents spatial geometry; the RNN state space can encode wave-like or distributed DA signals rather than assuming scalar homogeneity.

---

### Dimension 2: Receptor Identity (D1 vs. D2 Selectivity)

**Classical view:** Dopamine is dopamine; receptor subtype is mechanistically secondary.

**New view:** D1 and D2 receptors implement antagonistic or context-dependent control functions.

- [[frank_adaptive_2025]]: D1 stabilizes (persistence), D2 enables switching (exploration)  
- [[cools_2011]] and [[cools_2019]]: inverted-U relationship—optimal DA levels differ for D1-dependent stability vs. D2-dependent flexibility  
- [[jaskir_normative_nodate]]: normative models predict when D1 vs. D2 should dominate  
- [[kutter_dopamine_2026]]: D1/D2 role assignment is learnable and context-dependent, not hard-wired

**dopaCTRNN engagement:** The model can parameterize separate D1/D2 pathways or their functional balance, allowing stability-flexibility to emerge from network dynamics rather than prescriptive rules.

---

### Dimension 3: Functional Multiplicity (Beyond Reward Learning)

**Classical view:** Dopamine's primary function is reward-based learning.

**New view:** Dopamine simultaneously encodes motivation, effort, attention, and value restructuring.

- [[greenstreet_dopaminergic_2025]]: dopamine gates Active Policy Exploration (APE), decoupled from Q-learning  
- [[lloyd_reframing_2023]]: dopamine signals when task framing changes, enabling rapid value reassessment  
- [[lloyd_tamping_2015]]: dopamine modulates vigour (speed) and gain (intensity) in action execution  
- [[westbrook_striatal_2025]]: dopamine encodes effort cost, integrating metabolic and cognitive load signals

**dopaCTRNN engagement:** The RNN can learn to multiplex these functions—using dopaminergic states not just to update weights but to gate exploration, reframe values, and scale effort in parallel. The network internalizes the functional diversity.

---

### Dimension 4: Expression Gating (Dopamine as Gain Modulation)

**Classical view:** Dopamine affects learning rates and baseline firing; its role in action expression is secondary.

**New view:** Dopamine gates the effective weight (or efficacy) of learned behaviors; behavior is suppressed when DA is low and amplified when DA is high.

- [[naude_dopamine_2024]]: dopamine scales the gain of striatal output neurons, modulating whether learned actions are behaviorally expressed  
- Complementary support: [[jaskir_normative_nodate]], [[frank_adaptive_2025]], [[lloyd_reframing_2023]]  
- Mathematical framing: $W_{\text{eff}} = f(\text{DA}) \cdot W$, where dopamine modulates the effective weight matrix  
- [[villet2025]]: this gating effect explains rapid reactivation—learned synapses remain dormant (low effective weight) until DA rises

**dopaCTRNN engagement:** Directly embedded: the RNN learns when to "turn up the volume" on behavior (high DA gain) or suppress it (low DA suppression). This is a core control knob.

---

## The dopaCTRNN Within This Framework

### Which Dimensions Are Exploited?

1. **Dimension 4 (Expression Gating):** *Fully integrated*  
   - The RNN learns dopaminergic gain signals that modulate effective action weights  
   - Enables rapid suppression and reactivation consistent with [[villet2025]]

2. **Dimension 3 (Functional Multiplicity):** *Partially via APE and DA-request*  
   - The RNN can learn to signal dopamine for exploration (APE, [[greenstreet_dopaminergic_2025]])  
   - Framing-dependent value shifts emerge from network state, consistent with [[lloyd_reframing_2023]]

3. **Dimension 2 (D1/D2 Selectivity):** *Optional refinement*  
   - The model can parameterize separate pathways or learn their balance  
   - Not required for basic function but can encode stability-flexibility constraints ([[cools_2019]], [[kutter_dopamine_2026]])

4. **Dimension 1 (Spatial Organization):** *Simplified*  
   - The RNN does not explicitly model dopamine waves or spatial gradients  
   - Treats dopamine as an effective scalar or low-dimensional latent signal  
   - Trade-off: computational tractability vs. neural plausibility

### Why This Matters

The dopaCTRNN succeeds by:
- Grounding expression gating (Dim 4) as a mechanistic cornerstone, aligning with [[naude_dopamine_2024]] and [[villet2025]]  
- Allowing functional multiplicity (Dim 3) to emerge from learned RNN dynamics, avoiding hard-coded dopamine roles  
- Remaining agnostic about receptor subtype (Dim 2) at the base level, but permitting it as a learned distinction  
- Operating at the "effective signal" level (Dim 1), side-stepping spatial explicitness without losing explanatory power

This design respects the empirical scope of newer dopamine models—particularly their emphasis on dormancy, reversibility, and flexible gating—while maintaining computational tractability.

---

## References

[[schultz_predictive_2015]]  
[[yaghoubi2026]]  
[[lloyd_tamping_2015]]  
[[lloyd_reframing_2023]]  
[[villet2025]]  
[[westbrook_striatal_2025]]  
[[cools_2019]]  
[[cools_2011]]  
[[greenstreet_dopaminergic_2025]]  
[[hamid_wave-like_2021]]  
[[frank_adaptive_2025]]  
[[jaskir_normative_nodate]]  
[[kutter_dopamine_2026]]  
[[naude_dopamine_2024]]
