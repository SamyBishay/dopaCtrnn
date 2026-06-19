# PAPERS_INDEX

> Routing table — one line per paper. Stop here unless you need more; follow [[citekey]] for project-specific notes, or go to the full-text `.md` / PDF for verification.

---

## ⭐ Target paper

**Villet et al. (2025)** — mPFC required for learning, DLS for habitual maintenance; goal-directed engram dormant but instantly reactivable; no dopamine mechanism provided. `[working-memory, mPFC, DLS, habit, dual-system, devaluation]` → [[villet2025]]

---

## [SUPERVISOR] Recommended reading

**Yaghoubi et al. (2026)** — Hippocampal reward coding shifts backward over weeks of learning (TD-like predictive shift from reward to cues). `[RPE, credit-assignment, hippocampus]` → [[yaghoubi2026]]

**Echeveste et al. (2020)** — E-I RNN optimized for sampling-based inference spontaneously exhibits gamma, variability, and onset transients — cortical motifs as functional signatures. `[CTRNN, RNN, attractor]` → [[echeveste2020]]

---

## Core mechanistic support (directly load-bearing for the model)

**Greenstreet et al. (2025)** — Tail-of-striatum DA encodes APE (value-free); parallel with RPE in other regions; justifies the habitual learning rule. `[dopamine, APE, DLS, dual-system]` → [[greenstreet_dopaminergic_2025]]

**Kutter et al. (2026)** — D1 promotes transient PFC coding, D2 promotes persistent attractor-like coding; grounds the D1/D2 timescale split in the model. `[dopamine, D1/D2, mPFC, working-memory, attractor]` → [[kutter_dopamine_2026]]

**Frank (2025)** — Review: heterogeneous DA dynamics across D1/D2 populations mediate cost-benefit meta-control; OpAL* framework. `[dopamine, D1/D2, striatum, RL, credit-assignment]` → [[frank_adaptive_2025]]

**Jaskir & Frank** — OpAL*: opponent D1/D2 pathways confer normative advantages in sparse-reward RL; normative justification for dual-pathway design. `[dopamine, D1/D2, opponency, RL, striatum]` → [[jaskir_normative_nodate]]

**Naudé et al. (2024)** — DA builds latent attractors via plasticity and reveals them via excitability modulation; parallel to W_eff = f(DA)·W. `[dopamine, attractor, RNN, D1/D2]` → [[naude_dopamine_2024]]

---

## Dopamine dynamics & credit assignment

**Hamid et al. (2021)** — DA waves propagate DMS→DLS or reverse depending on agency; graded spatiotemporal credit assignment to functionally specialized subregions. `[dopamine, DMS, DLS, credit-assignment]` → [[hamid_wave-like_2021]]

**Lloyd & Dayan (2023)** — DA in NAc implements cognitive reframing of Pavlovian-instrumental conflicts; DA as instrument of behavioral control, not just teaching signal. `[dopamine, striatum, APE, dual-system]` → [[lloyd_reframing_2023]]

**Lloyd & Dayan (2015)** — Three non-exclusive accounts of phasic DA ramps reconciling TD errors with empirical observations. `[dopamine, RL, credit-assignment]` → [[lloyd_tamping_2015]]

**Westbrook et al. (2025)** — DA simultaneously enhances fast WM deployment and slow RL while blunting effort-cost discounting; dissociable effects via synthesis capacity and methylphenidate. `[dopamine, working-memory, RL, striatum]` → [[westbrook_striatal_2025]]

**Findling et al. (2025)** — mPFC neural variability implements near-optimal adaptive behavior in volatile environments via belief-proportional stochastic fluctuations. `[mPFC, working-memory, CTRNN]` → [[findling_neural_2025]]

---

## Background / wider context (no citekey notes)


**Schultz** — Canonical RPE: DA as TD prediction error in primate experiments; foundational. `[dopamine, RPE, RL]`

**Lee et al. (2014)** — Neural arbitration between model-based and model-free RL via reliability signals in lateral PFC. `[RL, dual-system, RPE]`

**Seamans & Yang (2004)** — Review of D1/D2 mechanisms in PFC: 18 key features, two-state model, bell-shaped dose-response. `[dopamine, D1/D2, mPFC]`

**Gläscher et al. (2010)** — SPE vs RPE dissociable in fMRI; model-based (IPS/lPFC) vs model-free (ventral striatum) neural signatures. `[RL, dual-system, RPE]`

**Engel et al. (2024)** — VTA vs SNC DA neurons drive anatomically distinct and functionally segregated striatal signals during learning. `[dopamine, DMS, DLS, credit-assignment]`

**Smith & Graybiel (2013)** — Task-bracketing patterns in DLS + infralimbic cortex emerge as dual operator of habit crystallization. `[habit, DLS, mPFC]`

**Doll et al. (2016)** — Genetic dissociation: COMT (PFC DA) predicts model-based, DARPP-32 (striatal DA) predicts model-free learning. `[dopamine, D1/D2, dual-system, RL]`

**Cools & D'Esposito (2011)** — Inverted-U relationship between dopamine and working memory/cognitive control; too-low or too-high DA impairs performance; D1 vs D2 receptor balance is key. `[dopamine, D1/D2, working-memory, mPFC, inverted-U]` → [[cools_inverted-ushaped_2011]]

**Cools (2019)** — Review: dopamine as a multi-functional neuromodulator shaping cognitive flexibility, working memory, and motivation across prefrontal and striatal circuits. `[dopamine, D1/D2, mPFC, striatum, cognitive-control]` → [[cools_chemistry_2019]]
