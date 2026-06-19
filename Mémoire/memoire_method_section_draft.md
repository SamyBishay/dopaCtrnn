# Mémoire: Method Section

## 2.1 Overview and rationale

This study is an *in silico* reproduction: no new human or animal data were collected. We ask whether the reversible cortico-striatal handoff reported by Villet et al. (2025) — the migration of control from a goal-directed to a habitual system over overtraining, and its instant reversal when the habitual system is silenced — can *emerge* in a two-area recurrent network from three biologically grounded asymmetries rather than from an externally imposed schedule: (a) a split in sensory access between the two areas, (b) a value-free learning rule in the habitual area, and (c) dopamine-gated *expression* of intact learned weights. Throughout, "emergence" is treated strictly: no term in the model specifies when control should transfer, so any transfer must be a consequence of the learning dynamics. This claim is subjected to an explicit falsification test (Section 2.5.3). There are no participants in the human-subjects sense; the units of analysis are independently initialised network instances (random seeds), and the number of seeds is reported with each result.

## 2.2 Task: delayed non-match-to-place T-maze

We simulate the delayed non-match-to-place (DNMTP) T-maze used by Villet et al. (2025). Each trial comprises three phases. In the **sample** phase, one arm is physically blocked and the agent is forced into the open arm, where it receives reward. In the **delay** phase, the agent is held in the start box for a fixed number of timesteps (`DELAY_STEPS`) with no external cue, so that correct performance requires holding "which arm was sampled" in working memory across the gap. In the **test** phase, both arms are open and reward is delivered for entering the arm that was *blocked* during the sample phase (the non-match). Movement and waiting incur small step costs, which pressure the agent toward efficient trajectories. To preserve a genuine working-memory demand analogous to Villet's ~90 s delay, all reported results use a non-trivial delay (`DELAY_STEPS` ≥ 10); a zero-delay setting is used only for debugging. The task is held fixed across the entire study, so that any change in behaviour reflects learning and control dynamics rather than a change in task structure.

## 2.3 Agent architecture

The agent is a dual-system model composed of two continuous-time recurrent neural networks (CTRNNs; Beer, 1995) coupled at the output stage. Each network integrates its inputs with leaky dynamics of the form τ (dh/dt) = −h + ϕ(W h + W_in x), with learnable time constants τ and a pointwise nonlinearity ϕ. The two networks differ deliberately along three dimensions — sensory access, learning rule, and dopaminergic profile — instantiating the three asymmetries of Section 2.1.

### 2.3.1 Goal-directed system (mPFC / DMS)

The goal-directed system receives a 22-dimensional **allocentric** observation (an allocentric position map, a blocked-arm indicator, and a phase indicator), giving it the spatial information required to represent and apply the non-match rule. It contains two populations distinguished by dopamine affinity: a D1-like, phasic-DA-sensitive population and a D2-like, tonic-DA-sensitive population, each with learnable time constants and gains. This phasic/tonic affinity mapping follows the classical account of dopamine release dynamics and differential receptor engagement (Grace, 1991; Dreyer et al., 2010), and supplies the multi-timescale machinery that lets a single dopamine variable drive both fast and slow effects. The system is trained by advantage actor–critic (A2C; Mnih et al., 2016) with an additional dopamine-recruitment penalty (`DA_COST_LAMBDA · da_recruit²`) that pressures it to minimise its own dopamine demand. Besides action logits, it emits a scalar **DA-request** signal that drives recruitment.

> **[OPEN — DA-request training signal, unresolved]** As implemented, the DA-request signal is optimised within the A2C objective (i.e., it is shaped by a reward-based advantage) subject to the recruitment penalty. This couples the handoff to reward and carries a circularity risk relative to the value-free/independence claims made for the habitual system. The alternative is to train the DA-request from a purely local prediction-error signal (consistent with the local-PE thesis), which would require specifying that error. **This is a design decision still to be made; the Methods text above describes the current implementation and does not commit to it.**

### 2.3.2 Habitual system (DLS)

The habitual system receives a 14-dimensional **egocentric** observation (wall/proximity sensors, local landmarks, and the previous action) and, by construction, **no** allocentric map. It therefore cannot plan the non-match rule and must instead acquire a stimulus–response policy. It comprises two opponent pathways, Go (D1-like) and NoGo (D2-like), combined as `action = Go − NoGo`, following the opponent-actor architecture of OpAL\* (Jaskir & Frank, 2023). Crucially, it is trained **value-free**: its teaching signal is an action prediction error — the mismatch between the expected and the executed action (Greenstreet et al., 2025) — together with a small efficiency term, and reward never enters its objective. This is the mechanistic source of its devaluation-insensitivity: a controller that never represented reward cannot, in principle, change its behaviour when reward is devalued.

The allocentric/egocentric split is a deliberate modelling assumption rather than an established anatomical fact. It is motivated by the broad association of dorsomedial/prefrontal circuits with flexible, map-like representation and of dorsolateral striatum with egocentric stimulus–response control, and it is the assumption that makes the two systems functionally distinct in the model; it is flagged as such.

### 2.3.3 Dopamine-gated control: expression-level multiplicative gain

The core mechanism is **expression-level**, not weight-level. Dopamine sets a multiplicative gain on each system's *effective* weights,

  **W_eff = f(DA) · W**,

scaling how strongly the intact learned weights are expressed without altering the weights themselves. Because the learned solution is preserved under any dopamine level and is merely up- or down-expressed, a down-expressed controller can be restored instantly — this is the model's analogue of Villet's reactivation, and the reason the goal-directed solution can lie dormant rather than being erased or overwritten. At the policy level this gain appears as a mixing weight on the goal-directed output,

  **w_GD = σ(α · DA_total + bias)**,  combined logits = **w_GD · π_GD + (1 − w_GD) · π_H**,

where π_GD and π_H are the goal-directed and habitual policies. Because dopamine recruitment is penalised in the goal-directed objective, the goal-directed system is under continual pressure to lower DA_total; it can do so without sacrificing reward only once the habitual system's policy is competent. Lowering dopamine lowers w_GD, and control transfers to the habitual system. No term in the model schedules this transition — it is what we test in H2 and falsify in H5.

## 2.4 Training procedure

Stage 1 of the project is the anchor and the basis for all results reported here; later stages of the roadmap (Section 2.6.3 and the General Discussion) are presented as planned work. A single network is trained on the fixed DNMTP task under the rules of Section 2.3 until the learning criterion is met (H1), with the dopamine-gated handoff and all logged quantities recorded across training. The learning criterion is adapted from Villet et al. (2025): sustained performance at or above 80% correct (corrected from an earlier 90% figure to match the original study's criterion). Behavioural performance is assessed on held-out, frozen-policy test trials. To distinguish learned from architectural structure, the full pipeline is additionally run on untrained networks (Section 2.5.4). Unless otherwise noted, quantities are reported across multiple independently seeded networks (≥ 5 for the negative control).

## 2.5 Experimental design and manipulations

The task is fixed; manipulations are applied at evaluation to the trained network (and, for the control, to untrained networks). The principal factors are training **phase** (learning vs maintenance/overtrained) crossed with targeted **system manipulations**.

### 2.5.1 Reward devaluation

To reproduce Villet's pre-feeding devaluation, the reward the goal-directed critic learns from is set to approximately zero (or strongly reduced) for a block of evaluation trials, with the policy otherwise frozen. A value-sensitive controller (goal-directed) should reduce performance under devaluation, because the reason to act has been removed; a value-free controller (habitual) should be unaffected, never having represented reward. Devaluation is applied at a learning-phase checkpoint and at a maintenance-phase checkpoint. (An earlier "proxy" devaluation — comparing goal-directed-only vs habitual-only accuracy without altering reward — is retained only as a secondary sanity check and is labelled as a proxy wherever reported, since it does not manipulate value and therefore cannot demonstrate a value-driven behavioural change.)

### 2.5.2 Area silencing (lesions)

The mPFC system, the DLS system, or both are silenced at evaluation by zeroing the corresponding hidden activity (`h_gd` / `h_hab`), crossed with phase. This reproduces Villet's chemogenetic inhibition conditions in simulation.

### 2.5.3 Reactivation probe and emergence falsification

This is the decisive test of the mechanism. In the maintenance phase, the habitual system is silenced, and we measure, before vs after silencing: (a) accuracy, (b) the goal-directed DA-request signal, and (c) devaluation-sensitivity. The expression-level mechanism predicts that accuracy is preserved, the DA-request rises back up, and devaluation-sensitivity returns — i.e., the dormant goal-directed solution re-expresses. This is treated as a falsification: if the DA-request does **not** rise when the habitual system is removed, then the handoff tracked episode count rather than habitual competence, the mechanism is a schedule in disguise, and the result is reported honestly as a negative finding. This test is what separates a genuine reversible handoff from a hard-coded one.

### 2.5.4 Untrained negative control

The handoff timecourse and the working-memory analysis are re-run on untrained networks (random weights, ≥ 5 seeds). The mechanism predicts chance accuracy, no decodable delay representation, and no handoff, confirming that the structure observed in the trained network is learned rather than architectural.

## 2.6 Analysis pipeline

The analysis pipeline is built once and applied unchanged across all conditions and (in later stages) all roadmap variants.

### 2.6.1 Behavioural readouts

The primary behavioural measure is proportion correct on frozen-policy test trials. In addition we record "solo" evaluations of each system in isolation, and we log the control weight w_GD and total dopamine recruitment across training; together these constitute the emergent-handoff timecourse used to test H2.

### 2.6.2 Working-memory representation

Delay-period hidden states are projected with principal component analysis (PCA) and visualised with one point per trial, coloured by the sampled arm. A linear decoder (logistic regression or LDA on the leading principal components) then reads arm identity from delay activity, with cross-validated accuracy compared against the 50% chance level. Visible cluster separation plus above-chance decoding constitutes evidence that the network maintains a readable, trial-type-specific working-memory representation across the delay.

### 2.6.3 Optional dynamical characterisation (Annex / future stages)

Where a finer characterisation is warranted, candidate fixed points are located by numerically minimising ‖dh/dt‖² from delay-state seeds and classified by their Jacobian eigenvalues (stable iff all Re(λ) < 0), following the dynamical-systems approach to recurrent computation (Sussillo & Barak, 2013; Mante et al., 2013); representational dimensionality is summarised by the participation ratio with bootstrap confidence intervals. These analyses establish whether the delay representation is a genuine attractor rather than a slow transient, and how low-dimensional it is; they refine but do not determine whether the memory exists, and are therefore reported in the Annex only. Adopting a low-rank formulation of the habitual area (a natural Stage-2 manipulation) would read dimensionality directly off the rank.

## 2.7 Operational hypotheses and statistical analyses

Statistical threshold α = .05; tests are one-tailed where theory fixes the direction. Results are reported in APA format. Each hypothesis names the figure it feeds.

1. **H1 — Learning (prerequisite gate; Fig. 1).** The trained network reaches ≥ 80% correct at a non-trivial delay. Readout: proportion correct over N = 1000 frozen-policy test trials. Test: one-sample binomial against p₀ = .50, reporting the proportion, exact p, and 95% CI. Pass: ≥ .80, p < .001.

2. **H2 — Emergent, not scheduled, handoff (central hypothesis; Fig. 1).** Over training, w_GD and dopamine recruitment fall while combined accuracy is preserved, and the fall is temporally locked to the rise in habitual-solo accuracy (beginning after habitual competence crosses threshold, not at a fixed episode). Readout: the four logged curves. The lock is quantified by the cross-correlation/lag between the habitual-solo-accuracy rise and the w_GD fall, and by the across-seed spread of the handover-onset episode — an emergent handoff should track each seed's learning speed, whereas a schedule would fire at the same episode every seed. Pass: w_GD and DA drop by a large margin (e.g., ≥ 50% from peak) *after* habitual-solo accuracy crosses ~ .80, with combined accuracy maintained.

3. **H3 — Devaluation dissociation (Fig. 2).** Devaluation lowers accuracy in the learning phase (goal-directed control) but not in the maintenance phase (habitual control). Test: a phase × devaluation interaction, with per-phase before/after comparisons. Pass: a significant drop in learning and no significant drop in maintenance.

4. **H4 — Lesion × phase dissociation (Fig. 3).** Silencing mPFC breaks learning but not maintenance; silencing DLS leaves maintenance accuracy intact; silencing both collapses performance. Readout: accuracy with `h_gd`/`h_hab` zeroed at evaluation, per phase. Pass: reproduces the Villet inhibition pattern.

5. **H5 — Reactivation falsification (rigour anchor; Fig. 4).** In the maintenance phase, silencing the habitual system (a) preserves accuracy, (b) makes the goal-directed DA-request rise back up, and (c) restores devaluation-sensitivity. Readout: goal-directed DA-request and devaluation-sensitivity, before vs after silencing the habitual system. Pass: accuracy preserved, DA-request rises, devaluation-sensitivity returns. Fail (reported honestly): the DA-request does not change when the habitual system is removed — the handoff was a function of episode count, and the model does not reproduce Villet's reactivation.

6. **H6 — Working-memory representation (Fig. 5).** During the delay, the controlling system's hidden state separates into trial-type-specific regions (one per sampled arm). Readout: PCA scatter of delay states coloured by arm, plus a linear decoder's cross-validated accuracy. Pass: visible separation and decoder accuracy ≥ 90%, p < .001. Fail: no separation — reported descriptively, leaning on the PCA picture.

7. **H7 — Untrained negative control (Fig. 6).** Untrained networks (≥ 5 seeds) show chance accuracy, no decodable delay representation, and no handoff, confirming that Figs. 1 and 5 reflect learning rather than architecture.

The minimum sufficient result for the mémoire is H1, H2, and H7 together with the simple H6 representation; H3, H4, and H5 elevate the claim from "a handoff occurs" to "this is Villet's handoff," with H5 making the claim falsifiable.

## 2.8 Ethics and data statement

This work is purely computational; no new human or animal data were collected. The biological paradigm reproduced in simulation is taken from published work (Villet et al., 2025), conducted under that study's institutional approvals.

## 2.9 Software and reproducibility

The model, environment, training loop, and analyses are implemented in a single codebase (`model.py`, `environment.py`, `train.py`, `params.py`, `visualize.py`). All reported results use fixed random seeds; code and configuration are provided in the Annex to permit replication.
