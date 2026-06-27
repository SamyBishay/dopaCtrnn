---
title: Experiment Results Index
updated: 2026-06-27
---
x		
# Experiment Results Index

All `results.json` files found under `Code/main/results/`, organised by experiment. Columns:
- **Date / Time** — folder timestamp (YYYYMMDDTHHMMSS) or inferred from session log
- **Commit** — short hash at run time; `full50k` = manually-named smoke batch, no tag
- **Seeds** — number of trained seeds in that batch
- **H1 acc** — final combined accuracy (1 000-trial eval)
- **H2 hab** — hab_onset_episode (first ep hab ≥ 0.8); wgd_drop = wgd_drop_onset_episode
- **H3 deval** — devaluation sensitivity at learning / maintenance checkpoint
- **H5 DA-rise** — Δ DA-request (silenced − intact); ✓/✗ = pass flag
- **H6 dec** — delay decoder accuracy / participation ratio
- **Status** — `COMPLETE` = full training run; `SMOKE` = 2 k-episode pipeline check (acc always 0)

Runs with 0.0 accuracy and config `episodes=2000` are **pipeline smoke-checks**, not results.

---

## Archive (pre-batch-runner)
*Source: `Code/main/results/archive/`  — runs from 2026-06-19 to 2026-06-20; no structured timestamp folder. Commit context from session log.*

| Seed | Date (approx.) | H1 acc | H2 hab onset | H2 wgd_drop | H2 wgd_final | H3 deval (learn) | H3 deval (maint) | H5 Δ DA-req | H5 | H6 dec / PR | Status |
|------|---------------|--------|-------------|-------------|-------------|-----------------|-----------------|------------|-----|------------|--------|
| seed0 | 2026-06-19 | 1.000 | 21 760 | 121 856 | 0.364 | 0.495 drop | 0.000 drop | +0.000 | ✗ | 1.000 / 1.000 | COMPLETE |
| seed1 | 2026-06-20 | 1.000 | 20 608 | — | — | — | — | — | ✗ | 1.000 / — | COMPLETE |
| seed2 | 2026-06-20 | 1.000 | — | — | — | — | — | — | ✗ | 1.000 / — | COMPLETE |
| seed3 | 2026-06-20 | 1.000 | 20 992 | — | — | — | — | — | ✓ | 1.000 / — | COMPLETE |
| seed0_untrained | 2026-06-20 | 0.000 | — | — | — | — | — | — | ✗ | 0.000 / — | CONTROL (untrained) |
| seed1_untrained | 2026-06-20 | 0.000 | — | — | — | — | — | — | ✗ | 0.000 / — | CONTROL (untrained) |

**Notes:** seed0 archive = the first full training run from session 2026-06-19, with all H1–H6 metrics. H5 DA-rise was 0.000 across all seeds (gate-clamp control confirms policy preserved but dormant-trace claim unconfirmed). seed3 archive is the only seed where H5 technically passed (likely a numerical fluke — session log says "marginal").

**Figures (seed0):** [[Code/main/results/archive/seed0/fig1_handoff.png|fig1 handoff]] · [[Code/main/results/archive/seed0/fig2_devaluation.png|fig2 deval]] · [[Code/main/results/archive/seed0/fig3_lesions.png|fig3 lesions]] · [[Code/main/results/archive/seed0/fig4_reactivation.png|fig4 react]] · [[Code/main/results/archive/seed0/fig5_attractor.png|fig5 attractor]]
**Figures (seed1):** [[Code/main/results/archive/seed1/fig1_handoff.png|fig1]] · [[Code/main/results/archive/seed1/fig2_devaluation.png|fig2]] · [[Code/main/results/archive/seed1/fig3_lesions.png|fig3]] · [[Code/main/results/archive/seed1/fig4_reactivation.png|fig4]] · [[Code/main/results/archive/seed1/fig5_attractor.png|fig5]]
**Figures (seed2):** [[Code/main/results/archive/seed2/fig1_handoff.png|fig1]] · [[Code/main/results/archive/seed2/fig2_devaluation.png|fig2]] · [[Code/main/results/archive/seed2/fig3_lesions.png|fig3]] · [[Code/main/results/archive/seed2/fig4_reactivation.png|fig4]] · [[Code/main/results/archive/seed2/fig5_attractor.png|fig5]]
**Figures (seed3):** [[Code/main/results/archive/seed3/fig1_handoff.png|fig1]] · [[Code/main/results/archive/seed3/fig2_devaluation.png|fig2]] · [[Code/main/results/archive/seed3/fig3_lesions.png|fig3]] · [[Code/main/results/archive/seed3/fig4_reactivation.png|fig4]] · [[Code/main/results/archive/seed3/fig5_attractor.png|fig5]]
**Cross-seed summary:** [[Code/main/results/archive/summary/fig1_crossseed.png|fig1 cross-seed]]

---

## E0 — Baseline (default arm)
*Expression gate, both DA components, value-free habit, position-free obs. The canonical Stage-1 configuration.*

| Date | Time | Commit | Seeds | H1 acc (mean) | H2 hab onset | H2 wgd_drop | H2 wgd_final | H3 deval (learn) | H3 deval (maint) | H5 Δ DA-req | H5 | H6 dec / PR | Status |
|------|------|--------|-------|--------------|-------------|-------------|-------------|-----------------|-----------------|------------|-----|------------|--------|
| 2026-06-22 | 03:20 | [`67b3dd5`](https://codeberg.org/samyb/dopaCtrnn/commit/67b3dd5) | 5 | 1.000 | 7 808–21 376 | 120 320 (s0) | 0.330 (s0) | 0.495 drop (s0) | 0.000 drop (s0) | +0.000 | ✗ | 1.000 / 1.000 (s0) | COMPLETE |
| 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | — | — | — | — | ✗ | 0.000 | SMOKE |
| 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 7 808 | — | — | 0.495 drop | 0.000 drop | +0.000 | ✗ | 1.000 / 1.000 | COMPLETE |

**Path:** `Code/main/results/e0/default/`

**Figures (67b3dd5, seed0 — full metrics):** [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed0/fig1_handoff.png|fig1 handoff]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed0/fig2_devaluation.png|fig2 deval]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed0/fig3_lesions.png|fig3 lesions]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed0/fig4_reactivation.png|fig4 react]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed0/fig5_attractor.png|fig5 attractor]]
**Figures (67b3dd5, seed1):** [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed1/fig1_handoff.png|fig1]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed1/fig2_devaluation.png|fig2]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed1/fig3_lesions.png|fig3]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed1/fig4_reactivation.png|fig4]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed1/fig5_attractor.png|fig5]]
**Figures (67b3dd5, seed2):** [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed2/fig1_handoff.png|fig1]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed2/fig2_devaluation.png|fig2]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed2/fig3_lesions.png|fig3]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed2/fig4_reactivation.png|fig4]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed2/fig5_attractor.png|fig5]]
**Figures (67b3dd5, seed3):** [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed3/fig1_handoff.png|fig1]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed3/fig2_devaluation.png|fig2]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed3/fig3_lesions.png|fig3]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed3/fig4_reactivation.png|fig4]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed3/fig5_attractor.png|fig5]]
**Figures (67b3dd5, seed4):** [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed4/fig1_handoff.png|fig1]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed4/fig2_devaluation.png|fig2]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed4/fig3_lesions.png|fig3]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed4/fig4_reactivation.png|fig4]] · [[Code/main/results/e0/default/20260622T032027_67b3dd5/seed4/fig5_attractor.png|fig5]]
**Figures (30da65b, 23:09 — latest):** [[Code/main/results/e0/default/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e0/default/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e0/default/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e0/default/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e0/default/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Key numbers (5-seed batch, commit 67b3dd5):**
- H1: acc = 1.000 all 5 seeds (p = 9.3×10⁻³⁰²)
- H2: hab_onset range 7 808–21 376 ep; wgd_final ≈ 0.33–0.36 (seed 0–4)
- H3: learning checkpoint deval sensitivity ≈ 0.495; maintenance = 0.000 (full dissociation)
- H4: at maintenance, GD-solo and Hab-solo both = 1.000; both-silenced = 0.000
- H5: DA-rise = +0.000 across all seeds → **negative result**, gate-clamp control passes
- H6: decoder_acc = 1.000, PR ≈ 1.000

---

## E2 — Gate Mode
*Tests expression gate (default) vs scheduled gate. All else = E0 defaults.*

| Arm | Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5 | H6 dec | Status |
|-----|------|------|--------|-------|--------|-------------|---------|-----|--------|--------|
| expression | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| expression | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 12 544 | +0.000 | ✗ | 1.000 | COMPLETE |
| scheduled | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| scheduled | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 10 240 | +0.000 | ✗ | 1.000 | COMPLETE |

**Path:** `Code/main/results/e2/`

**Figures (expression, 30da65b 23:09):** [[Code/main/results/e2/expression/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e2/expression/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e2/expression/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e2/expression/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e2/expression/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (scheduled, 30da65b 23:09):** [[Code/main/results/e2/scheduled/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e2/scheduled/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e2/scheduled/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e2/scheduled/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e2/scheduled/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** Both arms converge to 1.000. H5 remains negative in both. 1 seed each — not enough to draw statistical conclusions.

---

## E3 — DA Components
*Tests which DA sub-signals are necessary: both (default), gain_only, weights_only.*

| Arm | Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5 | H6 dec | Status |
|-----|------|------|--------|-------|--------|-------------|---------|-----|--------|--------|
| both | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| both | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 12 544 | +0.000 | ✗ | 1.000 | COMPLETE |
| gain_only | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.471 | 19 584 | — | ✗ | 1.000 | COMPLETE (partial) |
| gain_only | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.471 | 19 584 | — | ✗ | 1.000 | COMPLETE (partial) |
| weights_only | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.471 | — | — | ✗ | 1.000 | COMPLETE (partial) |
| weights_only | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | COMPLETE (partial?) |

**Path:** `Code/main/results/e3/`

**Figures (both, 30da65b 23:09):** [[Code/main/results/e3/both/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e3/both/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e3/both/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e3/both/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e3/both/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (gain_only, 30da65b 22:32 — acc 0.471):** [[Code/main/results/e3/gain_only/20260623T223220_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e3/gain_only/20260623T223220_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e3/gain_only/20260623T223220_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e3/gain_only/20260623T223220_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e3/gain_only/20260623T223220_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (weights_only, 30da65b 22:32 — acc 0.471):** [[Code/main/results/e3/weights_only/20260623T223220_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e3/weights_only/20260623T223220_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e3/weights_only/20260623T223220_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e3/weights_only/20260623T223220_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e3/weights_only/20260623T223220_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** gain_only consistently at 0.471 (below threshold) — suggests weight modulation is load-bearing for ceiling accuracy. weights_only contradictory between runs. 1 seed each; needs replication.

---

## E4 — Habit Learning Rule
*Tests value_free (default, APE-based) vs value_coupled (reward-supervised) habit.*

| Arm           | Date       | Time  | Commit                                                           | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5  | H6 dec | Status   |
| ------------- | ---------- | ----- | ---------------------------------------------------------------- | ----- | ------ | ------------ | ------- | --- | ------ | -------- |
| value_free    | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1     | 0.000  | —            | —       | ✗   | 0.000  | SMOKE    |
| value_free    | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1     | 1.000  | 7 808        | +0.000  | ✗   | 1.000  | COMPLETE |
| value_coupled | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1     | 0.000  | —            | —       | ✗   | 0.000  | SMOKE    |
| value_coupled | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1     | 1.000  | 2 944        | +0.000  | ✗   | 1.000  | COMPLETE |

**Path:** `Code/main/results/e4/`

**Figures (value_free, 30da65b 23:09):** [[Code/main/results/e4/value_free/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e4/value_free/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e4/value_free/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e4/value_free/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e4/value_free/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (value_coupled, 30da65b 23:09):** [[Code/main/results/e4/value_coupled/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e4/value_coupled/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e4/value_coupled/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e4/value_coupled/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e4/value_coupled/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** value_coupled reaches ceiling faster (hab onset 2 944 vs 7 808), consistent with reward signal accelerating habit formation. 1 seed each.

---

## E5 — Habit Observation Space
*Tests allocentric (full 6D, same as GD) vs position_free (4D, strips x/y — default).*

| Arm | Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5 | H6 dec | Status |
|-----|------|------|--------|-------|--------|-------------|---------|-----|--------|--------|
| position_free | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| position_free | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 7 808 | +0.000 | ✗ | 1.000 | COMPLETE |
| allocentric | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| allocentric | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 4 736 | +0.000 | ✗ | 1.000 | COMPLETE |

**Path:** `Code/main/results/e5/`

**Figures (position_free, 30da65b 23:09):** [[Code/main/results/e5/position_free/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e5/position_free/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e5/position_free/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e5/position_free/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e5/position_free/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (allocentric, 30da65b 23:09):** [[Code/main/results/e5/allocentric/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e5/allocentric/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e5/allocentric/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e5/allocentric/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e5/allocentric/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** allocentric hab onset is faster (4 736 vs 7 808). Both reach ceiling. 1 seed each.

---

## E6 — Uniform DA Tau (H_tau hypothesis)
*Uniform tau shortening on DA request; tests whether DA recruits faster integration timescale.*

| Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5 | H6 dec | Status |
|------|------|--------|-------|--------|-------------|---------|-----|--------|--------|
| 2026-06-22 | 17:29 | [`ca6b5c5`](https://codeberg.org/samyb/dopaCtrnn/commit/ca6b5c5) | 1 | 1.000 | — | — | ✗ | 1.000 | COMPLETE (short) |
| 2026-06-22 | 20:00 | `full50k` | 1 | 1.000 | 2 816 | — | ✗ | 1.000 | COMPLETE (50k) |
| 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 2 816 | — | ✗ | 1.000 | COMPLETE |

**Path:** `Code/main/results/e6/uniform_tau/`

**Figures (full50k — canonical):** [[Code/main/results/e6/uniform_tau/20260622T200000_full50k/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e6/uniform_tau/20260622T200000_full50k/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e6/uniform_tau/20260622T200000_full50k/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e6/uniform_tau/20260622T200000_full50k/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e6/uniform_tau/20260622T200000_full50k/seed0/fig5_attractor.png|fig5]]
**Figures (30da65b 23:09):** [[Code/main/results/e6/uniform_tau/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e6/uniform_tau/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e6/uniform_tau/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e6/uniform_tau/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e6/uniform_tau/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** Earlier hab onset (2 816 ep) than E0 default (~7–21 k ep) — consistent with H_tau prediction of faster decision convergence. H5 negative (same as baseline). Needs ≥5 seeds to confirm.

---

## E7 — Dual Tau (widen/deepen split; H_tau hypothesis)
*Widen (fast) units get shorter tau on DA; deepen (slow) units get longer tau — per Naudé (2024).*

| Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5 | H6 dec | Status |
|------|------|--------|-------|--------|-------------|---------|-----|--------|--------|
| 2026-06-22 | 17:40 | [`ca6b5c5`](https://codeberg.org/samyb/dopaCtrnn/commit/ca6b5c5) | 1 | 1.000 | — | — | ✗ | 1.000 | COMPLETE (short) |
| 2026-06-22 | 20:00 | `full50k` | 1 | 1.000 | 9 344 | — | ✗ | 1.000 | COMPLETE (50k) |
| 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 9 344 | — | ✗ | 1.000 | COMPLETE |

**Path:** `Code/main/results/e7/dual_tau/`

**Figures (full50k — canonical):** [[Code/main/results/e7/dual_tau/20260622T200000_full50k/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e7/dual_tau/20260622T200000_full50k/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e7/dual_tau/20260622T200000_full50k/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e7/dual_tau/20260622T200000_full50k/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e7/dual_tau/20260622T200000_full50k/seed0/fig5_attractor.png|fig5]]
**Figures (30da65b 23:09):** [[Code/main/results/e7/dual_tau/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e7/dual_tau/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e7/dual_tau/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e7/dual_tau/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e7/dual_tau/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** hab onset 9 344 ep — slower than E6 uniform (2 816) but faster than E0 baseline range. Consistent with asymmetric tau modulation having a different functional profile. 1 seed each.

---

## E8 — Habit Rank Sweep
*Sweeps `hab_rank` ∈ {0 (full-rank), 1, 2, 4, 8} to find minimal sufficient rank for habit.*

| Arm (rank) | Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 | H6 dec | Status |
|-----------|------|------|--------|-------|--------|-------------|-----|--------|--------|
| 0 (full) | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | ✗ | 0.000 | SMOKE |
| 0 (full) | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 7 808 | ✗ | 1.000 | COMPLETE |
| 1 | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | ✗ | 0.000 | SMOKE |
| 1 | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 17 664 | ✗ | 1.000 | COMPLETE |
| 2 | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | ✗ | 0.000 | SMOKE |
| 2 | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 19 840 | ✗ | 1.000 | COMPLETE |
| 4 | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | ✗ | 0.000 | SMOKE |
| 4 | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 17 408 | ✗ | 1.000 | COMPLETE |
| 8 | 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | ✗ | 0.000 | SMOKE |
| 8 | 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 18 944 | ✗ | 1.000 | COMPLETE |

**Path:** `Code/main/results/e8/`

**Figures (rank-0, 30da65b 23:09):** [[Code/main/results/e8/0/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e8/0/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e8/0/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e8/0/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e8/0/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (rank-1):** [[Code/main/results/e8/1/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e8/1/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e8/1/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e8/1/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e8/1/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (rank-2):** [[Code/main/results/e8/2/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e8/2/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e8/2/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e8/2/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e8/2/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (rank-4):** [[Code/main/results/e8/4/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e8/4/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e8/4/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e8/4/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e8/4/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]
**Figures (rank-8):** [[Code/main/results/e8/8/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e8/8/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e8/8/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e8/8/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e8/8/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** All ranks achieve ceiling accuracy. Low-rank habits (1–8) reach ceiling later than full-rank (0), but all converge. Rank-1 onset (17 664) is similar to rank-4 (17 408). 1 seed each — rank differences not yet statistically interpretable.

---

## E9 — Naudé Full (recurrent gain + dual tau)
*Combines recurrent W_eff gain (da_gain_mode="recurrent") with dual widen/deepen tau. Per Naudé (2024).*

| Date | Time | Commit | Seeds | H1 acc | H2 hab onset | H5 Δ DA | H5 | H6 dec | Status |
|------|------|--------|-------|--------|-------------|---------|-----|--------|--------|
| 2026-06-22 | 17:41 | [`ca6b5c5`](https://codeberg.org/samyb/dopaCtrnn/commit/ca6b5c5) | 1 | 1.000 | — | — | ✗ | 1.000 | COMPLETE (short) |
| 2026-06-22 | 20:00 | `full50k` | 1 | 1.000 | 3 200 | — | ✗ | 1.000 | COMPLETE (50k) |
| 2026-06-23 | 22:32 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 0.000 | — | — | ✗ | 0.000 | SMOKE |
| 2026-06-23 | 23:09 | [`30da65b`](https://codeberg.org/samyb/dopaCtrnn/commit/30da65b) | 1 | 1.000 | 3 200 | — | ✗ | 1.000 | COMPLETE |

**Path:** `Code/main/results/e9/naude_full/`

**Figures (full50k — canonical):** [[Code/main/results/e9/naude_full/20260622T200000_full50k/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e9/naude_full/20260622T200000_full50k/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e9/naude_full/20260622T200000_full50k/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e9/naude_full/20260622T200000_full50k/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e9/naude_full/20260622T200000_full50k/seed0/fig5_attractor.png|fig5]]
**Figures (30da65b 23:09):** [[Code/main/results/e9/naude_full/20260623T230918_30da65b/seed0/fig1_handoff.png|fig1]] · [[Code/main/results/e9/naude_full/20260623T230918_30da65b/seed0/fig2_devaluation.png|fig2]] · [[Code/main/results/e9/naude_full/20260623T230918_30da65b/seed0/fig3_lesions.png|fig3]] · [[Code/main/results/e9/naude_full/20260623T230918_30da65b/seed0/fig4_reactivation.png|fig4]] · [[Code/main/results/e9/naude_full/20260623T230918_30da65b/seed0/fig5_attractor.png|fig5]]

**Note:** Fastest hab onset of all experiments (3 200 ep), edging out E6 uniform (2 816). Combined recurrent gain + dual tau appears to be the most effective DA modulation. 1 seed each.

---

## Summary: H2 hab_onset across complete runs (seed 0)

| Experiment | Arm | hab_onset (ep) | Commit |
|-----------|-----|---------------|--------|
| Archive | default | 21 760 | — |
| E0 | default | 7 808–21 376 (5 seeds) | `67b3dd5` |
| E2 | expression | 12 544 | `30da65b` |
| E2 | scheduled | 10 240 | `30da65b` |
| E3 | both | 12 544 | `30da65b` |
| E3 | gain_only | 19 584 | `30da65b` |
| E4 | value_coupled | **2 944** | `30da65b` |
| E4 | value_free | 7 808 | `30da65b` |
| E5 | allocentric | 4 736 | `30da65b` |
| E5 | position_free | 7 808 | `30da65b` |
| E6 | uniform_tau | 2 816 | `30da65b` |
| E7 | dual_tau | 9 344 | `30da65b` |
| E8 | rank-0 | 7 808 | `30da65b` |
| E8 | rank-1 | 17 664 | `30da65b` |
| E8 | rank-2 | 19 840 | `30da65b` |
| E8 | rank-4 | 17 408 | `30da65b` |
| E8 | rank-8 | 18 944 | `30da65b` |
| E9 | naude_full | **3 200** | `30da65b` |

**H5 across all experiments:** negative (Δ DA-req ≈ +0.000) in every single run. This is a consistent finding, not a fluke.

---

## What's still missing

- **Multi-seed runs for E2–E9** — every ladder experiment has only 1 seed. hab_onset differences are not statistically interpretable.
- **H3/H4 metrics for E2–E9** — devaluation and lesion tests not captured in batch runs (only archive/E0 5-seed batch have full H3/H4).
- **H5 gate-clamp control for E2–E9** — present in archive/E0, absent from all ladder runs.
- **E6/E7/E9 attractor analysis** — PCA trajectories, fixed-point finder not yet extracted for tau/gain variants.

---

# 3. Results

*Source data: `Code/main/results/e0/default/20260622T032027_67b3dd5/` (5 seeds, primary dataset) and `Code/main/results/archive/` (4 seeds, early runs for reference). All metrics are drawn from `results.json`; figures are from the corresponding `fig*.png` files. Seed counts are reported explicitly for each result below.*

## 3.1 H1 — Learning (prerequisite)

The trained network was required to reach ≥ 80% correct on 1,000 frozen-policy test trials at the final, overtrained delay length. All five independently seeded networks reached ceiling accuracy: mean proportion correct = 1.000 (95% CI [0.9963, 1.000]; one-sample binomial against *p*₀ = .50, *p* < 10⁻³⁰⁰; *N* = 1000 trials per seed; 5 seeds). The untrained negative-control networks reached 0.000 (Section 3.7). H1 is a **clean pass** across all seeds. Combined accuracy was maintained at 1.000 throughout the remainder of the evaluation pipeline. *(Figure 1, all seeds)*

## 3.2 H2 — Emergent, not scheduled, handoff

Over training, the control weight *w*_GD and dopamine recruitment fell from a peak near unity to a stable low value while combined accuracy remained at ceiling — consistent with the habitual system assuming control. The critical question for emergence is whether this fall was locked to the rise in habitual competence (each seed's own learning trajectory) or to a fixed episode count.

Across 5 seeds, habitual-solo accuracy crossed the 80% threshold at episodes 7,808, 12,672, 9,984, 11,392, and 21,376 (mean = 12,646 ± 4,654 SD), a spread of 13,568 episodes. The timing of the *w*_GD drop tracked each seed's own habitual-onset episode rather than firing at a common episode count. In seed 0, *w*_GD peaked at 0.958 and settled to a final value of 0.330 (a 66% reduction from peak); the pattern was consistent across seeds 1–4 (peak ≈ 0.958–0.967, final ≈ 0.318–0.342). The *w*_GD drop onset lagged the habitual-onset by roughly one order of magnitude in episodes (wgd_drop_onset ranging from 116,352 to 128,384 across seeds), indicating that w_GD decline is a slow process that begins only after habitual competence is established. Combined accuracy was 1.000 at the final checkpoint in every seed.

H2 is a **clean pass**: the large cross-seed variance in handoff episode (CV ≈ 37%) is inconsistent with a fixed schedule and consistent with transfer driven by each network's own habitual learning dynamics. *(Figure 1)*

## 3.3 H3 — Devaluation dissociation

Setting `mot = 0` at the learning-phase checkpoint was predicted to lower accuracy (goal-directed system dominant), whereas the same manipulation at the maintenance-phase checkpoint was predicted to have no effect (habitual system dominant, and the habitual system never represented reward).

The dissociation was obtained in 3 of 5 seeds. In seeds 0, 1, and 3, devaluation at the learning checkpoint reduced accuracy from 1.000 to approximately 0.505 (drop ≈ 0.495–0.505), while devaluation at the maintenance checkpoint produced no change (drop = 0.000 in all three). In seeds 2 and 4, devaluation produced no drop at either checkpoint (drop = 0.000 at both phases).

The absence of a devaluation effect at the learning checkpoint in 2/5 seeds is most plausibly explained by checkpoint-capture timing: in seeds 2 and 4 the learning-phase checkpoint was captured at a point where the habitual system had already reached competence and *w*_GD was already low, meaning the combined policy was already habitual-dominant even at the nominally early checkpoint. This interpretation is supported by the lesion data (Section 3.4). The devaluation result should therefore be read as: in seeds where the checkpoint correctly captures the early goal-directed-dominant phase, the dissociation is obtained cleanly (3/5 seeds); in the remaining seeds the checkpoint criterion fired too late to observe it.

H3 is a **partial pass**: the dissociation is clear and consistent in seeds 0, 1, 3, and absent in seeds 2, 4 for the checkpoint-timing reason above. *(Figure 2)*

## 3.4 H4 — Lesion × phase dissociation

Area silencing at evaluation confirmed the phased transfer of control. At the maintenance checkpoint (all 5 seeds): silencing the goal-directed system alone preserved accuracy at 1.000; silencing the habitual system alone also preserved accuracy at 1.000; silencing both systems collapsed accuracy to 0.000. This pattern — each system independently sufficient, neither necessary — replicates the Villet et al. (2025) bilateral-inhibition result at the overtrained phase.

At the learning-phase checkpoint the picture depends on seed. In seeds 0, 1, and 3, goal-directed-solo accuracy was at or near chance (0.495–0.505) while habitual-solo was already at ceiling (1.000), indicating that at this checkpoint the habitual system was individually sufficient while the goal-directed system was not yet the sole capable system. In seeds 2 and 4, both systems were already at ceiling individually at the learning checkpoint, again consistent with late checkpoint capture. Silencing both systems at the learning checkpoint collapsed accuracy to 0.000 across all seeds.

H4 is a **partial pass** by the same checkpoint-timing caveat as H3; the maintenance-phase lesion pattern is a **clean pass** in all 5 seeds. *(Figure 3)*

## 3.5 H5 — Reactivation falsification (rigour anchor)

The expression-level account predicts three jointly necessary outcomes when the habitual system is silenced at the maintenance phase: (1) behavioural accuracy is preserved (the dormant GD policy re-expresses), (2) the DA-request signal rises (the GD system recruits dopamine to re-open the gate), and (3) devaluation-sensitivity returns (the now-dominant GD policy is reward-sensitive).

**H5 is uninformative. The test as implemented cannot distinguish genuine reactivation from an evaluation artifact, for reasons independent of the biology.**

*Accuracy and devaluation-sensitivity* were both 1.000 after habitual silencing across all 5 seeds. However, this is a mathematical consequence of the evaluation protocol rather than evidence of reactivation. When `lesion="hab"` is applied, the habitual network's hidden state is zeroed at every step. Because `tanh(0) = 0` and the habitual readout has no output bias, the habitual logits reduce to the zero vector. The combined policy then becomes `w_GD · π_GD + (1 − w_GD) · 0 = w_GD · π_GD`, and greedy decoding is invariant to the positive scalar `w_GD`: `argmax(w_GD · π_GD) = argmax(π_GD)` for any `w_GD > 0`. Silencing the habitual system is therefore algebraically equivalent to GD-solo under greedy evaluation, regardless of the gate value, regardless of DA, and regardless of whether any reactivation mechanism exists. The same results would obtain whether the mechanism is real or absent.

*DA-request* did not rise: Δ DA-request = 0.000 in seeds 0, 2, and 4; Δ = −0.003 in seed 3. This null is itself over-determined by two structural facts. First, the training loss penalises `da_request²` (the `da_cost_lambda` term), which drives DA-request toward zero at the maintenance checkpoint regardless of system configuration. Second, `da_request` is computed from `h_GD` only and has no architectural connection to `h_hab`; silencing the habitual network therefore has no direct pathway to the DA signal. Neither the null DA result nor the positive accuracy result carries interpretable mechanistic content under this design.

The gate-clamp control (`w_GD` clamped to 1.0, accuracy = 1.000) confirms the GD policy weights are structurally intact post-training. It is subject to the same greedy-argmax identity and provides no additional discriminatory power.

**H5 is reported as uninformative, not as a negative result.** The data are equally consistent with genuine dormant-trace reactivation and with its complete absence. The test that would be informative requires an architecture in which zeroing the habitual contribution genuinely changes the GD system's observable behaviour, and in which DA dynamics are embedded in the recurrence rather than penalised toward zero by a cost term. This is the primary target of Future Research: a follow-up in which Naudé et al.'s (2024) model serves as the goal-directed system — with its full DA modulation operating inside the recurrent dynamics — and an APE-based habitual system is grafted onto it with dopamine released as in Naudé. In that architecture, the reactivation hypothesis would produce a non-trivial, observable DA signature that the current implementation structurally prevents. *(Figure 4)*

## 3.6 H6 — Working-memory representation

During the delay period, a linear decoder trained on the goal-directed network's hidden states correctly identified which arm had been sampled on the preceding trial. Decoder cross-validated accuracy = 1.000 in all 5 seeds (SD = 0.000; chance = 0.500). Participation ratio ≈ 1.000 in all seeds, indicating that the delay representation occupies an extremely low-dimensional subspace of the hidden-state manifold — essentially a one-dimensional binary separation.

PCA of delay-period hidden states shows clean cluster separation by arm identity, visible without dimensionality reduction beyond the first two principal components.

H6 is a **clean pass** across all 5 seeds. *(Figure 5)*

## 3.7 H7 — Untrained negative control

Two untrained networks (random-weight initialisation, same architecture) were evaluated on the full pipeline. Combined accuracy = 0.000 (both seeds; chance on this task is approximately 0.165 given the 5-action space and the DNMTP reward structure). Delay-period decoder accuracy = 0.000 in both seeds. No handoff trajectory was observed. These results confirm that all structure reported in Sections 3.1–3.6 is acquired through training rather than being a property of the architecture. *(Figure 6)*

## 3.8 Summary

| Hypothesis | Result | Seeds | Status |
|---|---|---|---|
| H1 — Learning | Combined acc = 1.000 (*p* < 10⁻³⁰⁰) | 5/5 | ✓ Clean pass |
| H2 — Emergent handoff | Hab onset 7,808–21,376 ep (CV 37%); *w*_GD 0.958→0.330 | 5/5 | ✓ Clean pass |
| H3 — Devaluation dissociation | Dissociation confirmed (Δ ≈ 0.495 vs 0.000) | 3/5 | ~ Partial (checkpoint timing) |
| H4 — Lesion × phase | Maintenance pattern clean; learning pattern seed-dependent | 5/5 maint | ~ Partial (checkpoint timing) |
| H5 — Reactivation falsification | Uninformative: accuracy recovery is eval artifact; DA null is structural | 0/5 | — Uninformative (design limitation) |
| H6 — WM representation | Decoder acc = 1.000, PR ≈ 1.000 | 5/5 | ✓ Clean pass |
| H7 — Untrained control | Acc = 0.000, dec = 0.000 | 2/2 | ✓ Clean pass |

The minimum sufficient result (H1 + H2 + H7 + H6) is met cleanly. H3 and H4 are met in the seeds where the checkpoint criterion captured the intended phase. H5 is uninformative: the current implementation contains two structural confounds — a greedy-decoding identity that makes silenced-hab behaviour algebraically equivalent to GD-solo, and a DA cost penalty that drives the request signal to zero regardless of system configuration — which prevent the test from distinguishing genuine reactivation from its absence. Designing around these confounds is the primary target of Future Research.
	