---
title: Fixes Summary — Agent Instructions
updated: 2026-06-28
---

# Fixes Summary

For each fix: what was agreed to keep from the analysis, written as a concrete instruction
for an agent in a new session. Fixes not yet discussed are marked **TODO**.

Cross-reference the full problem statements and tradeoff analysis in `fixes.md`.

---

## F1 — Greedy decoding / continuous training signal

**Agreed decisions:**

1. **Distance metric (option 1) — dropped.** The environment always terminates at an arm
   wall (correct or incorrect), so distance at trial end is binary. Efficiency (steps
   taken) is the meaningful continuous signal and is already logged. No code change needed
   for this.

2. **Rolling training accuracy (option 2) — keep.** Already implemented (`rolling_buf`,
   `rolling_window = 20_000` in `config.py`, applied in `train.py:339`). Training curves
   use the rolling window. Frozen-weight eval is kept **only** for checkpoint hypothesis
   tests (H1–H6). Do not abolish it, just demote it from primary training metric.

3. **APE teaching signal (option 3) — change teacher, keep per-timestep structure.**
   Currently APE uses the *combined* policy's argmax as the one-hot target. Change to use
   **GD's own output vector** as the teacher (not the combined policy). Rule per timestep:
   - If `argmax(pi_gd) == WAIT`: target = one-hot WAIT, cross-entropy loss.
   - Else: target = `softmax(pi_gd[N,S,E,W])` — soft distribution over the 4 movement
     directions only. Use KL divergence or cross-entropy with soft targets.
   This applies to both subnetworks (GD teaches hab; same signal, same rule).
   File: `train.py` — find where `ape_loss` is computed and change the target construction.

4. **Delay phase redesign — environment.py.** Three sub-changes:

   **a. New PUSHBACK phase** (between SAMPLE and CONFINED/DELAY):
   - When emouse reaches the sample arm end, enter PUSHBACK (add new phase constant,
     e.g. `PUSHBACK = 4`, shift existing indices if needed).
   - The environment **overrides agent actions**: move the emouse one cell per step back
     along the fixed return path: arm end → junction (along arm row toward `cx`) → down
     the stem column → START.
   - `delay_idx` starts counting from the first PUSHBACK step (not from arrival at START).
   - No step cost during PUSHBACK (same as current delay: `nav_active = active & ~in_delay`
     — extend this mask to cover PUSHBACK too).
   - Remove the teleport to START (current `environment.py:267`).

   **b. CONFINED sub-state** (emouse arrives at START):
   - When the emouse reaches START during PUSHBACK, set `obs[2] = 1.0` (confinement bit).
   - Block movement out of START: any action that would move the emouse off the START cell
     is rejected (position stays at START). WAIT and moves that stay in place are allowed.
   - `delay_idx` continues counting.

   **c. Transition to CHOICE** (when `delay_idx >= current_delay`):
   - Set `obs[2] = 0.0` (confinement bit cleared — "you may leave").
   - Set `sig_choice = cfg.sig_val` (obs[5]) as now.
   - Enter CHOICE phase. Max-step pressure plus `sig_choice` signal motivates navigation.
   - Add an assertion at env init: `cfg.delay_start >= pushback_path_length` where
     `pushback_path_length` is computable from geometry (arm half-length + stem length).

   **Note on obs[2]:** the supervisor's code confirms `obs[2]` is always 0 and never
   written except to zero it (`tunl_a2c_two_area.py:306`). Repurposing it as the
   confinement bit is safe and requires no `OBS_DIM` change. The GD model receives a
   meaningful obs[2] for the first time — no architecture change needed, the input
   dimension stays 6.

---

## F2 — Fixed eval seed / noise injection — TODO

---

## F3 — H5 mathematical identity — TODO

---

## F4 — DA-request penalty over-determines H5 null — TODO

---

## F5 — wgd_drop timing tracks penalty schedule — TODO

---

## F6 — n=1 seeds uninterpretable — TODO

---

## F7 — E3 weights_only inconsistency — TODO (covered by F6)

---

## F8 — Smoothing already done — TODO (already implemented)

---

## F9 — Results text overstates H5 — TODO (covered by F3)

---

## F10 — Protocol organised around wrong question — TODO

---

## F11 — No DA-modulated plasticity experiment — TODO

---

## F12 — Early stopping and hab_onset criterion — TODO
