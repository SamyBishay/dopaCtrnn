# Implementation Roadmap — Cortico-Striatal WM Model (v2)

> Ordered plan for implementing the ladder, the protocol, and the batch experiments.
> Read alongside `PROJECT_RECAP.md` (the science) and the Naudé/Kutter mechanism
> analysis. This document is about *what to change, in what order, and why*.
>
> **v2 changes:** the delay mechanism is now resolved against the supervisor's actual
> code (the TUNL file), not guessed; difficulty is pinned; rank moves early; the
> curriculum decision is made; Go/NoGo moves to future research; L/R distribution
> confirmed. Items still genuinely undecided are marked **[DECIDE]**.

---

## 0. Resolved from the supervisor's actual code (the TUNL file)

These were loose strings; reading `tunl_a2c_two_area.py` closed them.

### 0.1 Delay — RESOLVED
The supervisor's delay (lines 271–278) is a **step counter**: `delay_t` increments each
step while in the delay, the phase signal decays ×0.10/step, and the delay ends when
`delay_t >= len_delay`. `len_delay` is advanced by a **curriculum** (lines 1181–1189):
**promote** (`+1`) when accuracy is high and stable (≥ `TARGET_ACC`, fluctuation < 0.08,
≥ 200 stable steps), **demote** (`−1`) when accuracy collapses (< 0.35), bounded by
`START_DELAY`/`MAX_DELAY`.

This is mechanically what your current code already does. So:
- **Matching the supervisor = step-counted delay + curriculum.** You already have this.
- **Villet used a fixed 90-second delay** — a different unit (seconds) and no curriculum.
  The two targets *conflict*; you cannot be identical to both.
- **Resolution (no invention required):** train with the supervisor's curriculum
  (it solves the bootstrapping problem), and **report/evaluate at a fixed delay** for the
  Villet comparison. The supervisor already does exactly this — `solo_eval` and
  `delay_eval_grid` take a `fixed_delay` argument (line 587) and evaluate at pinned
  delays. Adopt that pattern: curriculum for training, fixed delay for reporting.

  The "hallucinated variable delay" worry is resolved: the variable/curriculum delay is
  the supervisor's real design, line for line — not invented. The fixed delay is the
  *evaluation* setting, also hers.

### 0.2 L/R distribution — RESOLVED (all three agree)
Villet: the blocked arm "was decided randomly and changed between trials" — an
unconstrained per-trial coin flip. Supervisor (line 200):
`self.sample_loc = left if rng.choice([0,1])==0 else right` — unbiased per-trial flip.
Your code: `rng.choice(["L","R"])`. **All three agree. Keep the per-trial coin flip.**

**One divergence to note:** the supervisor has a **correction-trial** mechanism (lines
199–200, 269) — after an error, the sample side is *not* re-randomised but repeats.
Villet does **not** describe correction trials. So here the supervisor diverges *from*
Villet. **Decision: do NOT add correction trials** — Villet wins, per the reproduction
goal. (This is a place where "match supervisor" and "match Villet" conflict; we choose
Villet.)

### 0.3 Curriculum timing vs. the handoff — RESOLVED (use the supervisor's pattern)
The curriculum interacts with the emergent-handoff claim (H2): both the curriculum's
promotion and the handoff are tied to rising accuracy, so a handoff measured *while the
curriculum is still advancing* is confounded — you cannot tell "handoff because the
habit became competent" (your claim) from "handoff because the curriculum stopped
moving."

The supervisor already solves this: her warmup, handover, and all the interesting
dynamics only begin **once `current_delay == max_delay`** (line 1127) — i.e., after the
curriculum has reached its ceiling and frozen. **Adopt this: freeze the curriculum at
its ceiling before the handoff/measurement window.** Train with the curriculum to reach
the ceiling, then hold the delay fixed; measure the handoff only in that stationary
regime. This keeps trainability (the scaffold) *and* a clean emergence claim (no moving
schedule during measurement). This is the middle path, and it is the supervisor's
actual design — not an invention.

---

## 1. Remaining decisions (genuinely undecided)

- **[DECIDE-SPLIT] The scalar split.** E2, E3, and the gate-clamp control require
  separating the expression-gain signal (→ `W_eff`, Naudé's DA-excitability term) from
  the arbitration signal (→ `w`). Strongly recommended — without it those experiments
  collapse into untestable forms, and the model does not faithfully implement the Naudé
  mechanism it cites. *Blocks: Steps 5, 7, 8 and experiments E2/E3.*

- **[DECIDE-DELAY-VALUE] The fixed evaluation delay.** Decide the single delay value at
  which Villet-comparison results are reported. Villet's is 90 s; your task is in steps,
  so this is a translation choice (what step-count stands in for 90 s in your maze). Pick
  one, justify it, hold it fixed across all reported results. *Blocks: reporting, not
  training.*

**Decided and recorded (do not silently reopen):**
- **Arbitration = convex mixture** `w·π_GD + (1−w)·π_H`. Not tested against alternative
  output-combination schemes; stated as a scope limitation in Methods.
- **Go/NoGo opponency → future research.** Drop the OpAL/Jaskir & Frank citation for
  opponency from the architecture claims. Do not test the cosmetic subtraction. The
  question "does true OpAL-style opponency (distinct Go/NoGo learning dynamics) improve
  the model?" is explicit future work.
- **Difficulty is pinned, not swept** in the ladder (see §4 note).

---

## 2. The changelog so far (already in the codebase)

For reference — from prior work; **verify against the actual current code, some may have
drifted**:
1. Vectorised training loss over B (`train.py`).
2. GAE(λ) + return normalisation (`train.py`, `config.py`); λ=1 reproduces old returns.
3. APE-decay option, off by default.
4. Checkpoint/resume, atomic.
5. Low-rank habitual net (`hab_rank` flag), full-rank default.
6. Vectorised + parametric environment; single-env facade preserves the scalar API.
7. Vectorised evaluation (`evaluate_vec`); H1/H3/H4/H5 use it.
8. Two pre-existing analysis bugs fixed; H6 one-class guard.
9. Degenerate-grid guard; default `len_edge=7, difficulty=0`.
10. Standalone visualiser.

**Note:** the parametric grid (6, 9) introduced the grid-geometry-vs-delay tangle. Per
§0.1, keep the grid *geometry* (stem length) and use the supervisor's *step-counted
delay + curriculum* mechanism; they are separable.

---

## 3. The order to implement the remaining changes

One change per commit; smoke test after each.

### Step 1 — Make the base model learn (E0 precondition)
**This is where you are.** Tune the optimisation only (lr, entropy, curriculum pace,
reward shaping, warmup/ramp) — never the architecture. Debug order: reward reaches the
agent → curriculum pace → exploration → two-system bootstrap → (diagnostic) confirm the
GD net alone can learn. **Gate:** accuracy ≥0.80, binomial p<1e-3, ≥5 seeds. If the
architecture (not optimisation) blocks learning, STOP — that is a scientific decision.

### Step 2 — Align the delay with the supervisor + Villet pattern  *(uses §0.1)*
- Confirm the training delay is step-counted with the curriculum (promote/demote as in
  the supervisor's lines 1181–1189). You largely have this; reconcile any differences
  against her code line-for-line.
- Implement **curriculum-freeze at ceiling**: the handoff/measurement window begins only
  after `current_delay == max_delay` (supervisor line 1127). Gate all phase checkpoints
  and the handoff measurement on this.
- Add a **fixed-delay evaluation path** (homologue of her `fixed_delay` in `solo_eval`)
  so Villet-comparison results are reported at one pinned delay. *(needs
  [DECIDE-DELAY-VALUE])*
- **Test:** scripted-optimal agent still solves the task; training delay behaviour
  matches the supervisor's; the fixed-delay eval runs at the chosen value.
- **Re-run Step 1's gate** — changing delay handling changes the learning problem.

### Step 3 — Lock checkpoints to Villet's behavioral phase criteria
Map the learning and maintenance checkpoints to *Villet's actual criteria* (maintenance:
80% for 3 consecutive days; learning-devaluation group: 70% for 2 non-consecutive days),
translated to your step/episode units — not arbitrary accuracy thresholds. **Test:** the
two checkpoints are reached and bracket the criteria correctly.

### Step 4 — Run the rank diagnostic early (E8 moved up)  *(no new code; flag exists)*
`hab_rank` already works and is unblocked (no scalar split needed). Run the rank sweep
(1/2/4/8) **now**, right after learning is established, because the minimum rank that
supports the task is a *diagnostic* that informs how you read every later experiment
(a rank-1 habit is a different system than a rank-8 one). This is the cheapest,
lowest-friction experiment and it can run in parallel while you implement the split.
**Test:** each rank learns (or fails) cleanly; record the minimum sufficient rank.

### Step 5 — Implement the scalar split  *(needs [DECIDE-SPLIT])*
Separate the **expression-gain** signal (→ `W_eff`, Naudé DA-excitability) from the
**arbitration** signal (→ `w`) in `model.py`, each at its single computation site.
Default (signals tied) must reproduce the Step-1 learned model exactly. Relabel the two
timescale populations **widen (fast/decision) vs. deepen (slow/maintenance)** per Naudé;
remove D1/D2 receptor semantics from comments/variables. Cite Kutter only for "cortical
DA controls strength + temporal stability of decision codes," not a receptor-direction
mapping. **Test:** tied → smoke output unchanged from Step 1; untied → both run.

### Step 6 — Add the gate-clamp control to the analysis
Add a routine that, for a reactivation test, clamps `w` open and measures whether the
*preserved GD policy* recovers accuracy (vs. recovery only because the gate moved). The
mandatory H5 control. **Test:** runs and returns the dissociation measure; interpretation
waits for a learned model.

### Step 7 — Add the variant flags for the ladder  *(needs [DECIDE-SPLIT])*
Config flags, each defaulting to current behaviour, each branching at one site:
- `gate_mode = "expression" | "scheduled"` (E2)
- `da_components = "both" | "gain_only" | "weights_only"` (E3, Naudé decomposition)
- `habit_rule = "value_free" | "value_coupled"` (E4)
- `habit_obs = "position_free" | "allocentric"` (E5, ego/allo)
- `hab_rank` already exists (used in Step 4)
**Test:** smoke passes for BOTH values of every flag.

### Step 8 — Wire the batch runner
Runs an experiment (a flag-pair) across ≥5 seeds as concurrent single-thread processes,
checkpointed, into the no-overwrite results layout. Chance-guard on each.
**Test:** both arms × 5 seeds land in `results/<exp>/<arm>/<ts>_<commit>/`.

---

## 4. The per-experiment protocol (apply identically)

1. **Pre-register** the predicted direction and the falsifying result, dated, before
   running.
2. **≥5 seeds per arm.**
3. **Learning gate first** — both arms must independently clear ≥0.80, p<1e-3. If
   either fails, the contrast is uninterpretable; stop and report, do not tune to rescue.
4. **Phenomenon measure** on both arms, with effect size and bootstrap CI across seeds.
5. **Chance-level guard** on every result; **gate-clamp control** on every reactivation
   claim.
6. **Skeptic pass** — seek the boring explanation before recording a pass.
7. **Record** mechanism varied, both arms' numbers, effect + CI, pass/fail, and if fail:
   tuning-issue vs. genuine-negative. Tie to seed + commit + config; never overwrite.

**The rule that protects the project:** tune ONLY to make models learn (the gate). NEVER
tune to make a hypothesis pass. A hypothesis that fails on models that genuinely learned
is a result — report it, including negatives.

**Difficulty is pinned across all ladder experiments.** Villet had one maze geometry;
reproducing Villet means one difficulty, held fixed, chosen once to match Villet's maze
proportions as closely as the grid allows. Varying difficulty mid-ladder would confound
every contrast. The *only* place difficulty becomes an experiment is the Tier-3
robustness check (E11 below) — "does the handoff replicate across maze sizes?" — which is
a generalization result, not part of reproducing Villet.

---

## 5. The experiments to run

E1 (single-system baseline) **removed**: Villet's premise is two areas; a one-area
baseline is not part of reproducing their finding.

**Tier 0 — precondition**
- **E0 — base model learns.** Gate on ≥5 seeds. Everything waits on this.

**Tier 0.5 — diagnostic (cheap, unblocked, run early)**
- **E8 — minimal habit rank** (`hab_rank` sweep 1/2/4/8). Moved up: no new code, no
  split needed, and the minimum sufficient rank informs how every later experiment is
  read. Run right after E0, in parallel with implementing the split.

**Tier 1 — essential to reproduce Villet's four findings**
- **E2 — expression-gating necessary for reversibility** *(the crux)*.
  `gate_mode = scheduled | expression`. Both may hand off; only expression should
  reactivate (survive gate-clamp). *Needs Step 5.*
- **E3 — plasticity/excitability decomposition** (Naudé-faithful necessity).
  `da_components = weights_only | gain_only | both`. Prediction: neither alone
  reactivates; only both. *Needs Step 5.* Cleanest necessity result.
- **E4 — value-free habit necessary for devaluation dissociation.**
  `habit_rule = value_free | value_coupled`. Dissociation with value-free, breaks with
  value-coupled.
- **E5 — ego/allo split** *(alternative-explanation killer)*.
  `habit_obs = position_free | allocentric`. Tests whether the handoff is driven by the
  dopamine mechanism or merely by the observation asymmetry. Must be Tier 1.

**Tier 2 — strengthening**
- **E6 — gain on assembly vs. uniform gain** (Naudé's multiplicative-on-potentiated
  claim).
- **E7 — learning-rule factorial** (2×2: habit rule × GD rule). Separates necessity for
  D vs. H. Expect one empty cell (fully value-free GD may not learn — itself a result).

**Tier 3 — novel predictions (after replication is established)**
- **E9 — delay controls handoff timing.** A *deliberate manipulation of Villet's
  held-constant delay*: away from the fixed evaluation value, measure how handoff timing
  shifts. Opposes migration. *(Frame explicitly as manipulating the variable Villet held
  at 90 s.)*
- **E10 — distance-to-attractor reactivation** (Naudé-grounded; unique to the embodied
  task).
- **E11 — difficulty robustness** (sweep maze difficulty). The *only* experiment that
  varies difficulty. Generalization check, not part of the Villet reproduction.

**Deliberately not run:** single-system baseline (removed); output-combination schemes
(fixed as convex mixture, named a limitation); cosmetic Go/NoGo opponency (→ future
research); any hand-designed gain schedule (would defeat the emergence claim — the gain
relationship must be a *measured outcome*, never a designed input).

**Scope line:** E0 + E2 + E3 + E4 + E5 (+ E8 as a cheap diagnostic), done well (≥5 seeds,
gate-clamp controls, pre-registered, curriculum frozen at ceiling before measurement),
is a complete, defensible M1 thesis. Tier 2–3 is upside. Six experiments done
rigorously beats eleven done badly.

---

## 6. Critical path

```
[DECIDE-SPLIT] ─────────────────────────────┐
                                             │
Step 1 (learn, E0) ─┬─> Step 2 (delay+freeze) ─> Step 3 (Villet phases) ─┐
                    │                                                      │
                    └─> Step 4 (RANK diagnostic E8, parallel, unblocked)   │
                                                                           │
[DECIDE-SPLIT] ─> Step 5 (split) ─> Step 6 (gate-clamp) ─> Step 7 (flags) ─> Step 8 (batch)
                                                                           │
                                                                           v
                                              E0 gate ─> E2,E3,E4,E5 (Tier 1) ─> Tier 2/3
```

Get the model learning; align the delay to the supervisor's mechanism with a frozen
ceiling for measurement and a fixed delay for reporting; run the rank diagnostic early;
then split the scalar and run the crux ladder. **Do not run any experiment until E0 is
green on ≥5 seeds** — every result above it is otherwise an artifact.
