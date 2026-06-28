# GOALS.md

_Keep this under one page. Update when direction changes; add a dated changelog line below._

---

## Current stage: writing the mémoire (due 22 June 2026)

**The project's claim:** Dopamine is the unstated mechanistic substrate of the Villet et al. (2025) cortico-striatal handoff. `W_eff = f(DA)·W` (expression gain, not weight modification) explains instant reactivation; the DA-request neuron makes the handoff emergent rather than scheduled; the APE habitual rule makes devaluation-insensitivity structural.

**Stage 1 is complete.** Verified results (default config, `Code/main/`):
Citation correction: do not cite a fixed D1/D2 receptor-affinity direction
(Grace 1991 / Dreyer et al. 2010) for the two-timescale engine. Kutter et al. (2026)
report the opposite direction for this task regime. Use the widen (fast/decision) vs.
deepen (slow/maintenance) framing from Naudé et al. (2024) instead — see CLAUDE.md §3.
| Hypothesis | Result |
|---|---|
| H1 learning ≥80% | combined accuracy → 1.00 ✓ |
| H2 emergent handoff | habitual competence (~ep1500) precedes w_GD drop (~ep2500) ✓ |
| H3 devaluation dissociation | learning-phase drop ≈0.5, maintenance ≈0.0 ✓ |
| H5 reactivation (falsification) | OLD RESULT INVALID — greedy eval made recovery a mathematical identity. Rerun with stochastic eval before reporting anything. |
| H6 working-memory attractor | delay-period arm decoder = 1.00 ✓ |
| H7 untrained control | ≈ chance ✓ |

Stages 2–7 = planned/future work. Write Stage 1 as self-sufficient.

**Planned tau/gain experiments (H_tau ladder — all require ≥5 seeds):**

| Experiment | Description | Status |
|---|---|---|
| E6 | DA uniform tau shortening — prerequisite for all below | planned |
| E7 | DA tau widen/deepen split — is asymmetry necessary? | planned |
| E9 | Recurrent gain + dual tau (parents: E10, E7) | planned |
| E10 (NEW) | Recurrent gain only — isolate before combining | planned |
| E11 (NEW) | Recurrent gain + DA plasticity (LR scaling) | planned |
| E12 (NEW) | DA excitability — additive bias in GDNet.step() | planned |
| E13 (NEW) | Excitability + plasticity + dual tau (parents: E12, E11, E7) | planned |

**Corrected codebase prerequisites before any ladder rerun:** stochastic eval, W_in std=1.0 (F13), APE teaching signal from GD output only, RPE-based DA gate, pushback delay phase, noise injection, DA excitability bias, hab_onset criterion ≥0.99. See `fixes_summary.md` §1.

---

## Remaining deliverables

| Item | Due | Status |
|---|---|---|
| Mémoire: Introduction | 22 Jun | draft started (`Mémoire/memoire_introduction_section_draft.md`) |
| Mémoire: Method | 22 Jun | draft started (`Mémoire/memoire_method_section_draft.md`) |
| Mémoire: Results, Discussion, Conclusion, Abstract | 22 Jun | not started |
| Stage report (≤12 pp) | 22 Jun | not started |
| Mémoire PowerPoint (15 min) | 29 Jun | not started |
| Stage report PowerPoint (10 min) | 29 Jun | not started |


---





