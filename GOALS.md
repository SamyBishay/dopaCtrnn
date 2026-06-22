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
| H5 reactivation (falsification) | silencing habitual → accuracy 1.00 ✓ (DA-request rise marginal — report honestly) |
| H6 working-memory attractor | delay-period arm decoder = 1.00 ✓ |
| H7 untrained control | ≈ chance ✓ |

Stages 2–7 = planned/future work. Write Stage 1 as self-sufficient.

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

| H5 reactivation (falsification) | accuracy → 1.00 ✓ ; DA-request rise marginal — NOT a clean confirmation. Data cannot yet distinguish genuine dormant-trace reactivation from a control-loop artifact (no gate-clamp control run). Write up as an honest partial result in Discussion; name the gate-clamp control as the first Future Research item. |

(This replaces the parenthetical "(DA-request rise marginal — report honestly)" with an
explicit statement of what the ambiguity is and what would resolve it, so whoever
writes Discussion doesn't have to reconstruct that reasoning under deadline pressure.)




