---
name: plan-memoire
type: writing-plan
created: 2026-06-28
status: in-progress
---

# Mémoire Plan — Big-picture session

---

## Deadline & scope

- **Actual deadline:** End of June 2026 (30 June or 1 July)
- **Writing approach:** Write as if results are TBD — describe the ideal experiment; placeholder numbers where needed

---

## Central claim

Dopamine as expression gain (`r_out = gain(DA) · tanh(h)`) explains the reversibility that migration accounts cannot. The model reproduces Villet et al.'s (2025) lesion/devaluation/reactivation pattern from four design choices, each tested by an ablation.

## Four design choices (+ ablation each)

| Choice | Ablation | Predicts |
|---|---|---|
| Ego/allo observational split (GD=6D, Hab=4D) | Give hab full 6D obs | Functional distinction may weaken |
| Low-rank habitual network (`hab_rank=4`) | Full-rank hab | Handoff/attractor may be less clean |
| Value-free APE learning rule | Reward-coupled habit | Devaluation insensitivity breaks (H3 fails) |
| DA output gain (`gain_DA=0.5`) | `gain_DA=0` (fixed gain) | Reactivation fails (H5 fails) |

---

## Methods section

**Status:** Rewritten 2026-06-28. See `memoire_method_section_draft.md`.

Key changes from old draft:
- Hypotheses moved to §2.2 (before architecture)
- Leans on intro — no biological re-motivation inline
- 4 phases described correctly (pre-sample, sample, delay with pushback+hold, choice)
- Correct obs dimensions: GD=6D, Hab=4D
- Low-rank hab added as design choice
- Four ablation conditions in §2.6
- Grace/Dreyer removed; widen/deepen framing (Naudé 2024)
- H5 explicitly framed as honest pass-or-fail
- H6 correctly attributed to habitual system's delay states

---

## Remaining sections to write

| Section | Status | Notes |
|---|---|---|
| Introduction | Draft exists (`memoire_introduction_section_draft.md`) | Good — check for Grace/Dreyer citation to replace |
| Methods | **Done** (2026-06-28) | |
| Results | Not started | Write with placeholder numbers |
| Discussion | Not started | Migration-vs-reversible-handoff framing; H5 honesty |
| Conclusion | Not started | Short; what Stage 1 established |
| Abstract | Not started | ≤250 words, write LAST |

---

## Writing order

1. Results (by hypothesis order: H1 → H2 → H3 → H4 → H5 → H6 → H7; then ablations)
2. Discussion
3. Conclusion
4. Abstract (last)

---

## Open questions

- Grace/Dreyer citation still appears in §2.3.1 of intro draft (§1.3, third paragraph) — needs replacing with widen/deepen framing per CLAUDE.md §3
- H5 stochastic eval rerun status unknown — write as "not conclusively tested due to evaluation confound; placed in Future Research" if rerun not available
