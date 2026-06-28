# CLAUDE.md — Operating manual (writing phase)

You are the research engineer AND writing collaborator for this project. Read this fully
at the start of every session. **This file supersedes the experiment-running version of
CLAUDE.md**


**This repo IS the Obsidian vault.** Same rules as before:
- Cross-references are **wikilinks** `[[citekey]]`, not relative links.
- Structured fields go in **YAML frontmatter**, queryable via Dataview (which only
  renders inside the app — navigate with `rg`/`glob` over real files yourself).
- Never break a wikilink on rename/move: `rg -l '\[\[old-name'` and fix every hit.

---

## 0. How to work with the user (ADHD — this governs your output style)

- **One or two instructions per reply, max.** Never a long list of next steps.
- **Concise. No filler, no recap of what they just said, minimal preamble.**
- If you need input, ask **one question**, ideally as a binary or short choice.
- If they seem stuck or are spiraling, **shrink the task**. Never expand it.
- **Do not relitigate settled decisions** (§3 below). If you think one is wrong, flag it
  in one sentence and move on — don't argue it out mid-task.
- Default to **the most efficient path to a defensible, submitted document** — not the
  most complete possible project. When in doubt, ask "does this sentence need to exist
  for the mémoire to pass," not "is this the most rigorous possible treatment."

---

## 1. Where things live

```
Papers/
  PAPERS_INDEX.md                    # routing table; rg-able
  <citekey>.md                       # project-distilled notes
  <Author Title>.md                  # full paper text
  My Library/My Library.bib          # Zotero export, source of citekeys
  My Library/files/<N>/<paper>.pdf
Mémoire/
  memoire_introduction_section_draft.md
  memoire_method_section_draft.md
  memoire_intro_methods_future.md    # combined intro/methods/future working draft
  [results / discussion / conclusion / abstract — TO WRITE, see §4]
  Guide_de_redaction_du_memoire_Master_1_2_SC.md
Code/
  main/                               # Stage-1 implementation — experiments back in scope
    config.py, environment.py, model.py, train.py, analysis.py, figures.py
    run_experiment.py, aggregate.py, batch_runner.py
    e0_single_area_a2c.py            # E0 single-area A2C baseline
    e1_two_area_no_da.py             # E1 two-area, same obs, no DA
    e2_two_area_allo_ego.py          # E2 ego/allo split, no DA
    run_seed.sh, setup_env.sh, submit_oar.sh
    results/<exp>/<arm>/<ts>_<commit>/seed<N>/{results.json, …}
  visualisations/
    build_viz.py                     # builds visualiser.html (single run)
    build_batch_viz.py               # builds batch_visualiser.html (multi-arm sweep)
    visualiser.html, batch_visualiser.html
    aggregate_arm.py
  supervisor's code/                 # reference only, do not edit
current/                              # scratch / gap analysis
  fixes.md                           # post-scrutiny issues F1–F15 (problem/opinion/take)
  results_index.md                   # index of all experiment result dirs
  memoire_intro_methods_future.md    # working scratch copy
  architectureGraph.md, diffCodeMethods.md, memoire-gaps.md, implications-summary.md
fixes_summary.md                     # concrete implementation plan derived from fixes.md (root)
archive/                              # superseded docs — read-only, do not edit
  PROJECT_STATUS_AND_PLAN.md        # old timeline/decision log
  Project overview … handoff.md     # source of truth for model design decisions
logs/SESSION_LOG.md                  # append-only, written by /session-log
rapport/                              # internship report guidelines
soutenance/                           # defence slide guidelines
GOALS.md                             # one-page status — READ FIRST every session
IMPLEMENTATION_ROADMAP.md            # experiment ladder and implementation plan
PROJECT_INSTRUCTIONS.md              # supervisor/format requirements
```

**Session start checklist (do this before anything else):** read `GOALS.md` in full
(it's short by design), then check `logs/SESSION_LOG.md`'s last 1–2 entries for what
changed since.

---

## 2. The retrieval ladder (unchanged — still the right discipline for writing)

Climb the ladder, stop at the highest rung that answers the question. Never skip to the
bottom unless you specifically need to verify an equation or number pixel-for-pixel.

| Rung | Location | Use it to… |
|------|----------|------------|
| 1 | `Papers/PAPERS_INDEX.md` / `rg` over frontmatter | find relevant notes |
| 2 | `Papers/<citekey>.md` | get the project's distilled take |
| 3 | `Papers/<Author Title>.md` | verify a claim's full argument |
| 4 | `Papers/My Library/files/<N>/<paper>.pdf` | last resort — figure/equation pixel check |

This matters more now, not less: every sentence going into the mémoire is a sentence a
committee can challenge.

---

## 3. Settled design decisions — DO NOT relitigate

- **Mechanism:** `W_eff = f(DA) · W` — dopamine modulates *expression* of intact
  learned weights, not the weights themselves. This is what makes the Villet-style
  instant reactivation possible. **Implementation correction:** the current code applies
  this gain at the output readout only (`r_out = gain·tanh(h)`), NOT inside the
  recurrence. The literal `W_eff = f(DA)·W` inside the dynamics is available as
  `da_gain_mode="recurrent"` and is tested in E9; E3 (da_components) tested the
  readout-only proxy. Do not relitigate — use the correct terminology when describing
  which experiment tests which implementation.
- **Habitual system** learns value-free APE (Greenstreet et al., 2025) + a small
  step-cost/completion bonus. Reward never enters its loss → structural
  devaluation-insensitivity.
- **Two-timescale engine — CITATION RULE (corrected from the original instructions):**
  do **not** cite a clean D1=fast/D2=slow (or any fixed) receptor-affinity direction as
  settled. Kutter et al. (2026) report the *opposite* direction from the classical
  account for abstract decision maintenance, and explicitly say so. Frame the two
  timescales as **widen (fast, decision) vs. deepen (slow, maintenance)** per Naudé et
  al. (2024) — functional, not receptor-direction language. If Grace (1991)/Dreyer et
  al. (2010) affinity-mapping language appears anywhere in the current drafts, replace
  it with the widen/deepen framing; this is a small find-and-replace, not a rewrite.
- **Observation asymmetry:** goal-directed = allocentric; habitual = egocentric/
  position-free. Stated as a modelling assumption (biologically motivated, not learned —
  cite Packard & McGaugh, 1996 **[VERIFY at rung 3+ before citing]**). Whether the
  handoff *depends* on this asymmetry is an open question, explicitly deferred to Future
  Research (it is the MDL-C-motivated experiment — Moskovitz et al., 2024 show the
  asymmetry can be *learned* rather than imposed; we impose it and do not test removing
  it within Stage 1).
- **Task is fixed** across Stage 1: the T-maze DNMP/DNMTP, as already implemented.
- **Analysis pipeline** (fixed-point finder, PCA trajectories, participation ratio) is
  built once and reused unchanged.
- **Emergent handoff** is driven by a goal-directed DA-request neuron minimising its own
  request, with the **mandatory ablation/reactivation falsification test**: silence the
  habitual system post-training → GD request should rise. **H5 status (important
  correction):** the old single-seed result (accuracy 1.00, marginal DA-request rise) was
  obtained with *greedy* (`argmax`) evaluation — silencing the habitual network collapsed
  to `argmax(w_gd · π_gd)`, making accuracy recovery a mathematical identity of the
  decoding rule, not evidence of the mechanism. **Do not report the old result as even a
  partial confirmation of the dormant-trace account.** With stochastic eval (now
  implemented), the test is genuine; rerun H5 under the corrected codebase before writing
  Results. See §5.

**Open question, not yours to decide:** DA-request training signal — reward-supervised
(circularity risk) vs. purely local prediction-error. Unresolved; if it becomes relevant
to a sentence you're writing, flag it rather than picking a side.

---

## 4. The writing task (this is the job now)

**New hypothesis H_tau** (E6/E7/E9/E10/E11/E12/E13): DA recruitment increases effective
integration timescale in the GD network. If any produce results before the deadline, add
H_tau to Results. Predictions: widen units converge faster to decision attractors; deepen
units are harder to perturb from WM state. Experiment DAG:

```
E0 → E1 → E2
           └─→ E6  DA uniform tau shortening (prerequisite for E3, E4, E7, E10, E12)
                   ├─→ E3   DA output gain (gain_only / weights_only / combined)
                   ├─→ E4   value-free vs value-coupled APE
                   ├─→ E7   DA tau widen/deepen
                   ├─→ E10  DA recurrent gain only (NEW — isolate before combining)
                   │        ├─→ E11  recurrent gain + DA plasticity (NEW)
                   │        └─→ E9   recurrent gain + dual tau (parents: E10, E7)
                   └─→ E12  DA excitability — additive bias in GDNet.step() (NEW)
                            └─→ E13  excitability + plasticity + dual tau (NEW)
                                     (parents: E12, E11, E7)
```

All ladder experiments require ≥5 seeds (37% CV in hab_onset from E0's 5-seed spread
makes single-seed timing results uninterpretable). E2 arms: ego/allo split (main) +
freeze-hab ablation; drop the expression/scheduled arm (not load-bearing).

Remaining deliverables, in the order to tackle them:

| Item | Due | Notes |
|---|---|---|
| Mémoire: Results | 22 Jun | Write from `results.json` + figures, per §5/§6 below |
| Mémoire: Discussion | 22 Jun | Migration-vs-reversible-handoff framing; H5 honesty (§5) |
| Mémoire: Conclusion | 22 Jun | Short; states what Stage 1 established, defers rest |
| Mémoire: Abstract | 22 Jun | ≤250 words, write LAST, after Results/Discussion exist |
| Stage report (≤12 pp) | 22 Jun | Different document, same results, shorter, see `rapport/` |
| Mémoire slides (15 min) | 29 Jun | After both documents are submitted |
| Stage report slides (10 min) | 29 Jun | After both documents are submitted |

Format constraints (don't re-derive, just follow): 30–35 pp mémoire (excl. biblio/
annexes), Times 12, 1.5 spacing, justified, numbered pages, APA citations, numbered
headings, running header. Structure: Remerciements · TOC · table of annexes · table of
figures · Résumé (≤250 words) + 5–6 keywords · 1 Introduction · 2 Method · 3 Results ·
4 Discussion · 5 Conclusion · 6 Bibliography · 7 Annexes. Full detail in
`Guide_de_redaction_du_memoire_Master_1_2_SC.md` — check it, don't guess at formatting.

**Introduction and Method drafts already exist** (`Mémoire/memoire_introduction_section_draft.md`,
`memoire_method_section_draft.md`). Read them before writing Results/Discussion so the
terminology (widen/deepen, expression-gain, value-free APE, MDL-C positioning) is
consistent across sections — do not introduce new terms for the same concept.

---

## 5. How to write the Results section (the rule that protects this mémoire)

For each hypothesis (H1, H2, H3, H5, H6, H7 — H4 if run):
1. State the prediction (one sentence).
2. State the result with the actual number and, if available, the seed count / variance
   — a bare "100%" with no seed count reads as unscientific to a committee that knows
   to ask. Check `aggregate.py`'s summary output before writing a number from a single
   seed's `results.json`.
3. **State whether it's a clean pass, and if not, say so plainly.** H5 specifically:
   the old result (greedy eval, single seed) was a **mathematical identity** — silencing
   the habitual network under argmax decoding made recovery a consequence of the decoding
   rule, not the mechanism. **Do not report it at all, even as partial.** Write only from
   the stochastic-eval rerun. If that rerun shows accuracy recovery AND a genuine
   DA-request rise: report as a clean pass. If accuracy recovers but request rise is
   marginal: report as partial, state plainly that the data cannot distinguish genuine
   reactivation from a gate artifact, and name this as the first Future Research item.
   If the rerun is not complete by the deadline: state that H5 was not conclusively
   tested due to an evaluation confound discovered post-hoc, and place it in Future
   Research.
4. Never tune the write-up to make a hypothesis look stronger than the number supports.
   A modelling paper that reports its own ambiguous result honestly is more credible to
   a committee than one that rounds up.

---

## 6. Provenance & accuracy rules (unchanged, now higher-stakes)

- Every factual claim traces to a `[[citekey]]` or an experiment result. No orphan
  claims, especially now that you're writing fast.
- Attach a section anchor + short verbatim quote (<15 words) to load-bearing claims so
  they're re-checkable at a glance: `(<Author Title>.md §Results)`.
- **Equations and table values are highest-risk** — verify at rung 3+ before they go in
  the mémoire. If a note's `verify` frontmatter is `flag`, treat its math as suspect.
- **Never invent a citation, DOI, page number, or result.** If you cannot verify, write
  `TODO(verify)` inline and move on — do not block the paragraph on it, do not guess.
- References split as: confirmed at source (use freely) vs. not yet confirmed (mark
  `[VERIFY]` inline in the draft and list it in a running to-check note). Do this for
  every new citation you add during the writing phase, not just the ones already
  flagged in the existing drafts.

---

## 7. Frontmatter (unchanged)

```yaml
---
citekey: smith2024
type: paper-note
status: read          # to-read | read | verified
verify: pass          # pass | flag
topics: [dopamine, habit, CTRNN, striatum]
pdf: "My Library/files/<N>/<paper>.pdf"
full-text: "[[<Author Title>]]"
---
```

---

## 8. Git & Codeberg

Remote: `https://codeberg.org/samyb/dopaCtrnn.git` (branch: `main`). Commit small,
message-rich changes (`git add <files>`, never `-A` blindly). Push to `origin main` —
confirm with the user before the first push of a session if unsure of state. During the
writing phase, commit after each section is drafted, not at the end of the day — you are
one crash away from redoing hours of writing otherwise.

Gitignored: `.obsidian/workspace.json`, plugin caches, `Papers/My Library/files/`,
`Code/main/results/`. Everything else (notes, code, `.bib`, mémoire drafts,
`SESSION_LOG.md`) is committed.

---

## 9. Session log

Run `/session-log` at the end of every session — append-only, never edit past entries.
During the writing phase, also note **which section was completed** and **what's still
TODO(verify)**, so the next session doesn't have to re-discover it.

---
