# PROJECT STATUS & PLAN

_Working spine for the M1 mémoire + stage report. Knowledge file — keep updated as tasks complete._

---

## Timeline (submission ~22 June, soutenance 29 June)

- **Thu Jun 18** — Method section + experiment code (one combined task; writing the protocol IS the code spec)
- **Fri Jun 19** — Refine code, test runs, fix bugs
- **Sat Jun 20** — Launch Stage-1 experiment; while jobs run, write Introduction from lit review
- **Sun Jun 21** — Results + Discussion + Conclusion + Abstract → mémoire draft complete
- **Mon Jun 22** — Format/polish mémoire + start stage report → **submit both PDFs**
- **Tue 23 – Sat 28** — Finish stage report, build both PowerPoints, rehearse (15 min mémoire / 10 min stage)
- **Sun Jun 29** — Soutenance

---

## Experiment scope (committed)

- **Anchor: Stage 1** — single 256-unit CTRNN, fixed T-maze DNMTP, actor-critic, establish working-memory attractor + validate analysis pipeline (fixed-point finder, PCA, participation ratio).
- **Stretch: Stages 2–3** only if time. Write so Stage 1 alone suffices.
- Later stages (DA handoff, D1/D2, effective-rank) = **planned/future work** in the mémoire.

---

## ONE-PAGE OUTLINE — MÉMOIRE

Front matter: Remerciements · TOC · Table of annexes · Table of figures · Résumé (EN ≤250 w) · Keywords (5–6)

**1. Introduction**
- 1.1 Goal-directed → habitual transition: documented but mechanistically unexplained
- 1.2 The phenomenon: Villet et al. (2025) — mPFC for learning, DLS for maintenance, devaluation shift, dormant instantly-reactivable fallback
- 1.3 The gap: Villet has no dopamine — no mechanism for why/how control transfers or why fallback is instant
- 1.4 Dopamine as candidate substrate (value/devaluation; prefrontal model-based; striatal DA)
- 1.5 Multiple timescales: D1/D2 phasic–tonic affinity
- 1.6 Local (not global scalar) prediction errors — APE framing *(most original thesis)*
- 1.7 Alternative framing: supervisor's low-rank/attractor account (the fork)
- 1.8 Problématique, objectives, general hypotheses

**2. Method**
- 2.1 Task & protocol: T-maze DNMTP, delay, devaluation, step/wait costs
- 2.2 Environment: batched, configurable observation modes
- 2.3 Architecture & complexity roadmap: the ladder, 256 units fixed
- 2.4 Learning rules: actor-critic (GD); value-free APE (habitual)
- 2.5 DA mechanism: W_eff = f(DA)·W, tonic baseline + phasic request neuron *(planned, Stage 5)*
- 2.6 Analysis pipeline: fixed-point finder, PCA, participation ratio
- 2.7 Controls & falsification: ablation/emergence test

**3. Results** *(= achieved stage)*
- 3.1 Descriptive results (table + figures)
- 3.2 Attractor / fixed-point analysis
- 3.3 Statistical tests (APA)

**4. Discussion** — objective recall · main results · interpretation vs hypotheses & literature · contribution · limits · future work
**5. Conclusion** (few lines) · **6. Bibliography** · **7. Annexes**

---

## ONE-PAGE OUTLINE — STAGE REPORT (M1 detailed, ≤12 pp)

Front matter: Remerciements (opt.) · TOC · Table of annexes (opt.)

**1. Introduction générale** — my academic path → why this stage; how it fits my curriculum
**2. La structure d'accueil** — 2.1 presentation · 2.2 organigramme · 2.3 projects/objectives/missions
**3. Le superviseur** — 3.1 background/specialty · 3.2 role/missions · 3.3 means (tools, methods, partners) · 3.4 discussion (fit between role & means; training vs current activity)
**4. Le stagiaire** — 4.1 my integration · 4.2 project & research question summary (details live in the mémoire) · 4.3 missions assigned · 4.4 tools/protocols used or observed · 4.5 discussion (my positioning, involvement per stage, autonomy, difficulties & workarounds, competencies developed incl. research ethics)
**5. Discussion générale (M1)** — how the stage shapes my professional project; critical analysis of my strengths/weaknesses + improvement paths
**6. Annexes** — at minimum: internship attestations + supervisors' full contact details

---

## DECISION LOG



## BIBLIOGRAPHY — add to Zotero (currently missing from the 94-entry library)

Must add (methods backbone already cited in the synthesis):
- Sussillo & Barak (2013), Neural Computation — fixed-point finder
- Mante, Sussillo, Shenoy & Newsome (2013), Nature — context-dependent computation in PFC RNNs
- Gallego et al. (2017/2020) and/or Churchland et al. (2012) — neural manifolds / population dynamics

Should add (devaluation/habit foundations a jury expects):
- Adams & Dickinson (1981) or Dickinson (1985) — instrumental devaluation
- Yin & Knowlton (2006) — basal ganglia in habit formation

---
