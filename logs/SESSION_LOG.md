# SESSION_LOG

Append-only. One dated entry per session.

---

## 2026-06-19 — init: vault structure, paper notes, GOALS, CLAUDE.md, SESSION_LOG
Commit: 956a7da

**Vault initialization and restructuring:**

- `claude.md.md` → renamed to `CLAUDE.md`; all stale `lit/` path references updated to `Papers/`
- H1 headings added to all 13 paper full-text `.md` files that were missing them
- Duplicate entries removed from `PAPERS_INDEX.md` (Collins & Frank, Lee et al.)
- `project_overview.md` (redundant lowercase copy) removed; canonical file is `Project overview a dopamine-mediated mechanism for the goal-directed-to-habitual handoff.md`

**CLAUDE.md trimmed** (182 → 107 lines): removed non-existent `$RESEARCH_LIBRARY`, `scripts/`, `experiments/`, `src/`, `paper/`, `talk/` sections; updated all paths to match actual vault structure (`Papers/My Library/`, `Code/main/`, `Mémoire/`).

**13 citekey-named paper notes created** (`Papers/<citekey>.md`) with YAML frontmatter (pdf path, full-text wikilink, topics, status) and five structured sections: What the paper does · Key claims · Mechanism/model details · Implications for our model (referencing H1–H7 and design decisions) · Open questions / caveats.

**PAPERS_INDEX.md** slimmed from ~2000-word paragraph summaries to one-line routing entries; all 13 headings updated with correct `→ [[citekey]]` wikilinks; three agent-mismatched links manually corrected (Naudé not Hamid for attractors paper; Jaskir not Frank for OpAL*; Findling not Gershman for mPFC variability).

**GOALS.md created** — half-page north star with H1–H7 verified results table, remaining deliverable checklist, and key open question on project framing.

**SESSION_LOG.md created** with append-only protocol added to CLAUDE.md §7.

## 2026-06-19 — notes: Cools (2011, 2019); current/ synthesis files
Commit: 5c71189

- `Papers/cools_inverted-ushaped_2011.md` and `Papers/cools_chemistry_2019.md` created from PDFs with full project-note structure
- `Papers/PAPERS_INDEX.md` updated with Cools entries
- `current/implications-summary.md` — synthesis of "Implications for our model" across all 13 paper notes
- `current/memoire-gaps.md` — opus gap analysis of introduction + method drafts against the agreed outline
- `current/traditionalvsnewdopa.md` — opus synthesis: classical RPE view vs multi-dimensional modern dopamine picture, citing vault papers

## 2026-06-19 — fix: SSH auth via new ed25519 key; push backlog to Codeberg
Commit: 5c71189 (no new commit this session)

- Old id_rsa had an unknown passphrase; ksshaskpass was silently failing
- Generated new ~/.ssh/id_codeberg (ed25519, no passphrase)
- Added SSH config block: Host codeberg.org → IdentityFile ~/.ssh/id_codeberg
- Key added to Codeberg as "thinkpad"; connection verified
- Pushed all pending commits (956a7da, 5c71189, 5aee5b7) to origin main

## 2026-06-19 — fix: detach w_GD tensor before float conversion in train.py
Commit: d5e179a

- Created Python 3.12.13 venv at Code/venv/ and installed requirements
- Ran full 5000-episode experiment (seed 0): all H1–H6 pass
  - H1 acc=1.000 (p=9.3e-302), H2 hab_onset=ep1600/wgd_drop=ep2300
  - H3 deval drop: learning=0.51, maintenance=0.00
  - H4 lesions: at maintenance both GD and hab solo carry task independently
  - H5 DA-rise=+0.001 (marginal but PASS; behavioural reactivation strong, deval_sensitivity=1.0)
  - H6 decoder_acc=1.000, PR=1.0
- Fixed bug in train.py:44 — missing .detach() on w_GD tensor before float()
- Added Code/venv/, __pycache__/ to .gitignore

## 2026-06-19 — notes: diffsupervisorv1 comparison
Commit: d5e179a

- Wrote diffsupervisorv1.md comparing my implementation vs supervisor's tunl_a2c_two_area.py
- Key finding: my version is trivial in 3 compounding ways:
  (1) sample phase is scripted (agent not in control)
  (2) arm identity persists as a non-decaying one-hot in the allocentric observation
  (3) delay only 10 steps, no curriculum

## 2026-06-20 — env: replace scripted DNMTP with free-navigation TUNL-style environment
Commit: 5ed718d

- Fixed broken wikilinks in current/ (cools_2011→cools_inverted-ushaped_2011, cools_2019→cools_chemistry_2019, hamid2021a→hamid_wave-like_2021, ProjectOverview→full filename)
- Deleted empty root stubs (cools_2019.md, Schad2024.md, Tran2021.md) and duplicate Projects/dopaCTRNN/cools_inverted-ushaped_2011.md
- Pushed all vault changes to Codeberg (commits 3c8b109, d0ee410, 5ed718d)
- Replaced scripted TMazeDNMTP with free-navigation TMazeFreeNav: agent navigates all phases, 6D obs with decaying phase signal, 5 actions (WAIT added), curriculum delay (delay_start=1→delay_max=15)
- Updated model.py: GDNet and HabNet W_in now cfg.obs_dim (6) — same obs to both nets
- Updated config.py: n_gd=n_hab=256, obs_dim=6, n_actions=5, new delay/curriculum params
- Updated train.py: unified obs stream, curriculum delay advancement, 6-tuple return; expression gate now gates ONLY policy+entropy (not critic/DA penalty) after Opus review
- Updated analysis.py: unified obs stream, fixed_points navigation updated, H6 guard for empty delay states
- Updated run_experiment.py: eval_env delay sync, H7 evaluates at delay_max, --n-gd/--n-hab CLI args
- Spawned Opus code review; applied 3 fixes: expression gate scope, H7 delay, checkpoint warnings
- Launched 256-neuron 8000-episode test run (/tmp/dopa_256/), still in progress

## 2026-06-20 — perf: vectorise evaluation
Commit: 7a6ffd4

- Root cause of 8% CPU: sequential evaluate() ran 200 B=1 rollouts (BLAS-2) every 3 training iterations — 20x eval overhead
- Added evaluate_vec(): runs ceil(n/B) batches of B=32 parallel envs (BLAS-3), 11.4x faster (7.1s → 0.62s per checkpoint)
- Result: 735% CPU (7+ cores, fans now active) vs 148% before
- 256-neuron run launched at /tmp/dopa_256v3 (PID 182275)

## 2026-06-20 — perf: batch_size 32->128 + pin BLAS to physical cores
Commit: 059bb13

- CPU saturation work for Code/main on AMD Ryzen 7 8840HS (8 physical / 16 logical cores).
- Benchmarked batch_size x thread combos on _train_batch (n_gd=n_hab=256): B=32/8t=276 ep/s, B=128/8t=469, B=256/8t=540; B=64/16t=232 and B=128/16t=305 (16 threads thrash SMT siblings, LOWER throughput).
- Confirmed 2 procs x 4 threads = ~803 ep/s aggregate vs 1 proc x 8t = 465 (option-3 multiprocessing would nearly 2x but yields K independent models = algorithm change; skipped per medium-effort rule).
- Applied options 1+2: batch_size 32->128 in config.py; torch.set_num_threads pinned to physical cores (logical//2) in train.py and run_experiment.py, DOPA_NUM_THREADS override.
- v3 run had already finished training; crashed only in post-hoc sklearn decoder (one-class data, H6) — separate pre-existing issue. PID 182275 already dead.
- Launched v4 at /tmp/dopa_256v4 (PID 185241): steady-state ~740-770% CPU (~7.5/8 physical cores), ~2x throughput vs v3. Single-process ceiling; 1600% needs multiprocessing (not done).

## 2026-06-20 — fix: learning failure (sparse reward, WAIT bias, delay)
Commit: 5e329ea

- Diagnosed 0 0x0p+0ccuracy root causes: (1) WAIT cheaper than moves → policy collapsed to WAIT at START; (2) sparse reward, only +1.0 at final choice; (3) delay_start=1 put agent AT sampled arm end when choice begins → 4/5 actions immediately wrong
- Fixes: wait_cost=step_cost=0.02; junction_bonus=+0.30/arm_end_bonus=+0.20 (dense shaping); delay_start=5 (buffer to leave arm end)
- Stochastic baseline: 5% → 16%
- Result v5 run: model learns — comb/hab/gd all hit 1.00 at ep 21760 (delay=6); curriculum advanced to delay=10 by ep 32000
- w_GD: 0.80 → 0.38 (handoff happening as designed)
- Next: more episodes (60k+) needed to train through delay=10+

## 2026-06-20 — viz: step-by-step playback + heatmap overlay
Commit: d5e179a

- Rewrote make_viz.py: step slider + play/pause per panel (every timestep visible)
- Heatmap overlay (visit-frequency per cell) replaces overlapping line overlay
- w_GD timeline needle synced to step slider
- Two-panel layout by sample side preserved; fixed duplicate const{L,R} JS bug
- Ran seed=0 experiment: H1 acc=1.00 (p=9e-302), H6 decoder=1.00, H5 DA-rise FAIL
- Delay curriculum reached 10; ckpt_maint at delay 8-9

## 2026-06-20 — jury-proof figures and trajectory viewer
Commit: dc16852b6a6703c7189848f178c59da44a22970d

- Multi-agent pipeline (Opus + 2×Sonnet): Opus drafted ideal mémoire figure list; Sonnet audited figures.py and make_viz.py for scientific gaps and bugs
- fig1: vertical onset markers (hab≥0.8 vs w_GD drop) to prove emergent ordering; fig2/3: chance baselines and Δ annotations; fig4: fixed hardcoded 0.0 for intact devaluation sensitivity, added PASS/FAIL+Δ to title
- fig5: explained variance on PCA axes, system label "Habitual (DLS)", empty-data guard; fig6: jitter + empty-list guard; _save: bbox_inches="tight"
- analysis.py: h6 now returns explained_variance_ratio; run_experiment.py: passes h2 to fig1, h3 to fig4
- make_viz.py: fixed w===0 showing "?" (most important readout); fixed off-by-one in path rendering; added training epoch window (all/early/late 30
## 2026-06-20 -- jury-proof figures and trajectory viewer
Commit: dc16852b6a6703c7189848f178c59da44a22970d

- Multi-agent pipeline (Opus + 2xSonnet): Opus drafted ideal memoire figure list; Sonnet audited figures.py and make_viz.py for scientific gaps and bugs
- fig1: vertical onset markers (hab>=0.8 vs w_GD drop) to prove emergent ordering; fig2/3: chance baselines and delta annotations; fig4: fixed hardcoded 0.0 for intact devaluation sensitivity, added PASS/FAIL+delta to title
- fig5: explained variance on PCA axes, system label "Habitual (DLS)", empty-data guard; fig6: jitter + empty-list guard; _save: bbox_inches="tight"
- analysis.py: h6 now returns explained_variance_ratio; run_experiment.py: passes h2 to fig1, h3 to fig4
- make_viz.py: fixed w===0 showing "?" bug; fixed off-by-one in path rendering; added training epoch window (all/early/late 30%); untrained banner; w_GD timeline label explains it is an expression weight

## 2026-06-20 — architecture graphs, code-vs-methods diff, and branch split
Commit: 425d55f701d8c25b3d7097cb8b28f5e2fb4ad17c

- Created current/architectureGraph.md: Mermaid diagrams of DualSystemModel, GDNet, HabNet, T-maze trial flow, training loop, and evaluation pipeline
- Created current/diffCodeMethods.md: catalogued 12 discrepancies between Code/main/ and the mémoire drafts (D1 CRITICAL: methods claimed 22D/14D allocentric/egocentric split not present in code)
- Created branch 6D (from main): rewrites methods+intro to match existing code — both systems share identical 6D observation; learning rule and DA profile are the sole asymmetries
- Created branch allo-ego (from main): implements the split in code (obs_dim_hab=4, env.obs_hab() strips x/y, HabNet W_in resized, train.py and analysis.py pass separate obs2) and updates methods+intro accordingly
- Both branches pushed to Codeberg; old branch names (fix/methods-6d, feat/ego-split) deleted

## 2026-06-20 — Vectorised env rewrite landed + obs/D1-D2 walkthrough
Commit: 5d839e11e8d9d738d390b31f792b2f57e3c6c480

- Walked through 6D goal-directed vs 4D habitual obs vectors; confirmed prev-action slot is hard-zeroed in both our code and supervisor (deliberate: forces cue memory into recurrence, not action feedback).
- Documented the goal-directed double-timescale scheme (D1/phasic+fast-tau vs D2/tonic+slow-tau, aligned at n//2); flagged docstring vs code mismatch — DA gain modulates policy readout, not recurrent W or value.
- Ingested user fixes from files(3).zip into Code/main/ (environment, config, train, analysis, model, run_experiment): TMazeVecEnv batched env, parametric len_edge/difficulty grid, optional low-rank habit, GAE+return-norm, APE-decay, resumable checkpoints. TMazeFreeNav kept as wrapper.
- Committed the 6 files and pushed allo-ego (new upstream). Untracked tests/ left out: 117 failures are stale tests vs the new API (e.g. HabNet fed 6D not 4D), not regressions; 358 pass, all 6 files compile.
- Saved memory: escape whitespace in Bash paths instead of quoting.
