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
Commit: –

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
Commit: –

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
