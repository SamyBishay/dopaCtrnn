# Visualisations

Interactive trajectory/maze visualiser for Stage-1 experiment results. It consumes the
output of `Code/main/`'s pipeline and turns it into self-contained HTML files you can
open directly in a browser, no server required. There are two tools, for two different
shapes of input:

- **`build_viz.py`** — one seed at a time. Consumes a flat `results/seed<N>/` folder
  (what `run_experiment.py` writes directly, or one arm/run's worth of seeds from a
  batch). Use this when you want the full trial player / per-seed detail for a single
  run, or when there's no batch to compare against.
- **`build_batch_viz.py`** — a whole ladder experiment at once. Consumes the richer
  `results/<exp>/<arm>/<ts>_<commit>/seed<N>/` layout that `batch_runner.py` writes
  (Step 8 of the roadmap) and produces one page that aggregates each arm's seeds and
  compares arms against each other. Use this whenever you've run a contrast (E2/E3/E4/
  E5/E8) and want to see whether the arms actually differ — that comparison is exactly
  what a per-seed page can't show you, since it only ever has one seed loaded.

The two are complementary, not a replacement of one by the other: `build_batch_viz.py`
calls into `build_viz.py`'s per-seed builder itself to produce drill-down pages linked
from the batch view, so the trial player is never duplicated, only reused.

This supersedes the earlier `make_viz.py`-based workflow (see `logs/SESSION_LOG.md`,
2026-06-20/21): that tool has been replaced by `build_viz.py` + `visualiser.html`.

## Files
- `build_viz.py` — per-seed CLI tool (see Usage below).
- `visualiser.html` — the per-seed viewer template; also works standalone via drag-and-drop.
- `build_batch_viz.py` — batch/arm-comparison CLI tool (see Usage below).
- `batch_visualiser.html` — the batch viewer template.
- `aggregate_arm.py` — cross-seed aggregation math for one arm (mean/sd per episode,
  per-hypothesis pass rates); imported by `build_batch_viz.py`, not run directly. Mirrors
  `Code/main/aggregate.py`'s `_ms()` helper but is arm-aware, since `aggregate.py` only
  knows about the old flat `results/seed<N>/` layout.
- `sample_results.json`, `sample_trajectories.json` — example fixture data (see below).

## Usage — per seed
```bash
python build_viz.py <results_dir> [--out <out_dir>]

# example, against Code/main's own results directory
python build_viz.py ../main/results
```
`results_dir` must contain `seed<N>/` subfolders. For each one that has a
`trajectories.json`, the script embeds that seed's trajectories, `results.json` (if
present — its `logs` sub-object is split out separately for the handoff chart), and any
`fig*.png` files (base64-encoded) into a copy of `visualiser.html`, replacing its
`<script id="embedded-data">` placeholder. Output goes to `<results_dir>/viz/` by
default (override with `--out`), one `viz_seed<N>.html` per seed. Seeds without
`trajectories.json` are skipped with a warning; if `results_dir` has no `seed*/`
subfolders at all, the script exits with an error.

## Usage — batch / arm comparison
```bash
python build_batch_viz.py <exp_dir> [--out <out_dir>] [--run <ts>_<commit>]

# example, against one ladder experiment's output
python build_batch_viz.py ../main/results/e4
```
`exp_dir` must be one experiment's directory as written by `batch_runner.py` — i.e. it
contains `<arm>/` subfolders, each containing one or more `<ts>_<commit>/seed<N>/` run
directories, plus (if the batch ran to completion) a `batch_summary_<ts>_<commit>.json`
at the top level. For each arm, the script:
1. Picks the most recent `<ts>_<commit>` run (lexical sort on the timestamp prefix —
   pass `--run <ts>_<commit>` to pin a specific one instead, e.g. to compare an older
   run after a re-run exists).
2. Aggregates that run's seeds with `aggregate_arm.py` (cross-seed mean/sd of the H2
   per-episode logs, plus per-hypothesis pass counts for H1/H2/H3/H5/H6).
3. Builds a per-seed drill-down page for each seed (via `build_viz.py`'s builder,
   reused as a library call — not reimplemented) into `<out_dir>/seeds/`.

It then embeds every arm's aggregate, the seed drill-down links, and the latest
`batch_summary_*.json` (if present) into a copy of `batch_visualiser.html`, replacing
its `<script id="embedded-batch-data">` placeholder. Output goes to `<exp_dir>/viz/` by
default (override with `--out`), as `batch_<exp>.html` + a `seeds/` subfolder of
per-seed pages it links to. If no arm has any usable seed data, or no `<ts>_<commit>`
run directories exist at all, the script exits with an error.

### What the batch HTML lets you do
One page per experiment, still fully self-contained (open it, no server). It shows, in
order:
- **Gate banner** — reads `all_arms_gate_clear` from `batch_summary_*.json`. If false: a
  large, hard-to-miss red banner naming exactly which arm(s) failed the H1 gate, quoting
  the roadmap's protocol (every arm must independently clear accuracy ≥ 0.80, p < 1e-3
  before a contrast is interpretable) and saying explicitly not to tune to rescue a
  failing arm. If true, just a small green pill in the header — deliberately quiet when
  there's nothing to flag. If no `batch_summary_*.json` is found at all (e.g. you're
  pointing it at a partially-run or hand-assembled directory), it says so instead of
  guessing.
- **Experiment overview table** — one row per arm: seed count, H1 gate pass count, mean
  accuracy ± sd, and pass counts for H2/H3/H5, plus how many seeds preserved the
  gate-clamp-control policy and which `<ts>_<commit>` run was used.
- **H2 — handoff timing, arm vs. arm** — combined / habitual-solo / goal-directed-solo
  accuracy and `w_GD`, one mean line per arm (colour-coded) with a ±1 sd band across that
  arm's seeds, overlaid on one chart so arms are directly comparable.
- **H5 — reactivation & gate-clamp control, arm vs. arm** — the E2/E3 crux test: per-arm
  bars (with individual seed points jittered on top) for the DA-request Δ
  (silenced − intact) and the gate-clamp control's GD-clamped accuracy, so you can see at
  a glance whether, e.g., the `expression` arm reactivates while `scheduled` doesn't.
- **H3 — devaluation dissociation, arm vs. arm** — learning-phase vs. maintenance-phase
  accuracy drop, per arm, with seed-level error bars.
- **Per-arm cards** — one card per arm with the full numeric summary (mean ± sd for every
  hypothesis) and a row of buttons, one per seed, linking to that seed's full drill-down
  page (`seeds/viz_<arm>_<seed>.html`) — the same trial player / H2 chart / H5 panel /
  static figures that `build_viz.py` produces standalone, reused rather than rebuilt.

Any panel whose relevant data is entirely absent across all arms (e.g. no `h3` anywhere)
is simply not shown, rather than rendered empty.

## What the per-seed HTML lets you do
Each generated page is fully self-contained (data is inlined, so it works with no
server — just open it in a browser). It shows:
- **Trial player** — a maze rendering with a slider/play-pause to step through a chosen
  trial. Tabs switch between training trials and final-policy (eval) trials; a
  prev/next/slider control picks which trial. The agent's path is colour-coded per step
  by `w_GD`, the goal-directed expression gain: green = habitual driving, orange =
  goal-directed driving (interpolated in between). A `w_GD` strip beneath the maze shows
  the same signal as a timeline across the trial, with phase shading (pre-sample /
  sample / delay / choice) for training trials.
- **H2 handoff chart** — combined/habitual-solo/goal-directed-solo accuracy and `w_GD`
  plotted across training episodes, with the competence-locked handoff window shaded;
  clicking a point jumps the trial player to the nearest training episode. Only shown if
  `results.json`'s `logs` are present.
- **H5 reactivation panel** — a bar chart of mean DA-request intact vs. habitual-silenced,
  plus an explicit honest verdict (clean pass / partial / not met) and, if present, the
  gate-clamp control readout. Only shown if `results.json` has an `h5` entry.
- **Supporting evidence** (collapsible) — H1 competence, H3 devaluation, H4 lesion
  double-dissociation, H6 delay-attractor PCA, whichever of these are present in
  `results.json`.
- **Static figures** (collapsible) — the embedded `fig*.png` images, or a drop zone to
  add them manually.

If no data was embedded by `build_viz.py` (the placeholder script tag is left as `{}`),
`visualiser.html` falls back to its original drag-and-drop mode: drop `trajectories.json`
(required, drives the maze player) and optionally `results.json` (adds the handoff chart
and H1/H3/H4/H5/H6 panels) onto the page, or click to choose files.

## Sample data
`sample_results.json` and `sample_trajectories.json` are standalone example/test
fixtures — neither `build_viz.py` nor `visualiser.html` reference these filenames
anywhere; there is no default/demo dataset that auto-loads. To inspect them, open
`visualiser.html` directly in a browser and drag-and-drop the two files onto the drop
zone.
