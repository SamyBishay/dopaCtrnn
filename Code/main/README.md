# dopaCTRNN — Stage 1

In-silico reproduction of the reversible cortico-striatal handoff (Villet et al., 2025):
control migrates from a goal-directed (mPFC/DMS) controller to a habitual (DLS) one over
training, and reverses instantly when the habitual system is silenced — the cortical
solution stays **dormant but reactivable**, not erased.

A dual-system CTRNN solves a delayed non-match-to-place (DNMTP) T-maze. Dopamine sets a
multiplicative **expression gain** on the goal-directed weights (`W_eff = f(DA)·W`); a
scalar **DA-request**, penalised to be minimised, drives an **emergent** handoff. The
habitual system learns **value-free** (action-prediction-error + intrinsic efficiency;
the task reward never enters its loss), which is what makes it structurally
devaluation-insensitive.

## Files
- `config.py` — all (pinned) hyperparameters + the OPEN #1 switch.
- `environment.py` — DNMTP T-maze; allocentric (6-d, with position) GD stream vs position-free (4-d) habitual stream.
- `model.py` — `DualSystemModel`: GD CTRNN (fast/widen + slow/deepen halves, DA gain, DA-request, critic) + habitual CTRNN (Go/NoGo) + DA-gated mixing.
- `train.py` — A2C for GD (habitual detached → no reward gradient) + value-free habitual loss; expression-gated plasticity; DA-penalty warmup/ramp.
- `analysis.py` — H1 (`h1_learning`), H2 (`h2_handoff`), H3 (`h3_devaluation`), H4 (`h4_lesions`), H5 (`h5_reactivation`, which internally calls `h5_gate_clamp_control` — see below), H6 (`h6_attractor`), attractor PCA + decoder, participation ratio, optional fixed-point finder. H7 (untrained control) is not a function here — it's the `--untrained` flag handled in `run_experiment.py`.
- `figures.py` — Figures 1–6 (`fig1_handoff` … `fig5_attractor`, `fig6_control`).
- `run_experiment.py` — one seed end-to-end → `results.json` + `trajectories.json` + figures + checkpoints.
- `batch_runner.py` — Step 8: one ladder experiment (E2–E5, E8; see `IMPLEMENTATION_ROADMAP.md` §5) across ≥5 seeds/arm as concurrent single-thread processes → `results/<exp>/<arm>/<ts>_<commit>/seed<N>/` (no-overwrite) + a `batch_summary_*.json` chance-guard.
- `aggregate.py` — combine seeds → across-seed summary + cross-seed figures.
- `run_seed.sh`, `setup_env.sh`, `submit_oar.sh` — Grid5000 (OAR) launchers, live directly in this directory (no `g5k/` subdirectory).

## Install (local)
```bash
pip install -r requirements.txt        # torch (CPU is fine), numpy, scipy, scikit-learn, matplotlib
```

## Run (local)
```bash
# one seed (trained) — full pipeline, ~1.5–2 min on a laptop CPU
python run_experiment.py --seed 0 --outdir results

# the untrained negative control for the same seed (H7)
python run_experiment.py --seed 0 --outdir results --untrained

# a few seeds, then aggregate
for s in 0 1 2 3 4; do
  python run_experiment.py --seed $s --outdir results
  python run_experiment.py --seed $s --outdir results --untrained
done
python aggregate.py --root results --out results/summary
```
Add `--fixed-points` to also run the Sussillo & Barak-style fixed-point analysis.

Other `run_experiment.py` flags (all optional, override `config.py` defaults):
`--episodes`, `--device` (default `cpu`), `--n-gd`, `--n-hab` (GD/habitual network
sizes), `--hab-rank` (0=full-rank habitual recurrent weight, default; >0=low-rank),
`--len-edge` / `--difficulty` (maze size/shape), `--gae-lambda`, `--no-ret-norm`
(disable GD return normalisation), `--ape-decay` (fade APE weight once habitual
surpasses combined; **note: `config.ape_decay` already defaults to `True`, so this flag
is currently a no-op unless you've changed the default**), `--ckpt-every N` (write a
resumable checkpoint — `train_state.pt` — every N episodes; 0/default = off).

Ladder-experiment flags (E2–E5; see `IMPLEMENTATION_ROADMAP.md` §5 for the science):
`--gate-mode {expression,scheduled}` (E2), `--da-components {both,gain_only,weights_only}`
(E3), `--habit-rule {value_free,value_coupled}` (E4), `--habit-obs {position_free,allocentric}`
(E5), `--da-split` (Step 5 scalar split — required for E2/E3 to be non-vacuous; both arms
of E2/E3 should set it). Running a single seed of one arm by hand:
```bash
python run_experiment.py --seed 0 --outdir results --gate-mode scheduled --da-split
```
To run a full ladder experiment (≥5 seeds/arm, concurrent, into the no-overwrite layout)
use `batch_runner.py` instead of calling `run_experiment.py` per arm by hand:
```bash
python batch_runner.py e2                       # 5 seeds/arm, full episodes, all logical cores
python batch_runner.py e4 --seeds 8 --episodes 50000
python batch_runner.py e5 --dry-run              # print the commands without running
```
Each child process is pinned to a single BLAS thread (`DOPA_NUM_THREADS=1`) and the pool
runs up to `--workers` (default: `os.cpu_count()`) concurrently — running many
single-threaded processes saturates every core, including the GIL-bound per-step
env/Python sections that leave a single multi-threaded run at well under 100% CPU.

## Trajectory viewer (interactive, Firefox)
Each `run_experiment.py` run also writes `seed<N>/trajectories.json` (compact: a
subsampled training-progression set + eval trials, each a cell path with per-step w_GD).
The viewer itself now lives in `../visualisations/` (`build_viz.py` + the
`visualiser.html` template), not in this directory. It builds one self-contained HTML
**per seed** (data inlined — no server needed):
```bash
python ../visualisations/build_viz.py results --out results/viz
# writes results/viz/viz_seed<N>.html (and viz_seed<N>_untrained.html) for every
# seed</N> subfolder that has a trajectories.json
# then open results/viz/viz_seed0.html in Firefox
```
`--out` is optional; it defaults to `<results_dir>/viz`. Controls pick the mode
(training progression / eval) and the trial within that file. The selected trial shows
the agent's path coloured per step by **w_GD** (teal = habitual in control, crimson =
goal-directed) with a w_GD-vs-step timeline; the overlay shows every trajectory of that
mode as thin lines shaded **white (earliest) → black (latest)** so you can watch the
route sharpen over training. Logging is subsampled (`config.traj_log_every`) so it adds
negligible storage/compute.

## Run (Grid5000)
Grid5000 uses **OAR**, not SLURM. `setup_env.sh`, `submit_oar.sh`, and `run_seed.sh`
live directly in this directory (`Code/main/`) — there is no `g5k/` subdirectory. Run
the first step on a **site frontend** (frontends have proxied internet; compute nodes do
not). CPU only — no GPU reservation needed.
```bash
# 1) once, on a frontend: build the venv on your NFS home
bash setup_env.sh

# 2) submit N seeds as an OAR array job (each array task = one seed + its control)
NSEEDS=10 WALLTIME=02:00:00 bash submit_oar.sh
oarstat -u $USER            # watch progress

# 3) when finished, aggregate
source venv/bin/activate
python aggregate.py --root results --out results/summary
```
TODO(verify): `setup_env.sh` and `run_seed.sh` both resolve their own root as
`dirname(script)/..` (i.e. they expect to live one level *below* the project root and
put the venv at `Code/venv/`), while `submit_oar.sh` resolves its root as `dirname
(script)` (i.e. `Code/main/` itself) and submits the OAR job as
`bash ${HERE}/g5k/run_seed.sh` — a `g5k/` path that does not exist now that these
scripts moved into `Code/main/` directly. This looks like a real path bug (the submitted
job would not find `run_seed.sh`), but fixing it means editing the scripts, which is out
of scope for this README-only pass — flagging instead of silently working around it.

## Outputs
Per seed under `results/seed<N>/` (trained run): `results.json`, `trajectories.json`,
`ckpt_learning.pt`, `ckpt_maintenance.pt`, and `fig1_handoff.png`–`fig5_attractor.png`.
The `--untrained` (H7) run writes to `results/seed<N>_untrained/` and skips
`fig1_handoff.png` (no training logs) and `ckpt_learning.pt`/`ckpt_maintenance.pt`
(the untrained model's freshly-initialised state dict is kept in memory for the H3–H6
analyses but is never written to disk in that branch — see `run_experiment.py`'s
`--untrained` branch). If `--ckpt-every N>0` is
passed, a resumable `train_state.pt` is also written/updated during training (deleted
implicitly only if you remove it yourself; it is not auto-cleaned on completion).
Aggregate under `results/summary/`: `summary.{json,md,csv}`, cross-seed
`fig1_crossseed.png`, and `fig6_control.png` (trained vs untrained).

## Verified Stage-1 results (default config, single seed)
**These numbers are from a single earlier run** (see `logs/SESSION_LOG.md`,
2026-06-19/20 entries) and are reported as-is per the project's no-rounding-up rule —
they are not a guarantee of what the *current* config defaults will reproduce on a
fresh run, and later sessions changed the environment/training code after this run was
recorded. Re-run `run_experiment.py` and check `results.json` / `aggregate.py`'s
`summary.md` before citing a number in the mémoire.

| hypothesis | result |
|---|---|
| H1 learning (≥80%, binomial) | combined accuracy → 1.00 |
| H2 emergent handoff | habitual competence (~ep1500) precedes the w_GD drop (~ep2500); PASS |
| H3 devaluation dissociation | learning-phase drop ≈ 0.5, maintenance drop ≈ 0.0; PASS |
| H5 reactivation (falsification) | silencing the habitual system restores accuracy to 1.00; PASS |
| H6 working-memory attractor | delay-period arm decoder = 1.00; PASS |
| H7 untrained control | ≈ chance |

Note on H5: behavioural reactivation is strong, but the **DA-request rise is marginal**
(the reactivation works because the goal-directed policy, preserved intact, dominates the
choice once the habitual system is removed). Report this honestly.

`analysis.py` also defines `h5_gate_clamp_control(model, state_maint, env, cfg)`, a
negative control for exactly this ambiguity: it force-clamps the gate open (`w_gd=1.0`,
GD acts alone) on the maintenance-checkpoint model and checks accuracy stays ≥0.65. It
is **not run as a separate top-level hypothesis** — `h5_reactivation` calls it
internally and nests its output under `results["h5"]["gate_clamp_control"]` in
`results.json`. No figure or `aggregate.py` summary field surfaces it; reading it
requires opening `results.json` directly. The table above does not report its result.

## Key design notes
- **Devaluation** is operationalised as removing the goal-directed motivational drive
  (`mot → 0`), reflecting the value-sensitivity of goal-directed control; the value-free
  habitual system is unaffected — hence the dissociation.
- **Expression-gated plasticity**: `train.py` scales the GD **policy + entropy** loss by
  a single batch-averaged, detached gate value (`expr_gate`, the mean `w_GD` over the
  iteration) before backprop — when the average gate is low, the policy gradient
  shrinks, so its solution is largely preserved (dormant but reactivable), consistent
  with `W_eff = f(DA)·W`. The critic and DA-penalty loss terms are **not** scaled by
  this gate. Without the gating term entirely, the dormant policy would be overwritten
  and H5 would be expected to fail.
- **Mixed time constants** (fast/widen units track phasic DA-request, slow/deepen units
  track the tonic low-pass — a timescale split, not a fixed D1/D2 receptor-affinity
  direction; see `model.py` `GDNet.step`) give both fast decision dynamics and working
  memory that survives the delay.
- **Hyperparameters are pinned by us**, not inherited — tune freely; the protocol treats
  a learning failure as a tuning problem, not a scientific one.

## OPEN QUESTION #1 — DA-request training signal
`config.da_request_training` documents how the DA-request is intended to be trained:
- `"a2c_coupled"` (default): the DA-request is shaped by the A2C advantage
  (reward-coupled, via the expression-gated GD loss in `train.py`). Carries the
  circularity risk discussed in the mémoire.
- `"local_pe"`: train it from a purely local prediction error instead. Specifying that
  error is the open design decision.

TODO(verify): the field is currently a **config stub only** — `train.py` and `model.py`
never read `cfg.da_request_training` at all (there is no branch on its value anywhere
in the codebase, and no `NotImplementedError` is raised for `"local_pe"`). In practice
the DA-request is *always* trained the `"a2c_coupled"` way regardless of what this field
is set to; setting it to `"local_pe"` currently does nothing. If the mémoire describes
this as an implemented switch, that needs the same correction.
