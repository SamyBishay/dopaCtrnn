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
- `environment.py` — DNMTP T-maze; allocentric (22-d) vs egocentric (14-d) streams.
- `model.py` — `DualSystemModel`: GD CTRNN (D1/phasic + D2/tonic, DA gain, DA-request, critic) + habitual CTRNN (Go/NoGo) + DA-gated mixing.
- `train.py` — A2C for GD (habitual detached → no reward gradient) + value-free habitual loss; expression-gated plasticity; DA-penalty warmup/ramp.
- `analysis.py` — H1–H7, attractor PCA + decoder, participation ratio, optional fixed-point finder.
- `figures.py` — Figures 1–6.
- `run_experiment.py` — one seed end-to-end → `results.json` + figures + checkpoints.
- `aggregate.py` — combine seeds → across-seed summary + cross-seed figures.
- `g5k/` — Grid5000 launchers (OAR).

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

## Trajectory viewer (interactive, Firefox)
Each `run_experiment.py` run also writes `seed<N>/trajectories.json` (compact: a
subsampled training-progression set + eval trials, each a cell path with per-step w_GD).
Build a single self-contained HTML (data inlined — no server needed):
```bash
python make_viz.py --root results --out results/maze_viz.html
# then open results/maze_viz.html in Firefox
```
Controls pick the experiment (seed), the mode (training progression / eval), and the
trial. The selected trial shows the agent's path coloured per step by **w_GD** (teal =
habitual in control, crimson = goal-directed) with a w_GD-vs-step timeline; the overlay
shows every trajectory of that mode as thin lines shaded **white (earliest) → black
(latest)** so you can watch the route sharpen over training. Logging is subsampled
(`config.traj_log_every`) so it adds negligible storage/compute.

## Run (Grid5000)
Grid5000 uses **OAR**, not SLURM. Run the first step on a **site frontend** (frontends
have proxied internet; compute nodes do not). CPU only — no GPU reservation needed.
```bash
# 1) once, on a frontend: build the venv on your NFS home
bash g5k/setup_env.sh

# 2) submit N seeds as an OAR array job (each array task = one seed + its control)
NSEEDS=10 WALLTIME=02:00:00 bash g5k/submit_oar.sh
oarstat -u $USER            # watch progress

# 3) when finished, aggregate
source venv/bin/activate
python aggregate.py --root results --out results/summary
```

## Outputs
Per seed under `results/seed<N>/`: `results.json`, `ckpt_learning.pt`,
`ckpt_maintenance.pt`, and `fig1`–`fig5`. Aggregate under `results/summary/`:
`summary.{json,md,csv}`, cross-seed Fig 1, and Fig 6 (trained vs untrained).

## Verified Stage-1 results (default config, single seed)
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

## Key design notes
- **Devaluation** is operationalised as removing the goal-directed motivational drive
  (`mot → 0`), reflecting the value-sensitivity of goal-directed control; the value-free
  habitual system is unaffected — hence the dissociation.
- **Expression-gated plasticity**: when the GD system is not expressed (`w_GD` low) it is
  not updated, so its solution is preserved (dormant but reactivable), consistent with
  `W_eff = f(DA)·W`. Without this the dormant policy is erased and H5 fails.
- **Mixed time constants** (D1/phasic fast, D2/tonic slow) give both fast decision
  dynamics and working memory that survives the delay.
- **Hyperparameters are pinned by us**, not inherited — tune freely; the protocol treats
  a learning failure as a tuning problem, not a scientific one.

## OPEN QUESTION #1 — DA-request training signal
`config.da_request_training` selects how the DA-request is trained:
- `"a2c_coupled"` (default, implemented): the DA-request is shaped by the A2C advantage
  (reward-coupled). Carries the circularity risk discussed in the mémoire.
- `"local_pe"` (**intentionally not implemented**): train it from a purely local
  prediction error instead. Specifying that error is the open design decision; the code
  raises `NotImplementedError` rather than choosing it for you.
