"""Step 8 — batch runner: run one ladder experiment (a flag-pair/triple) across
>=5 seeds per arm as concurrent single-thread processes, into the no-overwrite
results/<exp>/<arm>/<ts>_<commit>/seed<N>/ layout, with a chance-guard summary.

  python batch_runner.py e2                     # 5 seeds/arm, full default episodes
  python batch_runner.py e4 --seeds 8 --episodes 50000
  python batch_runner.py e5 --workers 6 --dry-run

CPU note: each child process is pinned to a single BLAS thread
(DOPA_NUM_THREADS=1) and the pool runs up to --workers children concurrently
(default: all logical cores). A single multi-threaded run only saturates CPU
in bursts (BLAS matmuls are multi-threaded; the per-step Python/env loop
in between is single-threaded and GIL-bound), which is why one run at a time
was observed at ~60% utilisation. Many single-threaded processes keep every
core busy on the GIL-bound sections too, since they belong to independent
interpreters.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

# Each entry: flag name (None = no varying flag, just repeat the default arm),
# the arm values to sweep, and any extra config overrides every arm needs
# (e.g. E2/E3 need da_split=True for the split to be non-vacuous -- see
# IMPLEMENTATION_ROADMAP.md Step 5/§5).
EXPERIMENTS = {
    "e0": {"flag": None,           "arms": ["default"],
           "extra": {}},
    "e2": {"flag": None,           "arms": ["main"],
           "extra": {}},
    "e2_ablation": {"flag": None,  "arms": ["freeze_hab"],
                    "extra": {"freeze_hab": True}},
    "e3": {"flag": "da_components", "arms": ["both", "gain_only", "weights_only"],
           "extra": {"da_split": True}},
    "e4": {"flag": "habit_rule",   "arms": ["value_free", "value_coupled"],
           "extra": {}},
    "e5": {"flag": "habit_obs",    "arms": ["position_free", "allocentric"],
           "extra": {}},
    "e8": {"flag": "hab_rank",     "arms": ["0", "1", "2", "4", "8"],
           "extra": {}},
    "e6": {"flag": None, "arms": ["uniform_tau"],
           "extra": {"da_tau": True, "tau_mode": "uniform"}},
    "e7": {"flag": None, "arms": ["dual_tau"],
           "extra": {"da_tau": True}},
    "e9": {"flag": None, "arms": ["naude_full"],
           "extra": {"da_gain_mode": "recurrent", "da_tau": True}},
}

CLI_FLAG = {
    "gate_mode": "--gate-mode",
    "da_components": "--da-components",
    "habit_rule": "--habit-rule",
    "habit_obs": "--habit-obs",
    "hab_rank": "--hab-rank",
    "da_gain_mode": "--da-gain-mode",
    "tau_mode":     "--tau-mode",
    "da_tau":       "--da-tau",
    "freeze_hab":   "--freeze-hab",
}


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=HERE, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "nocommit"


def _build_cmd(exp_def, arm, seed, episodes, outdir, device=None):
    cmd = [sys.executable, str(HERE / "run_experiment.py"),
           "--seed", str(seed), "--outdir", str(outdir)]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    if device is not None:
        cmd += ["--device", device]
    flag = exp_def["flag"]
    if flag is not None and arm != "default":
        cmd += [CLI_FLAG[flag], arm]
    for k, v in exp_def["extra"].items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k.replace('_', '-')}")
            # False bool flags: omit (don't pass --no-X)
        else:
            cli_k = CLI_FLAG.get(k, f"--{k.replace('_', '-')}")
            cmd += [cli_k, str(v)]
    return cmd


def _available_mem_gb():
    """Free+reclaimable RAM in GB, from /proc/meminfo (Linux). Returns inf elsewhere."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=sorted(EXPERIMENTS) + ["all"],
                    help="which ladder experiment, or 'all' to run every experiment "
                         "in one memory-gated queue")
    ap.add_argument("--seeds", type=int, default=5,
                    help="seeds per arm (roadmap protocol: >=5)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override cfg.episodes (smaller for a smoke run)")
    ap.add_argument("--workers", type=int, default=None,
                    help="max concurrent single-thread processes "
                         "(default: os.cpu_count(), all logical cores)")
    ap.add_argument("--root", type=str, default=str(HERE / "results"),
                    help="results root (default: Code/main/results)")
    ap.add_argument("--device", type=str, default=None,
                    help="device passed to run_experiment.py (e.g. cuda, cpu)")
    ap.add_argument("--mem-reserve", type=float, default=2.0,
                    help="minimum free RAM (GB) to keep before launching the next job "
                         "(default: 2.0); set 0 to disable memory-aware throttling")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, without running them")
    ap.add_argument("--no-open", action="store_true",
                    help="skip building the visualiser and launching Firefox")
    args = ap.parse_args()

    exps = list(EXPERIMENTS.keys()) if args.exp == "all" else [args.exp]
    workers = args.workers or os.cpu_count() or 1
    threads_per_worker = 1
    ts = time.strftime("%Y%m%dT%H%M%S")
    commit = _git_commit()

    jobs = []  # (exp_name, arm, seed, cmd, outdir, log_path)
    for exp_name in exps:
        exp_def = EXPERIMENTS[exp_name]
        batch_dir = Path(args.root) / exp_name
        for arm in exp_def["arms"]:
            arm_dir = batch_dir / arm / f"{ts}_{commit}"
            for seed in range(args.seeds):
                log_path = arm_dir / f"seed{seed}_run.log"
                cmd = _build_cmd(exp_def, arm, seed, args.episodes, arm_dir, args.device)
                jobs.append((exp_name, arm, seed, cmd, arm_dir, log_path))

    print(f"[{args.exp}] {len(jobs)} job(s) across {len(exps)} experiment(s), "
          f"max {workers} concurrent, >={args.mem_reserve:.1f}GB free required "
          f"(DOPA_NUM_THREADS={threads_per_worker}/child)")
    for exp_name, arm, seed, cmd, outdir, _ in jobs:
        print(f"  [{exp_name}/{arm} seed{seed}] {' '.join(cmd)}")
    if args.dry_run:
        return

    env = os.environ.copy()
    env["DOPA_NUM_THREADS"] = str(threads_per_worker)

    results = {}  # (exp_name, arm, seed) -> (returncode, wall_s)
    active = []   # list of (exp_name, arm, seed, Popen, t0, log_fh)

    def _reap(active):
        still = []
        for e, a, s, proc, t0, fh in active:
            rc = proc.poll()
            if rc is None:
                still.append((e, a, s, proc, t0, fh))
            else:
                fh.close()
                wall_s = time.time() - t0
                results[(e, a, s)] = (rc, wall_s)
                status = "ok" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  done: {e}/{a} seed{s} -- {status} ({wall_s:.0f}s)", flush=True)
        return still

    for exp_name, arm, seed, cmd, outdir, log_path in jobs:
        outdir.mkdir(parents=True, exist_ok=True)
        # Wait until both a worker slot and enough free RAM are available.
        while True:
            active = _reap(active)
            slots_ok = len(active) < workers
            mem_ok = _available_mem_gb() >= args.mem_reserve
            if slots_ok and mem_ok:
                break
            time.sleep(0.5)
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        active.append((exp_name, arm, seed, proc, time.time(), log_fh))
        print(f"  started: {exp_name}/{arm} seed{seed} "
              f"[{len(active)} active, {_available_mem_gb():.1f}GB free]", flush=True)

    # Drain remaining jobs.
    while active:
        active = _reap(active)
        if active:
            time.sleep(0.5)

    # ---- chance-guard summary: every arm must independently clear the gate ----
    overall_ok = True
    for exp_name in exps:
        exp_def = EXPERIMENTS[exp_name]
        batch_dir = Path(args.root) / exp_name
        summary = {"exp": exp_name, "ts": ts, "commit": commit,
                   "seeds": args.seeds, "arms": {}}
        exp_ok = True
        for arm in exp_def["arms"]:
            arm_dir = batch_dir / arm / f"{ts}_{commit}"
            passes, accs = [], []
            for seed in range(args.seeds):
                rc, _ = results.get((exp_name, arm, seed), (1, 0.0))
                rfile = arm_dir / f"seed{seed}" / "results.json"
                if rc == 0 and rfile.exists():
                    with open(rfile) as f:
                        r = json.load(f)
                    passes.append(bool(r["h1"]["pass"]))
                    accs.append(float(r["h1"]["accuracy"]))
                else:
                    passes.append(False)
                    accs.append(None)
            n_pass = sum(passes)
            arm_ok = n_pass == args.seeds
            exp_ok &= arm_ok
            summary["arms"][arm] = {"n_pass": n_pass, "n_seeds": args.seeds,
                                    "accuracies": accs, "gate_clear": arm_ok}
            flag = "OK" if arm_ok else "GATE NOT CLEARED"
            print(f"  [{exp_name}/{arm}] H1 gate: {n_pass}/{args.seeds} -- {flag}")

        overall_ok &= exp_ok
        summary["all_arms_gate_clear"] = exp_ok
        with open(batch_dir / f"batch_summary_{ts}_{commit}.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

    if not args.no_open and not args.dry_run:
        viz_script = HERE.parent / "visualisations" / "build_viz.py"
        for exp_name in exps:
            exp_def = EXPERIMENTS[exp_name]
            batch_dir = Path(args.root) / exp_name
            for arm in exp_def["arms"]:
                arm_dir = batch_dir / arm / f"{ts}_{commit}"
                print(f"  building visualiser for {exp_name}/{arm} ...", flush=True)
                subprocess.run([sys.executable, str(viz_script), str(arm_dir)], check=False)
                html = arm_dir / "seed0" / "viz_seed0.html"
                if not html.exists():
                    html = next(arm_dir.glob("**/viz_seed*.html"), None)
                if html:
                    print(f"  opening {html} in Firefox", flush=True)
                    subprocess.Popen(["firefox", str(html)])

    if not overall_ok:
        print("\nWARNING: not every arm independently cleared the H1 learning gate.\n"
              "Per protocol (IMPLEMENTATION_ROADMAP.md §4): the contrast is "
              "uninterpretable until every arm clears it on its own -- do not "
              "tune to rescue a failing arm; report the failure.")


if __name__ == "__main__":
    main()
