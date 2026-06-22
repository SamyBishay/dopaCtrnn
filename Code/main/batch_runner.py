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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent

# Each entry: flag name (None = no varying flag, just repeat the default arm),
# the arm values to sweep, and any extra config overrides every arm needs
# (e.g. E2/E3 need da_split=True for the split to be non-vacuous -- see
# IMPLEMENTATION_ROADMAP.md Step 5/§5).
EXPERIMENTS = {
    "e0": {"flag": None,           "arms": ["default"],
           "extra": {}},
    "e2": {"flag": "gate_mode",    "arms": ["expression", "scheduled"],
           "extra": {"da_split": True}},
    "e3": {"flag": "da_components", "arms": ["both", "gain_only", "weights_only"],
           "extra": {"da_split": True}},
    "e4": {"flag": "habit_rule",   "arms": ["value_free", "value_coupled"],
           "extra": {}},
    "e5": {"flag": "habit_obs",    "arms": ["position_free", "allocentric"],
           "extra": {}},
    "e8": {"flag": "hab_rank",     "arms": ["0", "1", "2", "4", "8"],
           "extra": {}},
}

CLI_FLAG = {
    "gate_mode": "--gate-mode",
    "da_components": "--da-components",
    "habit_rule": "--habit-rule",
    "habit_obs": "--habit-obs",
    "hab_rank": "--hab-rank",
}


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=HERE, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "nocommit"


def _build_cmd(exp_def, arm, seed, episodes, outdir):
    cmd = [sys.executable, str(HERE / "run_experiment.py"),
           "--seed", str(seed), "--outdir", str(outdir)]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    flag = exp_def["flag"]
    if flag is not None and arm != "default":
        cmd += [CLI_FLAG[flag], arm]
    for k, v in exp_def["extra"].items():
        if k == "da_split" and v:
            cmd.append("--da-split")
        else:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd


def _run_one(cmd, env, log_path):
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=sorted(EXPERIMENTS), help="which ladder experiment")
    ap.add_argument("--seeds", type=int, default=5,
                    help="seeds per arm (roadmap protocol: >=5)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="override cfg.episodes (smaller for a smoke run)")
    ap.add_argument("--workers", type=int, default=None,
                    help="max concurrent single-thread processes "
                         "(default: os.cpu_count(), all logical cores)")
    ap.add_argument("--root", type=str, default=str(HERE / "results"),
                    help="results root (default: Code/main/results)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, without running them")
    args = ap.parse_args()

    exp_def = EXPERIMENTS[args.exp]
    workers = args.workers or os.cpu_count() or 1
    ts = time.strftime("%Y%m%dT%H%M%S")
    commit = _git_commit()
    batch_dir = Path(args.root) / args.exp

    jobs = []  # (arm, seed, cmd, outdir, log_path)
    for arm in exp_def["arms"]:
        arm_dir = batch_dir / arm / f"{ts}_{commit}"
        for seed in range(args.seeds):
            log_path = arm_dir / f"seed{seed}_run.log"
            cmd = _build_cmd(exp_def, arm, seed, args.episodes, arm_dir)
            jobs.append((arm, seed, cmd, arm_dir, log_path))

    print(f"[{args.exp}] {len(exp_def['arms'])} arm(s) x {args.seeds} seed(s) "
          f"= {len(jobs)} job(s), {workers} concurrent worker(s) "
          f"(DOPA_NUM_THREADS=1/child)")
    print(f"  -> {batch_dir}/<arm>/{ts}_{commit}/seed<N>/")
    for arm, seed, cmd, outdir, _ in jobs:
        print(f"  [{arm} seed{seed}] {' '.join(cmd)}")
    if args.dry_run:
        return

    env = os.environ.copy()
    env["DOPA_NUM_THREADS"] = "1"

    results = {}  # (arm, seed) -> (returncode, wall_s)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for arm, seed, cmd, outdir, log_path in jobs:
            outdir.mkdir(parents=True, exist_ok=True)
        futures = {
            ex.submit(_run_one, cmd, env, log_path): (arm, seed)
            for arm, seed, cmd, outdir, log_path in jobs
        }
        for fut in as_completed(futures):
            arm, seed = futures[fut]
            rc, wall_s = fut.result()
            results[(arm, seed)] = (rc, wall_s)
            status = "ok" if rc == 0 else f"FAILED (rc={rc})"
            print(f"  done: {arm} seed{seed} -- {status} ({wall_s:.0f}s)")

    # ---- chance-guard summary: every arm must independently clear the gate ----
    summary = {"exp": args.exp, "ts": ts, "commit": commit, "seeds": args.seeds, "arms": {}}
    overall_ok = True
    for arm in exp_def["arms"]:
        arm_dir = batch_dir / arm / f"{ts}_{commit}"
        passes, accs = [], []
        for seed in range(args.seeds):
            rc, _ = results.get((arm, seed), (1, 0.0))
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
        overall_ok &= arm_ok
        summary["arms"][arm] = {"n_pass": n_pass, "n_seeds": args.seeds,
                                "accuracies": accs, "gate_clear": arm_ok}
        flag = "OK" if arm_ok else "GATE NOT CLEARED"
        print(f"  [{arm}] H1 gate: {n_pass}/{args.seeds} seeds pass -- {flag}")

    summary["all_arms_gate_clear"] = overall_ok
    with open(batch_dir / f"batch_summary_{ts}_{commit}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if not overall_ok:
        print("\nWARNING: not every arm independently cleared the H1 learning gate.\n"
              "Per protocol (IMPLEMENTATION_ROADMAP.md §4): the contrast is "
              "uninterpretable until every arm clears it on its own -- do not "
              "tune to rescue a failing arm; report the failure.")


if __name__ == "__main__":
    main()
