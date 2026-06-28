#!/usr/bin/env python3
"""Live watcher for experiment results. Polls results/<exp>/ every 30 s and
prints a plain-text table: exp / arm / seeds-done / mean-acc±std / hab_onset
/ H1-H6 pass-fail. Works over SSH, no GUI required.

Usage:
    python watch_results.py                  # watch all experiments
    python watch_results.py e2               # watch only e2
    python watch_results.py e2 e0            # watch e2 and e0
    python watch_results.py --once           # print once and exit
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
POLL_INTERVAL = 30  # seconds


def _h_pass(results, key):
    h = results.get(key)
    if h is None:
        return "?"
    p = h.get("pass")
    if p is True:
        return "✓"
    if p is False:
        return "✗"
    return "?"


def _acc(results):
    h1 = results.get("h1", {})
    return h1.get("accuracy", float("nan"))


def _hab_onset(results):
    h2 = results.get("h2", {})
    v = h2.get("hab_onset_episode")
    return v if v is not None else "—"


def collect_seed_results(exp_dir: Path):
    """Collect results.json from all seed dirs under exp/<arm>/<ts>/seed<N>/."""
    rows = {}  # arm -> list of result dicts
    for arm_dir in sorted(exp_dir.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm = arm_dir.name
        for ts_dir in sorted(arm_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            for seed_dir in sorted(ts_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                rjson = seed_dir / "results.json"
                if rjson.exists():
                    try:
                        data = json.loads(rjson.read_text())
                        rows.setdefault(arm, []).append(data)
                    except (json.JSONDecodeError, OSError):
                        pass
    return rows


def _mean_std(vals):
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, float("nan")
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var)


def render_table(exp_dirs):
    lines = []
    header = f"{'Exp':<6}  {'Arm':<14}  {'Seeds':>5}  {'Acc mean±std':>16}  {'hab_onset':>10}  H1  H2  H3  H5  H6"
    lines.append(header)
    lines.append("-" * len(header))

    for exp_dir in sorted(exp_dirs):
        exp = exp_dir.name
        rows_by_arm = collect_seed_results(exp_dir)
        if not rows_by_arm:
            lines.append(f"{exp:<6}  (no results yet)")
            continue
        for arm, results_list in sorted(rows_by_arm.items()):
            n = len(results_list)
            accs = [_acc(r) for r in results_list if not math.isnan(_acc(r))]
            mu, sd = _mean_std(accs)
            acc_str = f"{mu:.3f}±{sd:.3f}" if not math.isnan(mu) else "—"
            onsets = [_hab_onset(r) for r in results_list if isinstance(_hab_onset(r), int)]
            onset_str = str(int(sum(onsets) / len(onsets))) if onsets else "—"
            # H pass/fail from first seed for now (common across seeds)
            r0 = results_list[0]
            h1 = _h_pass(r0, "h1")
            h2 = _h_pass(r0, "h2")
            h3 = _h_pass(r0, "h3")
            h5 = _h_pass(r0, "h5")
            h6 = _h_pass(r0, "h6")
            lines.append(
                f"{exp:<6}  {arm:<14}  {n:>5}  {acc_str:>16}  {onset_str:>10}"
                f"  {h1:>2}  {h2:>2}  {h3:>2}  {h5:>2}  {h6:>2}"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exps", nargs="*", help="Experiment names to watch (default: all)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print(f"Results directory not found: {RESULTS_DIR}")
        return

    while True:
        if args.exps:
            exp_dirs = [RESULTS_DIR / e for e in args.exps if (RESULTS_DIR / e).is_dir()]
        else:
            exp_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name != "archive"]

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n=== {ts} ===")
        print(render_table(exp_dirs))

        if args.once:
            break
        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
