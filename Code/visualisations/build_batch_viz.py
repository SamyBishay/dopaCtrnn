#!/usr/bin/env python3
"""Build one self-contained batch-comparison HTML for a ladder experiment.

Where build_viz.py turns a flat results/seed<N>/ directory into one viewer
per seed, this tool walks the richer batch_runner.py layout --

  results/<exp>/<arm>/<ts>_<commit>/seed<N>/{results.json, trajectories.json, ...}
  results/<exp>/batch_summary_<ts>_<commit>.json

-- aggregates each arm's seeds (aggregate_arm.py), and embeds the per-arm
aggregates + the batch_summary gate-clear status into a copy of
batch_visualiser.html: one page that shows every arm of the experiment next
to each other, with seed variance visible, plus the gate-not-cleared banner
from batch_runner.py's chance-guard. It also builds (by shelling out to
build_viz.py) one per-seed viz_seed<N>.html per seed and links to them, so
the batch view can drill down into the existing trial-player/H2/H5 page
without rebuilding any of that UI.

Usage:
  python build_batch_viz.py <exp_dir> [--out <out_dir>] [--run <ts>_<commit>]

  python build_batch_viz.py ../main/results/e4
  python build_batch_viz.py /tmp/viz_dev/e4 --out /tmp/viz_dev/e4/viz

If an arm has more than one <ts>_<commit> run, the most recent (by the
timestamp prefix, which sorts lexically) is used by default; pass --run to
pin a specific one explicitly (applied to every arm that has it).
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "batch_visualiser.html"
EMBED_RE = re.compile(
    r'(<script id="embedded-batch-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)

sys.path.insert(0, str(HERE))
from aggregate_arm import aggregate_arm  # noqa: E402


def _json_safe(x):
    """Recursively replace non-finite floats (NaN/Inf) with None.

    Belt-and-suspenders final pass before json.dumps: aggregate_arm.py already
    sanitizes the values it computes, but this catches anything that slips
    through unchanged from a seed's results.json (Python's json module emits
    the non-standard NaN/Infinity literals; strict browser JSON.parse rejects
    them -- see the EMBED_RE replacement below for the matching encoding fix).
    """
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _latest_run_dir(arm_dir: Path, pin: str | None):
    """Pick the <ts>_<commit> subdir to use for this arm."""
    run_dirs = sorted(p for p in arm_dir.iterdir() if p.is_dir())
    if not run_dirs:
        return None
    if pin is not None:
        for p in run_dirs:
            if p.name == pin:
                return p
        return None
    return run_dirs[-1]  # ts prefix sorts lexically -> last is most recent


def _latest_batch_summary(exp_dir: Path, pin: str | None):
    summaries = sorted(exp_dir.glob("batch_summary_*.json"))
    if not summaries:
        return None
    if pin is not None:
        for p in summaries:
            if p.name == f"batch_summary_{pin}.json":
                return json.loads(p.read_text())
        return None
    return json.loads(summaries[-1].read_text())


def build(exp_dir: Path, out_dir: Path, pin: str | None):
    exp_name = exp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_viz_dir = out_dir / "seeds"
    seed_viz_dir.mkdir(parents=True, exist_ok=True)
    out_dir_resolved = out_dir.resolve()

    # Exclude the output dir itself (commonly <exp_dir>/viz, a sibling of the
    # real arm/ subfolders when --out isn't given a separate location) so a
    # second run doesn't try to treat its own previous output as an arm.
    arm_dirs = sorted(
        p for p in exp_dir.iterdir()
        if p.is_dir() and p.resolve() != out_dir_resolved
    )
    if not arm_dirs:
        sys.exit(f"no arm/ subfolders found in {exp_dir}")

    arms = {}
    run_used = {}
    for arm_dir in arm_dirs:
        run_dir = _latest_run_dir(arm_dir, pin)
        if run_dir is None:
            print(f"  {arm_dir.name}: skipped (no <ts>_<commit> run dir found)")
            continue
        agg = aggregate_arm(run_dir, arm_dir.name)
        if agg is None:
            print(f"  {arm_dir.name}/{run_dir.name}: skipped (no seed results.json)")
            continue
        run_used[arm_dir.name] = run_dir.name

        # Build/link the per-seed drill-down viewers for this arm's seeds,
        # reusing build_viz.py rather than re-implementing the trial player.
        seed_links = {}
        for seed_dir_name in agg["seed_dirs"]:
            seed_dir = run_dir / seed_dir_name
            out_html = seed_viz_dir / f"viz_{arm_dir.name}_{seed_dir_name}.html"
            built = _build_one_seed_viz(seed_dir, out_html)
            if built:
                seed_links[seed_dir_name] = f"seeds/{out_html.name}"
        agg["seed_links"] = seed_links
        arms[arm_dir.name] = agg
        print(f"  arm {arm_dir.name}: {agg['n_seeds']} seed(s) from {run_dir.name}, "
              f"{len(seed_links)} drill-down page(s)")

    if not arms:
        sys.exit(f"no usable arm data found under {exp_dir}")

    batch_summary = _latest_batch_summary(exp_dir, pin)

    payload = {"exp": exp_name, "arms": arms, "batch_summary": batch_summary,
               "run_used": run_used}
    payload = _json_safe(payload)

    if not TEMPLATE.exists():
        sys.exit(f"template not found: {TEMPLATE}")
    template = TEMPLATE.read_text()
    replacement = (
        '<script id="embedded-batch-data" type="application/json">'
        + json.dumps(payload, default=float)
        + "</script>"
    )
    html, n = EMBED_RE.subn(lambda _m: replacement, template, count=1)
    if n != 1:
        raise RuntimeError("could not find embedded-batch-data placeholder in template")

    out_path = out_dir / f"batch_{exp_name}.html"
    out_path.write_text(html)
    return out_path


def _build_one_seed_viz(seed_dir: Path, out_path: Path) -> bool:
    """Shell out to build_viz.py's build_one() for a single seed dir."""
    sys.path.insert(0, str(HERE))
    import build_viz  # noqa: E402
    try:
        return build_viz.build_one(seed_dir, out_path)
    except Exception as e:
        print(f"    warning: could not build drill-down viz for {seed_dir}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", type=Path,
                     help="experiment dir containing arm/ subfolders, "
                          "e.g. results/e4")
    ap.add_argument("--out", type=Path, default=None,
                     help="output dir (default: <exp_dir>/viz)")
    ap.add_argument("--run", type=str, default=None,
                     help="pin a specific <ts>_<commit> run instead of the "
                          "latest one per arm")
    args = ap.parse_args()

    out_dir = args.out or (args.exp_dir / "viz")
    out_path = build(args.exp_dir, out_dir, args.run)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
