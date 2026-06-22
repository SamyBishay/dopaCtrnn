#!/usr/bin/env python3
"""Build one self-contained visualiser HTML per experiment in a results batch.

Each seed<N>/ in a results batch (results.json, trajectories.json, fig*.png)
becomes its own viz_seed<N>.html with all data embedded inline — no drag-and-
drop needed, just double-click / open in Firefox.

Usage:
  python build_viz.py <results_dir> [--out <out_dir>]

  python build_viz.py ../main/results
  python build_viz.py /tmp/dopa_test_results --out /tmp/dopa_test_results/viz
"""
import argparse
import base64
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "visualiser.html"
EMBED_RE = re.compile(
    r'(<script id="embedded-data" type="application/json">)(.*?)(</script>)',
    re.DOTALL,
)


def build_one(seed_dir: Path, out_path: Path) -> bool:
    traj_path = seed_dir / "trajectories.json"
    if not traj_path.exists():
        return False
    traj = json.loads(traj_path.read_text())

    results_path = seed_dir / "results.json"
    logs = None
    results = None
    if results_path.exists():
        results = json.loads(results_path.read_text())
        logs = results.get("logs")

    figs = {}
    for png in sorted(seed_dir.glob("fig*.png")):
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        figs[png.name] = f"data:image/png;base64,{b64}"

    # `results` carries the per-hypothesis dicts (h1..h6) for the supporting-
    # evidence panels; `logs` is duplicated out of it for the handoff chart so
    # the drag-and-drop fallback (which loads results.json directly) still works.
    payload = {"traj": traj, "logs": logs, "results": results, "figs": figs}
    template = TEMPLATE.read_text()
    replacement = (
        '<script id="embedded-data" type="application/json">'
        + json.dumps(payload, default=float)
        + "</script>"
    )
    # Use a replacement function so backslash sequences in the JSON (e.g. —)
    # are inserted literally rather than interpreted as regex escapes.
    html, n = EMBED_RE.subn(lambda _m: replacement, template, count=1)
    if n != 1:
        raise RuntimeError("could not find embedded-data placeholder in template")

    out_path.write_text(html)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path,
                     help="batch dir containing seed<N>/ subfolders")
    ap.add_argument("--out", type=Path, default=None,
                     help="output dir for the per-experiment HTML files "
                          "(default: <results_dir>/viz)")
    args = ap.parse_args()

    if not TEMPLATE.exists():
        sys.exit(f"template not found: {TEMPLATE}")

    seed_dirs = sorted(
        p for p in args.results_dir.iterdir()
        if p.is_dir() and p.name.startswith("seed")
    )
    if not seed_dirs:
        sys.exit(f"no seed*/ subfolders found in {args.results_dir}")

    out_dir = args.out or (args.results_dir / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    built = []
    for seed_dir in seed_dirs:
        out_path = out_dir / f"viz_{seed_dir.name}.html"
        if build_one(seed_dir, out_path):
            built.append(out_path)
            print(f"  {seed_dir.name} -> {out_path}")
        else:
            print(f"  {seed_dir.name}: skipped (no trajectories.json)")

    print(f"\n{len(built)} file(s) written to {out_dir}")


if __name__ == "__main__":
    main()
