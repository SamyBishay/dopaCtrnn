#!/usr/bin/env python3
"""Watch results/ and auto-rebuild batch visualisers when new seeds complete.

Polls every --poll seconds (default 30). Rebuilds any experiment that has
new results.json files since the last build. Opens the first rebuilt HTML
in the browser automatically.

Usage:
    python viz_watch.py               # watch all experiments
    python viz_watch.py e2            # watch only e2
    python viz_watch.py e2 e0 --poll 60
    python viz_watch.py --once        # build once and exit
"""
import argparse
import hashlib
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = (HERE / "../main/results").resolve()
VIZ_DIR = HERE / "out"


def _fingerprint(exp_dir: Path) -> str:
    files = sorted(exp_dir.rglob("results.json"))
    if not files:
        return ""
    parts = []
    for f in files:
        try:
            st = f.stat()
            parts.append(f"{f}:{st.st_mtime:.3f}:{st.st_size}")
        except OSError:
            pass
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _rebuild(exp_dir: Path) -> "Path | None":
    sys.path.insert(0, str(HERE))
    import importlib
    import build_batch_viz as bv
    importlib.reload(bv)
    out_dir = VIZ_DIR / exp_dir.name
    try:
        out_path = bv.build(exp_dir, out_dir, pin=None)
        return out_path
    except SystemExit:
        print(f"  [skip] {exp_dir.name}: no usable arm data (old layout or empty)", flush=True)
        return None
    except Exception as exc:
        print(f"  [warn] {exp_dir.name}: {exc}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="*",
                    help="Experiment names to watch (default: all in results/)")
    ap.add_argument("--poll", type=int, default=30, help="Poll interval in seconds (default 30)")
    ap.add_argument("--once", action="store_true", help="Build once and exit")
    ap.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = ap.parse_args()

    prints: dict[str, str] = {}
    browser_opened = False
    anything_rebuilt = False

    while True:
        if args.exps:
            exp_dirs = [RESULTS_DIR / e for e in args.exps if (RESULTS_DIR / e).is_dir()]
        else:
            if not RESULTS_DIR.exists():
                print(f"[warn] results dir not found: {RESULTS_DIR}", flush=True)
                break
            exp_dirs = sorted(
                d for d in RESULTS_DIR.iterdir()
                if d.is_dir() and d.name not in ("archive", "viz")
            )

        for exp_dir in exp_dirs:
            if not list(exp_dir.rglob("results.json")):
                continue
            fp = _fingerprint(exp_dir)
            if not fp or fp == prints.get(exp_dir.name):
                continue

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] {exp_dir.name}: changes detected — rebuilding...", flush=True)
            out_path = _rebuild(exp_dir)
            if out_path:
                prints[exp_dir.name] = fp
                print(f"  → {out_path}", flush=True)
                if not browser_opened and not args.no_browser:
                    webbrowser.open(f"file://{out_path.resolve()}")
                    browser_opened = True

        if args.once:
            if not browser_opened:
                print("No new results found — nothing rebuilt.", flush=True)
            break
        try:
            time.sleep(args.poll)
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
            break


if __name__ == "__main__":
    main()
