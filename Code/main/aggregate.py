"""Aggregate per-seed results into across-seed summaries and figures.

  python aggregate.py --root results --out results/summary
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figures as F


def load(root):
    trained, untrained = [], []
    for path in sorted(glob.glob(os.path.join(root, "seed*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        (untrained if r.get("untrained") else trained).append(r)
    return trained, untrained


def _ms(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results")
    ap.add_argument("--out", type=str, default="results/summary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    trained, untrained = load(args.root)
    if not trained:
        print("no trained seeds found under", args.root); return

    h1 = [r["h1"]["accuracy"] for r in trained]
    h1_un = [r["h1"]["accuracy"] for r in untrained]
    hab_onset = [r["h2"]["hab_onset_episode"] for r in trained if "h2" in r]
    wgd_onset = [r["h2"]["wgd_drop_onset_episode"] for r in trained if "h2" in r]
    h5_delta = [r["h5"]["da_request_delta"] for r in trained]
    h5_pass = [bool(r["h5"]["pass"]) for r in trained]
    h6_dec = [r["h6"]["decoder_acc"] for r in trained]
    h6_dec_un = [r["h6"]["decoder_acc"] for r in untrained]
    dev_learn = [r["h3"]["learning"]["drop"] for r in trained]
    dev_maint = [r["h3"]["maintenance"]["drop"] for r in trained]

    summary = {
        "n_trained": len(trained), "n_untrained": len(untrained),
        "H1_accuracy_mean_sd": _ms(h1), "H1_pass_rate": float(np.mean([r["h1"]["pass"] for r in trained])),
        "H2_hab_onset_mean_sd": _ms(hab_onset), "H2_wgd_drop_onset_mean_sd": _ms(wgd_onset),
        "H3_deval_drop_learning_mean_sd": _ms(dev_learn),
        "H3_deval_drop_maintenance_mean_sd": _ms(dev_maint),
        "H5_DA_delta_mean_sd": _ms(h5_delta), "H5_pass_rate": float(np.mean(h5_pass)),
        "H6_decoder_mean_sd": _ms(h6_dec),
        "H7_trained_acc_mean_sd": _ms(h1), "H7_untrained_acc_mean_sd": _ms(h1_un),
        "H7_untrained_decoder_mean_sd": _ms(h6_dec_un),
        "trained_acc": h1, "untrained_acc": h1_un,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- markdown table ----
    rows = [
        ("H1 learning (acc)", _fmt(_ms(h1)), f"{summary['H1_pass_rate']*100:.0f}% seeds pass"),
        ("H2 habitual onset (ep)", _fmt(_ms(hab_onset)), ""),
        ("H2 w_GD drop onset (ep)", _fmt(_ms(wgd_onset)), "should be >= habitual onset"),
        ("H3 deval drop, learning", _fmt(_ms(dev_learn)), "expect large"),
        ("H3 deval drop, maintenance", _fmt(_ms(dev_maint)), "expect ~0"),
        ("H5 DA-request rise", _fmt(_ms(h5_delta)), f"{np.mean(h5_pass)*100:.0f}% pass (must be >0)"),
        ("H6 WM decoder (acc)", _fmt(_ms(h6_dec)), "target >= 0.90"),
        ("H7 untrained (acc)", _fmt(_ms(h1_un)), "control, expect ~0.5"),
    ]
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("| metric | mean +/- sd | note |\n|---|---|---|\n")
        for a, b, c in rows:
            f.write(f"| {a} | {b} | {c} |\n")
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "mean", "sd", "note"])
        for a, b, c in rows:
            m, s = _ms_raw(b)
            w.writerow([a, m, s, c])

    # ---- cross-seed figures ----
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for r in trained:
        lg = r.get("logs")
        if lg:
            ax.plot(lg["episode"], lg["combined_acc"], color="tab:blue", alpha=.35)
            ax.plot(lg["episode"], lg["hab_solo_acc"], color="tab:orange", alpha=.35)
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_xlabel("episode"); ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Cross-seed handoff (blue=combined, orange=habitual solo)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "fig1_crossseed.png"), dpi=140)
    plt.close(fig)

    F.fig6_control(summary, args.out)
    print("wrote summary + figures to", args.out)


def _fmt(ms):
    m, s = ms
    return "n/a" if m is None else f"{m:.3g} +/- {s:.2g}"


def _ms_raw(text):
    if text == "n/a":
        return "", ""
    parts = text.split(" +/- ")
    return parts[0], parts[1] if len(parts) > 1 else ""


if __name__ == "__main__":
    main()
