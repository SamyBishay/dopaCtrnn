"""Figures 1-6. Each takes a results dict and writes a PNG into outdir."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig1_handoff(logs, outdir):
    ep = logs["episode"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ep, logs["combined_acc"], label="combined", lw=2)
    ax.plot(ep, logs["hab_solo_acc"], label="habitual solo", lw=2)
    ax.plot(ep, logs["gd_solo_acc"], label="goal-directed solo", lw=2, ls="--")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("episode"); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.plot(ep, logs["w_gd"], color="firebrick", lw=1.6, alpha=.8, label="$w_{GD}$")
    ax2.plot(ep, logs["da_recruit"], color="purple", lw=1.4, alpha=.7, ls="-.",
             label="DA-request")
    ax2.set_ylabel("$w_{GD}$  /  DA-request"); ax2.set_ylim(0, 1.02)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    ax.set_title("Fig 1. Emergent goal-directed -> habitual handoff")
    _save(fig, outdir, "fig1_handoff.png")


def fig2_devaluation(h3, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    groups = ["learning", "maintenance"]
    before = [h3[g]["before"] for g in groups]
    after = [h3[g]["after"] for g in groups]
    x = np.arange(2); wdt = 0.35
    ax.bar(x - wdt / 2, before, wdt, label="valued (mot=1)")
    ax.bar(x + wdt / 2, after, wdt, label="devalued (mot~0)")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.02)
    ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
    ax.set_title("Fig 2. Devaluation dissociation")
    _save(fig, outdir, "fig2_devaluation.png")


def fig3_lesions(h4, outdir):
    fig, ax = plt.subplots(figsize=(6, 4))
    conds = ["intact", "gd", "hab", "both"]
    labels = ["intact", "lesion mPFC", "lesion DLS", "lesion both"]
    x = np.arange(len(conds)); wdt = 0.38
    for i, g in enumerate(["learning", "maintenance"]):
        ax.bar(x + (i - .5) * wdt, [h4[g][c] for c in conds], wdt, label=g)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15); ax.set_ylim(0, 1.02)
    ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
    ax.set_title("Fig 3. Lesion x training phase")
    _save(fig, outdir, "fig3_lesions.png")


def fig4_reactivation(h5, outdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.8))
    a.bar(["habitual\nintact", "habitual\nsilenced"],
          [h5["da_request_intact"], h5["da_request_hab_silenced"]],
          color=["steelblue", "firebrick"])
    a.set_ylabel("mean DA-request"); a.set_title("DA-request")
    b.bar(["intact", "habitual\nsilenced"],
          [0.0, h5["deval_sensitivity_silenced"]], color=["grey", "firebrick"])
    b.set_ylabel("devaluation sensitivity (acc drop)")
    b.set_title("Value-sensitivity")
    fig.suptitle("Fig 4. Reactivation after silencing DLS (H5 falsification)")
    _save(fig, outdir, "fig4_reactivation.png")


def fig5_attractor(h6, outdir):
    coords = np.array(h6["pca_coords"]); y = np.array(h6["labels"])
    fig, ax = plt.subplots(figsize=(5, 4.4))
    for lab, name, col in [(0, "sampled L", "tab:blue"), (1, "sampled R", "tab:orange")]:
        m = y == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=.6, color=col, label=name)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=8)
    ax.set_title("Fig 5. Delay-period WM states\n"
                 f"decoder acc = {h6['decoder_acc']:.2f}, "
                 f"PR = {h6['participation_ratio']:.1f}")
    _save(fig, outdir, "fig5_attractor.png")


def fig6_control(summary, outdir):
    """Trained vs untrained (H7). summary has 'trained_acc' and 'untrained_acc' lists."""
    fig, ax = plt.subplots(figsize=(5, 4))
    tr = summary.get("trained_acc", []); un = summary.get("untrained_acc", [])
    data = [tr, un]
    ax.boxplot(data, labels=["trained", "untrained"], showmeans=True)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i), d, alpha=.6, color="k", s=14)
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Fig 6. Negative control (untrained)")
    _save(fig, outdir, "fig6_control.png")


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=140)
    plt.close(fig)
