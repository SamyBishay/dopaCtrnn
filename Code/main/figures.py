"""Figures 1-6. Each takes a results dict and writes a PNG into outdir."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig1_handoff(logs, outdir, h2=None):
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
    # Vertical onset markers — hab competence precedes w_GD drop (H2 claim)
    if h2 is not None:
        if h2.get("hab_onset_episode") is not None:
            ax.axvline(h2["hab_onset_episode"], color="steelblue", lw=1.2, ls="--",
                       label=f"hab≥0.8 (ep {h2['hab_onset_episode']})")
        if h2.get("wgd_drop_onset_episode") is not None:
            ax.axvline(h2["wgd_drop_onset_episode"], color="firebrick", lw=1.2, ls=":",
                       label=f"$w_{{GD}}$ drops (ep {h2['wgd_drop_onset_episode']})")
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8)
    ax.set_title("Fig 1. Emergent goal-directed → habitual handoff")
    _save(fig, outdir, "fig1_handoff.png")


def fig2_devaluation(h3, outdir):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    groups = ["learning", "maintenance"]
    before = [h3[g]["before"] for g in groups]
    after  = [h3[g]["after"]  for g in groups]
    drops  = [h3[g]["drop"]   for g in groups]
    x = np.arange(2); wdt = 0.35
    ax.bar(x - wdt / 2, before, wdt, label="valued (motivation=1)")
    ax.bar(x + wdt / 2, after,  wdt, label="devalued (motivation=0)")
    # Δ annotations
    for i, (xi, drop) in enumerate(zip(x, drops)):
        top = max(before[i], after[i]) + 0.04
        ax.annotate(f"Δ={drop:+.2f}", xy=(xi, top), ha="center", fontsize=8, color="dimgray")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0, 1.12)
    ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
    ax.set_title("Fig 2. Devaluation dissociation\n"
                 "(habitual loss is value-free by construction)")
    _save(fig, outdir, "fig2_devaluation.png")


def fig3_lesions(h4, outdir):
    fig, ax = plt.subplots(figsize=(6, 4))
    conds  = ["intact", "gd", "hab", "both"]
    labels = ["intact", "GD silenced\n(mPFC)", "Hab silenced\n(DLS)", "both\nsilenced"]
    x = np.arange(len(conds)); wdt = 0.38
    for i, g in enumerate(["learning", "maintenance"]):
        ax.bar(x + (i - .5) * wdt, [h4[g][c] for c in conds], wdt, label=g)
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15); ax.set_ylim(0, 1.12)
    ax.set_ylabel("test accuracy"); ax.legend(fontsize=8)
    ax.set_title("Fig 3. System necessity × training phase")
    _save(fig, outdir, "fig3_lesions.png")


def fig4_reactivation(h5, outdir, h3=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.8))

    # Left: DA-request with/without habitual
    da_vals = [h5["da_request_intact"], h5["da_request_hab_silenced"]]
    a.bar(["habitual\nintact", "habitual\nsilenced"], da_vals,
          color=["steelblue", "darkorchid"])
    a.set_ylabel("mean DA-request"); a.set_title("DA-request neuron")
    a.set_ylim(0, max(max(da_vals) * 1.5, 0.15))
    delta = h5["da_request_delta"]
    a.text(0.5, 0.93, f"Δ={delta:+.3f}", transform=a.transAxes,
           ha="center", fontsize=9, color="dimgray")

    # Right: devaluation sensitivity; use actual maintenance drop if h3 available
    intact_drop = h3["maintenance"]["drop"] if h3 is not None else 0.0
    b.bar(["intact", "habitual\nsilenced"],
          [intact_drop, h5["deval_sensitivity_silenced"]],
          color=["steelblue", "coral"])
    b.set_ylabel("devaluation sensitivity\n(accuracy drop)")
    b.set_title("Value-sensitivity")
    b.set_ylim(0, 1.02)

    status = "PASS" if h5["pass"] else "FAIL — DA-request rise marginal"
    fig.suptitle(f"Fig 4. Reactivation after silencing DLS  [H5: {status}]", fontsize=10)
    _save(fig, outdir, "fig4_reactivation.png")


def fig5_attractor(h6, outdir):
    if not h6.get("pca_coords"):
        fig, ax = plt.subplots(figsize=(5, 4.4))
        ax.text(0.5, 0.5, f"No delay states collected.\n{h6.get('note', '')}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                wrap=True)
        ax.set_title("Fig 5. Delay-period WM states (no data)")
        _save(fig, outdir, "fig5_attractor.png")
        return

    coords = np.array(h6["pca_coords"]); y = np.array(h6["labels"])
    fig, ax = plt.subplots(figsize=(5, 4.4))

    ev = h6.get("explained_variance_ratio")
    xlabel = f"PC1 ({ev[0]:.0%})" if ev else "PC1"
    ylabel = f"PC2 ({ev[1]:.0%})" if ev else "PC2"

    for lab, name, col in [(0, "blocked=L (go R)", "tab:blue"),
                            (1, "blocked=R (go L)", "tab:orange")]:
        m = y == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=12, alpha=.6, color=col, label=name)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    ax.set_title("Fig 5. Habitual (DLS) delay-period states\n"
                 f"decoder acc = {h6['decoder_acc']:.2f} (full hab. state), "
                 f"PR = {h6['participation_ratio']:.1f}")
    _save(fig, outdir, "fig5_attractor.png")


def fig6_control(summary, outdir):
    """Trained vs untrained (H7). summary has 'trained_acc' and 'untrained_acc' lists."""
    tr = summary.get("trained_acc", []); un = summary.get("untrained_acc", [])
    if not tr or not un:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [tr, un]
    ax.boxplot(data, labels=["trained", "untrained"], showmeans=True)
    rng = np.random.default_rng(42)
    for i, d in enumerate(data, 1):
        jitter = rng.uniform(-0.05, 0.05, len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, alpha=.6, color="k", s=14)
    ax.axhline(0.5, color="grey", ls=":", label="chance")
    ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("Fig 6. Negative control: untrained baseline (H7)")
    _save(fig, outdir, "fig6_control.png")


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name), dpi=140, bbox_inches="tight")
    plt.close(fig)
