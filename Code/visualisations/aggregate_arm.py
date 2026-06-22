"""Cross-seed aggregation for one arm of a ladder experiment.

Used by build_batch_viz.py to turn a directory of seed<N>/results.json (one
arm, e.g. results/e4/value_free/<ts>_<commit>/) into the mean/sd arrays the
batch HTML needs for its arm-comparison panels. The per-point math mirrors
Code/main/aggregate.py's `_ms()` helper (mean/sd, None-safe) — re-derived here
rather than imported, since aggregate.py works off a flat results/seed<N>/
layout and isn't meant to know about arms/experiments (see its docstring);
this module is the arm-aware analogue, kept import-free of aggregate.py so
neither file's behaviour is coupled to the other's.

No plotting here — this only returns plain dicts/lists of JSON-safe numbers
for build_batch_viz.py to embed; all rendering happens in JS in the browser,
same division of labour as build_viz.py/visualiser.html.
"""
import glob
import json
import os

import numpy as np


def _ms(xs):
    """Mean/sd over finite, non-None values; (None, None) if nothing usable."""
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (None, None)


def load_arm_seeds(arm_dir):
    """Load every seed<N>/results.json directly under arm_dir, sorted by seed."""
    seeds = []
    for path in sorted(glob.glob(os.path.join(arm_dir, "seed*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        seed_name = os.path.basename(os.path.dirname(path))
        r["_seed_dir"] = seed_name
        seeds.append(r)
    return seeds


def _nan_to_none(x):
    """Recursively replace NaN/Inf with None so json.dumps emits valid JSON
    (Python's json module happily writes the non-standard NaN/Infinity
    literals; browsers' strict JSON.parse rejects them -- this is the fix)."""
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _col_stats(cols_per_episode):
    """Per-episode mean/sd/min/max across seeds, ignoring None/NaN entries at
    each episode independently (a seed missing one point doesn't blank the
    whole column). cols_per_episode: list (over episodes) of lists (over seeds).
    """
    means, sds, mins, maxs = [], [], [], []
    for vals in cols_per_episode:
        usable = [v for v in vals if v is not None and np.isfinite(v)]
        if usable:
            means.append(float(np.mean(usable)))
            sds.append(float(np.std(usable)))
            mins.append(float(np.min(usable)))
            maxs.append(float(np.max(usable)))
        else:
            means.append(None); sds.append(None); mins.append(None); maxs.append(None)
    return means, sds, mins, maxs


def _align_logs(seeds):
    """Stack each seed's per-episode log arrays onto a common episode axis.

    Seeds can have slightly different episode counts (e.g. early stop, or a
    curriculum that advances at different paces) -- align on the shortest
    common length rather than assuming identical `episode` arrays. Returns
    None if no seed has usable logs.
    """
    logs_list = [r.get("logs") for r in seeds if r.get("logs")]
    if not logs_list:
        return None
    n = min(len(lg["episode"]) for lg in logs_list)
    if n == 0:
        return None
    episode = logs_list[0]["episode"][:n]
    fields = ["combined_acc", "hab_solo_acc", "gd_solo_acc", "w_gd", "da_recruit",
              "fixed_delay_acc"]
    out = {"episode": episode, "n_seeds": len(logs_list)}
    for f in fields:
        cols = [lg[f][:n] for lg in logs_list if lg.get(f) is not None]
        if not cols:
            continue
        # transpose to per-episode lists-over-seeds; None entries (e.g.
        # fixed_delay_acc before the curriculum reaches ceiling) pass through.
        per_episode = [[cols[s][i] for s in range(len(cols))] for i in range(n)]
        means, sds, mins, maxs = _col_stats(per_episode)
        out[f] = {
            "mean": means, "sd": sds, "min": mins, "max": maxs,
            "per_seed": _nan_to_none(cols),
        }
    return out


def aggregate_arm(arm_dir, arm_name):
    """Build the full cross-seed aggregate dict for one arm directory.

    Returns None if the arm dir has no seeds with results.json.
    """
    seeds = load_arm_seeds(arm_dir)
    if not seeds:
        return None

    h1_acc = [r["h1"]["accuracy"] for r in seeds if "h1" in r]
    h1_pass = [bool(r["h1"]["pass"]) for r in seeds if "h1" in r]
    hab_onset = [r["h2"]["hab_onset_episode"] for r in seeds if "h2" in r]
    wgd_onset = [r["h2"]["wgd_drop_onset_episode"] for r in seeds if "h2" in r]
    h2_pass = [bool(r["h2"]["pass"]) for r in seeds if "h2" in r]
    dev_learn = [r["h3"]["learning"]["drop"] for r in seeds if "h3" in r]
    dev_maint = [r["h3"]["maintenance"]["drop"] for r in seeds if "h3" in r]
    h3_pass = [bool(r["h3"]["dissociation_pass"]) for r in seeds if "h3" in r]
    h5_delta = [r["h5"]["da_request_delta"] for r in seeds if "h5" in r]
    h5_acc_sil = [r["h5"]["acc_hab_silenced"] for r in seeds if "h5" in r]
    h5_pass = [bool(r["h5"]["pass"]) for r in seeds if "h5" in r]
    gc = [r["h5"]["gate_clamp_control"] for r in seeds
          if "h5" in r and r["h5"].get("gate_clamp_control")]
    gc_gd_clamped = [g["acc_gd_clamped"] for g in gc]
    gc_policy_preserved = [bool(g["policy_preserved"]) for g in gc]
    h6_dec = [r["h6"]["decoder_acc"] for r in seeds if "h6" in r]

    config = seeds[0].get("config", {})

    return {
        "arm": arm_name,
        "n_seeds": len(seeds),
        "seed_dirs": [r["_seed_dir"] for r in seeds],
        "seeds": [r.get("seed") for r in seeds],
        "config": config,
        "logs_agg": _align_logs(seeds),
        "h1": {
            "accuracy_mean_sd": _ms(h1_acc), "accuracy": h1_acc,
            "n_pass": int(sum(h1_pass)), "n_seeds": len(h1_pass),
            "gate_clear": bool(h1_pass) and all(h1_pass),
        },
        "h2": {
            "hab_onset_mean_sd": _ms(hab_onset), "hab_onset": hab_onset,
            "wgd_drop_onset_mean_sd": _ms(wgd_onset), "wgd_drop_onset": wgd_onset,
            "n_pass": int(sum(h2_pass)), "n_seeds": len(h2_pass),
        },
        "h3": {
            "learning_drop_mean_sd": _ms(dev_learn), "learning_drop": dev_learn,
            "maintenance_drop_mean_sd": _ms(dev_maint), "maintenance_drop": dev_maint,
            "n_pass": int(sum(h3_pass)), "n_seeds": len(h3_pass),
        },
        "h5": {
            "da_request_delta_mean_sd": _ms(h5_delta), "da_request_delta": h5_delta,
            "acc_hab_silenced_mean_sd": _ms(h5_acc_sil), "acc_hab_silenced": h5_acc_sil,
            "n_pass": int(sum(h5_pass)), "n_seeds": len(h5_pass),
            "gate_clamp_acc_gd_clamped_mean_sd": _ms(gc_gd_clamped),
            "gate_clamp_acc_gd_clamped": gc_gd_clamped,
            "gate_clamp_n_policy_preserved": int(sum(gc_policy_preserved)),
            "gate_clamp_n_seeds": len(gc_policy_preserved),
        },
        "h6": {
            "decoder_acc_mean_sd": _ms(h6_dec), "decoder_acc": h6_dec,
        },
    }
