"""Evaluation harness and analyses (H1-H7). All eval is no-grad on frozen weights."""
import numpy as np
import torch
from torch.distributions import Categorical

from environment import TMazeDNMTP


def _t(x, dev):
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def rollout(model, env, cfg, force_w=None, lesion=None, mot=1.0,
            greedy=True, collect_delay=False, record=False):
    """One trial on frozen weights. Returns correctness + DA/gate traces (+delay states,
    +trajectory if record=True). Trajectory = test-phase cell path with per-step w_GD."""
    model.reset_state()
    env.reset()
    da, w, delay_states = [], [], []
    rec_pos, rec_w, rec_a = [], [], []
    dev = cfg.device
    with torch.no_grad():
        done = False
        info = {"correct": False}
        while not done:
            allo, ego = env.obs()
            out = model.step(_t(allo, dev), _t(ego, dev),
                             force_w=force_w, lesion=lesion, mot=mot)
            if collect_delay and env.phase == "delay":
                delay_states.append(model.h_hab.clone().cpu().numpy())
            if env.agent_controlled:
                logits = out["combined"]
                a = int(torch.argmax(logits)) if greedy else int(Categorical(logits=logits).sample())
                da.append(float(out["da_request"]))
                w.append(float(out["w_gd"]))
                if record:
                    rec_pos.append(list(env.pos))
                    rec_w.append(round(float(out["w_gd"]), 3))
                    rec_a.append(a)
            else:
                a = 0
            _, _, done, info = env.step(a)
    if record:
        rec_pos.append(list(env.pos))   # final position
    traj = ({"pos": rec_pos, "w": rec_w, "a": rec_a,
             "correct": bool(info["correct"]), "blocked": env.blocked} if record else None)
    last_delay = delay_states[-1] if delay_states else None
    return {"correct": bool(info["correct"]),
            "da_mean": float(np.mean(da)) if da else 0.0,
            "w_mean": float(np.mean(w)) if w else 0.0,
            "delay_state": last_delay, "blocked": env.blocked, "traj": traj}


def eval_trajectories(model, env, cfg, n, greedy=True, **kw):
    """Replay n trials and return their recorded trajectories (compact)."""
    out = []
    for _ in range(n):
        r = rollout(model, env, cfg, greedy=greedy, record=True, **kw)
        out.append(r["traj"])
    return out


def evaluate(model, env, cfg, n, **kw):
    res = [rollout(model, env, cfg, **kw) for _ in range(n)]
    k = sum(r["correct"] for r in res)
    return {"acc": k / n, "k": k, "n": n,
            "da_mean": float(np.mean([r["da_mean"] for r in res])),
            "w_mean": float(np.mean([r["w_mean"] for r in res]))}


# --------------------------------------------------------------- H1
def h1_learning(model, env, cfg):
    from scipy.stats import binomtest
    ev = evaluate(model, env, cfg, cfg.final_trials)
    bt = binomtest(ev["k"], ev["n"], 0.5, alternative="greater")
    lo, hi = bt.proportion_ci(confidence_level=0.95)
    return {"accuracy": ev["acc"], "k": ev["k"], "n": ev["n"],
            "p_value": bt.pvalue, "ci95": [lo, hi],
            "pass": ev["acc"] >= 0.80 and bt.pvalue < 1e-3}


# --------------------------------------------------------------- H2
def h2_handoff(logs):
    """Quantify the lock between habitual-solo competence and the w_GD drop."""
    ep = np.array(logs["episode"]); hab = np.array(logs["hab_solo_acc"])
    w = np.array(logs["w_gd"]); comb = np.array(logs["combined_acc"])
    # habitual-competence onset: first eval where hab-solo crosses 0.8
    cross = np.where(hab >= 0.8)[0]
    hab_onset = int(ep[cross[0]]) if len(cross) else None
    # w_GD drop onset: first eval where w_GD < 50% of its running peak
    peak = np.maximum.accumulate(w)
    drop = np.where(w < 0.5 * peak)[0]
    w_onset = int(ep[drop[0]]) if len(drop) else None
    return {"hab_onset_episode": hab_onset, "wgd_drop_onset_episode": w_onset,
            "wgd_peak": float(w.max()), "wgd_final": float(w[-1]),
            "combined_final": float(comb[-1]),
            "pass": (hab_onset is not None and w_onset is not None
                     and w_onset >= hab_onset and w[-1] <= 0.5 * peak.max()
                     and comb[-1] >= 0.8)}


# --------------------------------------------------------------- H3
def h3_devaluation(model, state_learn, state_maint, env, cfg):
    """mot=1 vs mot~0 at a learning- and a maintenance-phase checkpoint."""
    out = {}
    for name, state in [("learning", state_learn), ("maintenance", state_maint)]:
        model.load_state_dict(state)
        before = evaluate(model, env, cfg, cfg.eval_trials, mot=1.0)["acc"]
        after = evaluate(model, env, cfg, cfg.eval_trials, mot=0.0)["acc"]
        out[name] = {"before": before, "after": after, "drop": before - after}
    out["dissociation_pass"] = (out["learning"]["drop"] > 0.15
                                and out["maintenance"]["drop"] < 0.10)
    return out


# --------------------------------------------------------------- H4
def h4_lesions(model, state_learn, state_maint, env, cfg):
    out = {}
    for name, state in [("learning", state_learn), ("maintenance", state_maint)]:
        model.load_state_dict(state)
        out[name] = {les: evaluate(model, env, cfg, cfg.eval_trials, lesion=les)["acc"]
                     for les in ("gd", "hab", "both")}
        out[name]["intact"] = evaluate(model, env, cfg, cfg.eval_trials)["acc"]
    return out


# --------------------------------------------------------------- H5 (falsification)
def h5_reactivation(model, state_maint, env, cfg):
    """Silence habitual in maintenance: does GD's DA-request RISE and devaluation-
    sensitivity RETURN? If DA-request does not rise, the handoff was a schedule."""
    model.load_state_dict(state_maint)
    intact = evaluate(model, env, cfg, cfg.eval_trials)
    silenced = evaluate(model, env, cfg, cfg.eval_trials, lesion="hab")
    # devaluation-sensitivity with habitual silenced
    sil_before = silenced["acc"]
    sil_after = evaluate(model, env, cfg, cfg.eval_trials, lesion="hab", mot=0.0)["acc"]
    da_rise = silenced["da_mean"] - intact["da_mean"]
    return {"da_request_intact": intact["da_mean"],
            "da_request_hab_silenced": silenced["da_mean"],
            "da_request_delta": da_rise,
            "acc_intact": intact["acc"], "acc_hab_silenced": silenced["acc"],
            "deval_sensitivity_silenced": sil_before - sil_after,
            "pass": da_rise > 0 and silenced["acc"] >= 0.65 and (sil_before - sil_after) > 0.15,
            "note": ("DA-request did NOT rise -> handoff is a schedule in disguise; "
                     "report as a NEGATIVE result." if da_rise <= 0 else
                     "DA-request rose -> dormant goal-directed solution reactivates.")}


# --------------------------------------------------------------- H6
def participation_ratio(X):
    Xc = X - X.mean(0, keepdims=True)
    cov = Xc.T @ Xc / max(1, len(Xc) - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-12))


def h6_attractor(model, state_maint, env, cfg):
    """PCA + linear decoder of arm identity from delay-period habitual activity."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    model.load_state_dict(state_maint)
    X, y = [], []
    for _ in range(cfg.attractor_trials):
        r = rollout(model, env, cfg, collect_delay=True)
        if r["delay_state"] is not None:
            X.append(r["delay_state"])
            y.append(0 if r["blocked"] == "L" else 1)
    X = np.array(X); y = np.array(y)
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    clf = LogisticRegression(max_iter=1000)
    cv = cross_val_score(clf, X, y, cv=5)
    return {"pca_coords": coords.tolist(), "labels": y.tolist(),
            "decoder_acc": float(cv.mean()), "decoder_sd": float(cv.std()),
            "participation_ratio": participation_ratio(X),
            "pass": cv.mean() >= 0.90}


# --------------------------------------------------------------- optional fixed points
def fixed_points(model, env, cfg, n_seeds=30, steps=400, lr=0.05):
    """Light Sussillo & Barak (2013)-style finder on the habitual net under the
    (cueless) delay input. Reports speed ||dh/dt|| and slowest Jacobian eigenvalue."""
    env.reset()
    while env.phase != "delay":
        env.step(0)
    _, ego = env.obs()
    x = _t(ego, cfg.device)
    hab = model.hab
    res = []
    for _ in range(n_seeds):
        h = torch.randn(cfg.n_hab, device=cfg.device) * 0.5
        h.requires_grad_(True)
        opt = torch.optim.Adam([h], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            dh = -h + hab.W @ torch.tanh(h) + hab.W_in @ x + hab.b
            loss = 0.5 * (dh ** 2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            dh = -h + hab.W @ torch.tanh(h) + hab.W_in @ x + hab.b
            speed = float((dh ** 2).sum().sqrt())
        J = torch.autograd.functional.jacobian(
            lambda hh: -hh + hab.W @ torch.tanh(hh) + hab.W_in @ x + hab.b, h.detach())
        eig = torch.linalg.eigvals(J).real
        res.append({"speed": speed, "max_real_eig": float(eig.max()),
                    "stable": bool((eig < 0).all())})
    return res
