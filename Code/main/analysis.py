"""Evaluation harness and analyses (H1-H7). All eval is no-grad on frozen weights."""
import numpy as np
import torch
from torch.distributions import Categorical

from environment import TMazeFreeNav


def _t(x, dev):
    return torch.as_tensor(x, dtype=torch.float32, device=dev)


def rollout(model, env, cfg, force_w=None, lesion=None, mot=1.0,
            greedy=True, collect_delay=False, record=False):
    """One trial on frozen weights (B=1). Returns correctness + DA/gate traces."""
    model.reset_state(1)   # hidden states: [1, n_gd], [1, n_hab]
    env.reset()
    da, w, delay_states = [], [], []
    rec_pos, rec_w, rec_a = [], [], []
    dev = cfg.device
    with torch.no_grad():
        done = False
        info = {"correct": False}
        while not done:
            obs_t     = _t(env.obs(),     dev).unsqueeze(0)   # [1, obs_dim]
            obs_hab_t = _t(env.obs_hab(), dev).unsqueeze(0)   # [1, obs_dim_hab]
            out       = model.step(obs_t, obs_hab_t, force_w=force_w, lesion=lesion, mot=mot)
            if collect_delay and env.phase == "delay":
                delay_states.append(model.h_hab[0].clone().cpu().numpy())  # [n_hab]
            logits = out["combined"][0]                  # [n_actions]
            a = int(torch.argmax(logits)) if greedy else int(Categorical(logits=logits).sample())
            da.append(float(out["da_request"][0]))
            w.append(float(out["w_gd"][0]))
            if record:
                rec_pos.append(list(env.pos))
                rec_w.append(round(float(out["w_gd"][0]), 3))
                rec_a.append(a)
            _, _, done, info = env.step(a)
    if record:
        rec_pos.append(list(env.pos))
    traj = ({"pos": rec_pos, "w": rec_w, "a": rec_a,
             "correct": bool(info["correct"]), "blocked": env.blocked} if record else None)
    last_delay = delay_states[-1] if delay_states else None
    return {"correct": bool(info["correct"]),
            "da_mean": float(np.mean(da)) if da else 0.0,
            "w_mean": float(np.mean(w)) if w else 0.0,
            "delay_state": last_delay, "blocked": env.blocked, "traj": traj}


def eval_trajectories(model, env, cfg, n, greedy=True, **kw):
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


def evaluate_vec(model, env, cfg, n, force_w=None, lesion=None, mot=1.0):
    """Vectorised evaluation: one TMazeVecEnv steps B trials at once, ceil(n/B)
    batches. All B environments advance in a single C-level NumPy step() — no
    per-env Python loop — and the forward pass is a single [B x n] matmul."""
    from environment import TMazeVecEnv
    B = min(cfg.batch_size, n)
    rng = np.random.default_rng(0)
    venv = TMazeVecEnv(cfg, rng, batch_size=B)
    venv.current_delay = env.current_delay

    all_correct, all_da, all_w = [], [], []
    dev = cfg.device

    with torch.no_grad():
        while len(all_correct) < n:
            b = min(B, n - len(all_correct))
            model.reset_state(B)
            venv.reset()                       # fresh trials for all B lanes
            active = np.ones(B, dtype=bool)
            da_sum = np.zeros(B); w_sum = np.zeros(B); cnt = np.zeros(B)
            correct = np.zeros(B, dtype=bool)

            for _ in range(cfg.max_episode_steps):
                obs_t     = _t(venv.obs(),     dev)
                obs_hab_t = _t(venv.obs_hab(), dev)
                out  = model.step(obs_t, obs_hab_t, force_w=force_w,
                                  lesion=lesion, mot=mot)
                acts = out["combined"].argmax(-1).cpu().numpy()   # greedy [B]

                da_np = out["da_request"].cpu().numpy()
                w_np  = out["w_gd"].cpu().numpy()
                da_sum += da_np * active
                w_sum  += w_np  * active
                cnt    += active

                _, _, done, info = venv.step(acts)
                newly = active & done
                correct[newly] = info["correct"][newly]
                active &= ~done
                if not active.any():
                    break

            cnt = np.maximum(cnt, 1)
            all_correct.extend(correct[:b].tolist())
            all_da.extend((da_sum[:b] / cnt[:b]).tolist())
            all_w.extend((w_sum[:b] / cnt[:b]).tolist())

    k = sum(all_correct[:n])
    return {"acc": k / n, "k": k, "n": n,
            "da_mean": float(np.mean(all_da[:n])),
            "w_mean": float(np.mean(all_w[:n]))}


# --------------------------------------------------------------- H1
def h1_learning(model, env, cfg):
    from scipy.stats import binomtest
    ev = evaluate_vec(model, env, cfg, cfg.final_trials)
    bt = binomtest(ev["k"], ev["n"], 0.5, alternative="greater")
    lo, hi = bt.proportion_ci(confidence_level=0.95)
    return {"accuracy": ev["acc"], "k": ev["k"], "n": ev["n"],
            "p_value": bt.pvalue, "ci95": [lo, hi],
            "pass": ev["acc"] >= 0.80 and bt.pvalue < 1e-3}


# --------------------------------------------------------------- H2
def h2_handoff(logs):
    ep = np.array(logs["episode"]); hab = np.array(logs["hab_solo_acc"])
    w  = np.array(logs["w_gd"]);   comb = np.array(logs["combined_acc"])
    cross = np.where(hab >= 0.8)[0]
    hab_onset = int(ep[cross[0]]) if len(cross) else None
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
    out = {}
    for name, state in [("learning", state_learn), ("maintenance", state_maint)]:
        model.load_state_dict(state)
        before = evaluate_vec(model, env, cfg, cfg.eval_trials, mot=1.0)["acc"]
        after  = evaluate_vec(model, env, cfg, cfg.eval_trials, mot=0.0)["acc"]
        out[name] = {"before": before, "after": after, "drop": before - after}
    out["dissociation_pass"] = (out["learning"]["drop"] > 0.15
                                and out["maintenance"]["drop"] < 0.10)
    return out


# --------------------------------------------------------------- H4
def h4_lesions(model, state_learn, state_maint, env, cfg):
    out = {}
    for name, state in [("learning", state_learn), ("maintenance", state_maint)]:
        model.load_state_dict(state)
        out[name] = {les: evaluate_vec(model, env, cfg, cfg.eval_trials, lesion=les)["acc"]
                     for les in ("gd", "hab", "both")}
        out[name]["intact"] = evaluate_vec(model, env, cfg, cfg.eval_trials)["acc"]
    return out


# --------------------------------------------------------------- H5
def h5_reactivation(model, state_maint, env, cfg):
    model.load_state_dict(state_maint)
    intact   = evaluate_vec(model, env, cfg, cfg.eval_trials)
    silenced = evaluate_vec(model, env, cfg, cfg.eval_trials, lesion="hab")
    sil_before = silenced["acc"]
    sil_after  = evaluate_vec(model, env, cfg, cfg.eval_trials, lesion="hab", mot=0.0)["acc"]
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
    if len(X) < 10:
        return {"pca_coords": [], "labels": [], "decoder_acc": 0.0, "decoder_sd": 0.0,
                "participation_ratio": 0.0, "pass": False,
                "note": f"Too few delay states collected ({len(X)}); model may not reach delay phase."}
    X = np.array(X); y = np.array(y)
    if len(np.unique(y)) < 2:
        return {"pca_coords": [], "labels": y.tolist(), "decoder_acc": 0.0,
                "decoder_sd": 0.0, "participation_ratio": participation_ratio(X),
                "pass": False,
                "note": ("All collected delay states share one class "
                         f"({len(X)} states, label {int(y[0])}); decoder undefined. "
                         "Model likely reaches only one arm — not yet learned.")}
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    clf = LogisticRegression(max_iter=1000)
    cv = cross_val_score(clf, X, y, cv=5)
    return {"pca_coords": coords.tolist(), "labels": y.tolist(),
            "decoder_acc": float(cv.mean()), "decoder_sd": float(cv.std()),
            "participation_ratio": participation_ratio(X),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pass": cv.mean() >= 0.90}


# --------------------------------------------------------------- optional fixed points
def fixed_points(model, env, cfg, n_seeds=30, steps=400, lr=0.05):
    """Sussillo & Barak (2013)-style fixed-point finder on the habitual net
    under a cueless delay observation (signal fully decayed)."""
    # Navigate to delay phase, then zero the phase signal (fully-decayed)
    env.reset()
    env.step(0); env.step(0)   # START→STEM→JUNCTION (enters sample)
    if env.open_side == "R":
        env.step(2); env.step(2)
    else:
        env.step(3); env.step(3)   # enters delay
    env._phase_signal[:] = 0.0    # cueless (fully-decayed delay)
    obs = env.obs_hab()           # habitual net takes the 4D obs, not the 6D GD obs
    x = _t(obs, cfg.device)   # [obs_dim_hab]
    hab = model.hab
    W_rec = hab.rec_weight().detach()   # frozen weights; only h is optimised
    W_in  = hab.W_in.detach()
    b     = hab.b.detach()
    res = []
    # dh dynamics use single-vector form (equivalent to batched for 1-D h)
    def _dh(hh):
        r = torch.tanh(hh)
        return -hh + r @ W_rec.T + x @ W_in.T + b
    for _ in range(n_seeds):
        h = torch.randn(cfg.n_hab, device=cfg.device) * 0.5
        h.requires_grad_(True)
        opt = torch.optim.Adam([h], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = 0.5 * (_dh(h) ** 2).sum()
            loss.backward(); opt.step()
        with torch.no_grad():
            speed = float((_dh(h) ** 2).sum().sqrt())
        J = torch.autograd.functional.jacobian(_dh, h.detach())
        eig = torch.linalg.eigvals(J).real
        res.append({"speed": speed, "max_real_eig": float(eig.max()),
                    "stable": bool((eig < 0).all())})
    return res
