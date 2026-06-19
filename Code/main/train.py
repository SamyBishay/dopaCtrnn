"""Training. Two optimizers:
  - goal-directed: A2C over the COMBINED policy with pi_H DETACHED (so the habitual
    net never receives a reward gradient) + critic + entropy + DA-request penalty.
    The DA gate w_GD sits in this graph, so the DA-request is shaped by the A2C
    advantage -> this is the "a2c_coupled" path (OPEN #1 default).
  - habitual: value-free. Action-prediction-error (imitate the executed action) plus a
    small intrinsic-efficiency REINFORCE on step-cost/completion-bonus. The task reward
    NEVER enters this loss -> structural devaluation-insensitivity.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import evaluate, _t


def discounted(rewards, gamma, dev):
    out, R = [], 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    return torch.tensor(list(reversed(out)), dtype=torch.float32, device=dev)


def _train_episode(model, env, cfg, opt_gd, opt_hab, da_lambda, record=False):
    dev = cfg.device
    model.reset_state()
    env.reset()
    pi_gd, pi_h, ws, das, vals, acts, task_rs, int_rs = ([] for _ in range(8))
    rec_pos, rec_w, rec_phase = [], [], []
    done = False
    while not done:
        obs = env.obs()
        out = model.step(_t(obs, dev), _t(obs, dev))
        a = Categorical(logits=out["combined"]).sample()
        pi_gd.append(out["pi_gd"]); pi_h.append(out["pi_h"]); ws.append(out["w_gd"])
        das.append(out["da_request"]); vals.append(out["value"]); acts.append(a)
        if record:
            rec_pos.append(list(env.pos))
            rec_w.append(round(float(out["w_gd"].detach()), 3))
            rec_phase.append(env.phase)
        tr, ir, done, info = env.step(int(a))
        task_rs.append(tr); int_rs.append(ir)

    traj = None
    if record:
        rec_pos.append(list(env.pos))
        traj = {"pos": rec_pos, "w": rec_w, "phase": rec_phase,
                "correct": bool(info["correct"]), "blocked": env.blocked}
    if not acts:
        return 0.0, 0.0, traj
    n = min(len(acts), len(task_rs))
    pi_gd, pi_h, ws, das, vals, acts = (z[:n] for z in (pi_gd, pi_h, ws, das, vals, acts))
    task_rs, int_rs = task_rs[:n], int_rs[:n]

    # ---- goal-directed A2C (habitual detached) ----
    returns = discounted(task_rs, cfg.gamma, dev)
    values = torch.stack(vals)
    adv = returns - values.detach()
    logps, ents = [], []
    for pg, ph, w, a in zip(pi_gd, pi_h, ws, acts):
        c = w * pg + (1.0 - w) * ph.detach()
        d = Categorical(logits=c)
        logps.append(d.log_prob(a)); ents.append(d.entropy())
    logps = torch.stack(logps); ents = torch.stack(ents); das_t = torch.stack(das)
    if cfg.da_request_training != "a2c_coupled":
        raise NotImplementedError(
            "OPEN #1: DA-request training '%s' is not implemented." % cfg.da_request_training)
    # Expression-gated plasticity: when GD is not expressed (w_GD low) its
    # POLICY gradient is suppressed, preserving the dormant solution for
    # reactivation (H5). Critic and DA penalty continue updating so advantage
    # estimates and DA cost remain valid during maintenance.
    expr_gate = torch.stack(ws).mean().detach().clamp(0.0, 1.0)
    policy_loss = -(adv.detach() * logps).sum()
    critic_loss = cfg.value_coef * ((returns - values) ** 2).sum()
    entropy_loss = -cfg.entropy_beta * ents.sum()
    da_penalty = da_lambda * (das_t ** 2).sum()
    gd_loss = expr_gate * (policy_loss + entropy_loss) + critic_loss + da_penalty

    # ---- habitual: value-free APE + intrinsic efficiency (NO task reward) ----
    returns_int = discounted(int_rs, cfg.gamma, dev)
    adv_int = returns_int - returns_int.mean().detach()
    ce, logp_h = [], []
    for ph, a, ai in zip(pi_h, acts, adv_int):
        ce.append(F.cross_entropy(ph.unsqueeze(0), a.view(1)))
        logp_h.append(Categorical(logits=ph).log_prob(a))
    ce = torch.stack(ce).sum(); logp_h = torch.stack(logp_h)
    hab_loss = cfg.ape_weight * ce + cfg.eff_weight * (-(adv_int.detach() * logp_h).sum())

    opt_gd.zero_grad(); gd_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.gd_params(), cfg.grad_clip); opt_gd.step()
    opt_hab.zero_grad(); hab_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.hab_params(), cfg.grad_clip); opt_hab.step()
    return float(gd_loss.detach()), float(hab_loss.detach()), traj


def train(cfg, verbose=True):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = TMazeFreeNav(cfg, rng)
    model = DualSystemModel(cfg).to(cfg.device)
    opt_gd = torch.optim.Adam(model.gd_params(), lr=cfg.lr_gd)
    opt_hab = torch.optim.Adam(model.hab_params(), lr=cfg.lr_hab)

    logs = {k: [] for k in ("episode", "combined_acc", "hab_solo_acc",
                            "gd_solo_acc", "w_gd", "da_recruit", "delay")}
    ckpt_learn = ckpt_maint = first_state = None
    train_trajs = []
    delay_advance_count = 0   # consecutive evals above threshold

    for ep in range(1, cfg.episodes + 1):
        if ep <= cfg.da_warmup:
            da_lambda = 0.0
        elif ep <= cfg.da_warmup + cfg.da_ramp:
            da_lambda = cfg.da_cost_lambda * (ep - cfg.da_warmup) / cfg.da_ramp
        else:
            da_lambda = cfg.da_cost_lambda
        rec = cfg.traj_log_every > 0 and ep % cfg.traj_log_every == 0
        _, _, traj = _train_episode(model, env, cfg, opt_gd, opt_hab, da_lambda, record=rec)
        if rec and traj is not None:
            traj["episode"] = ep
            train_trajs.append(traj)

        if ep % cfg.eval_every == 0 or ep == cfg.episodes:
            comb = evaluate(model, env, cfg, cfg.eval_trials)
            hab  = evaluate(model, env, cfg, cfg.eval_trials, force_w=0.0)
            gd   = evaluate(model, env, cfg, cfg.eval_trials, force_w=1.0)
            logs["episode"].append(ep)
            logs["combined_acc"].append(comb["acc"])
            logs["hab_solo_acc"].append(hab["acc"])
            logs["gd_solo_acc"].append(gd["acc"])
            logs["w_gd"].append(comb["w_mean"])
            logs["da_recruit"].append(comb["da_mean"])
            logs["delay"].append(env.current_delay)

            if first_state is None:
                first_state = copy.deepcopy(model.state_dict())
            if (ckpt_learn is None and comb["acc"] >= cfg.learn_combined_min
                    and hab["acc"] <= cfg.learn_habsolo_max):
                ckpt_learn = copy.deepcopy(model.state_dict())
            if hab["acc"] >= cfg.maint_solo_min and comb["acc"] >= cfg.maint_solo_min:
                ckpt_maint = copy.deepcopy(model.state_dict())

            # curriculum: advance delay when both combined and habitual are accurate
            if (comb["acc"] >= cfg.delay_advance_acc
                    and hab["acc"] >= cfg.delay_advance_acc):
                delay_advance_count += 1
            else:
                delay_advance_count = 0
            if delay_advance_count >= cfg.delay_advance_evals:
                env.advance_delay()
                delay_advance_count = 0
                if verbose:
                    print(f"  -> delay → {env.current_delay}", flush=True)

            if verbose:
                print(f"ep {ep:5d} | comb {comb['acc']:.2f} habSolo {hab['acc']:.2f} "
                      f"gdSolo {gd['acc']:.2f} | w_GD {comb['w_mean']:.2f} "
                      f"DA {comb['da_mean']:.2f} delay {env.current_delay}", flush=True)

    if ckpt_maint is None:
        import warnings
        warnings.warn("maint_solo_min never reached — ckpt_maint falls back to final weights. "
                      "H3/H5 dissociation results may be invalid.")
        ckpt_maint = copy.deepcopy(model.state_dict())
    if ckpt_learn is None:
        import warnings
        warnings.warn("learn checkpoint never reached — ckpt_learn falls back to first_state. "
                      "H3/H4 learning-phase results may be invalid.")
        ckpt_learn = first_state
    return model, logs, ckpt_learn, ckpt_maint, train_trajs, env.current_delay
