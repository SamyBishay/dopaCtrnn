"""Training — vectorised over B parallel environments.

Each training iteration steps all B environments in lockstep. The batch
dimension lets PyTorch dispatch [B × n] matrix multiplies instead of
n-length vector-matrix products, saturating BLAS and cutting wall-time
by ~10-20× vs. single-env sequential training.

Two optimizers (goal-directed, habitual) are updated once per iteration
over the summed losses from all B completed episodes.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import evaluate, _t


def discounted(rewards_list, gamma, dev):
    out, R = [], 0.0
    for r in reversed(rewards_list):
        R = r + gamma * R
        out.append(R)
    return torch.tensor(list(reversed(out)), dtype=torch.float32, device=dev)


def _train_batch(model, envs, cfg, opt_gd, opt_hab, da_lambda):
    """One vectorised training iteration over B=len(envs) parallel episodes."""
    B   = len(envs)
    dev = cfg.device
    model.reset_state(B)
    for env in envs:
        env.reset()

    # Per-step accumulators — each is a list of [B, ...] tensors
    all_pi_gd, all_pi_h, all_w, all_da = [], [], [], []
    all_val, all_acts = [], []
    all_task_r, all_int_r, all_valid = [], [], []

    active = [True] * B

    for _ in range(cfg.max_episode_steps):
        obs_np = np.stack([env.obs() for env in envs])     # [B, obs_dim]
        obs_t  = _t(obs_np, dev)
        out    = model.step(obs_t, obs_t)
        acts   = Categorical(logits=out["combined"]).sample()  # [B]

        all_pi_gd.append(out["pi_gd"]); all_pi_h.append(out["pi_h"])
        all_w.append(out["w_gd"]);      all_da.append(out["da_request"])
        all_val.append(out["value"]);   all_acts.append(acts)

        task_rs = torch.zeros(B); int_rs = torch.zeros(B)
        valid   = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            if active[b]:
                tr, ir, done, _ = envs[b].step(int(acts[b].item()))
                task_rs[b] = tr; int_rs[b] = ir; valid[b] = True
                if done:
                    active[b] = False
        all_task_r.append(task_rs); all_int_r.append(int_rs)
        all_valid.append(valid)

        if not any(active):
            break

    # Stack tensors: leading dim = T (steps taken)
    pi_gd_t = torch.stack(all_pi_gd)    # [T, B, n_actions]
    pi_h_t  = torch.stack(all_pi_h)
    w_t     = torch.stack(all_w)         # [T, B]
    da_t    = torch.stack(all_da)
    val_t   = torch.stack(all_val)
    acts_t  = torch.stack(all_acts)      # [T, B]
    task_t  = torch.stack(all_task_r).to(dev)
    int_t   = torch.stack(all_int_r).to(dev)
    valid_t = torch.stack(all_valid)     # [T, B] bool

    total_gd = total_hab = None

    for b in range(B):
        mask = valid_t[:, b]             # [T] — steps where env b was active
        if not mask.any():
            continue

        task_rs_b = task_t[:, b][mask]
        int_rs_b  = int_t[:, b][mask]
        pi_gd_b   = pi_gd_t[:, b][mask]   # [T_b, n_actions]
        pi_h_b    = pi_h_t[:, b][mask]
        w_b       = w_t[:, b][mask]         # [T_b]
        da_b      = da_t[:, b][mask]
        val_b     = val_t[:, b][mask]
        acts_b    = acts_t[:, b][mask]      # [T_b]

        # ---- goal-directed A2C ----
        returns = discounted(task_rs_b.cpu().tolist(), cfg.gamma, dev)
        adv     = returns - val_b.detach()
        c       = w_b.unsqueeze(-1) * pi_gd_b + (1 - w_b.unsqueeze(-1)) * pi_h_b.detach()
        dist    = Categorical(logits=c)
        logps   = dist.log_prob(acts_b); ents = dist.entropy()

        expr_gate = w_b.mean().detach().clamp(0, 1)
        policy_l  = -(adv.detach() * logps).sum()
        critic_l  = cfg.value_coef * ((returns - val_b) ** 2).sum()
        entropy_l = -cfg.entropy_beta * ents.sum()
        da_pen    = da_lambda * (da_b ** 2).sum()
        gd_b = expr_gate * (policy_l + entropy_l) + critic_l + da_pen

        # ---- habitual: value-free APE + intrinsic efficiency ----
        returns_int = discounted(int_rs_b.cpu().tolist(), cfg.gamma, dev)
        adv_int     = returns_int - returns_int.mean().detach()
        ce_b        = F.cross_entropy(pi_h_b, acts_b, reduction="sum")
        logp_h      = Categorical(logits=pi_h_b).log_prob(acts_b)
        hab_b = cfg.ape_weight * ce_b + cfg.eff_weight * (-(adv_int.detach() * logp_h).sum())

        total_gd  = gd_b  if total_gd  is None else total_gd  + gd_b
        total_hab = hab_b if total_hab is None else total_hab + hab_b

    if total_gd is None:
        return 0.0, 0.0

    opt_gd.zero_grad(); total_gd.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(model.gd_params(), cfg.grad_clip); opt_gd.step()
    opt_hab.zero_grad(); total_hab.backward()
    torch.nn.utils.clip_grad_norm_(model.hab_params(), cfg.grad_clip); opt_hab.step()
    return float(total_gd.detach()), float(total_hab.detach())


def _record_episode(model, env, cfg):
    """Record one trajectory for the visualiser (single env, greedy)."""
    dev = cfg.device
    model.reset_state(1)
    env.reset()
    rec_pos, rec_w, rec_phase = [], [], []
    done = False; info = {"correct": False}
    with torch.no_grad():
        while not done:
            obs_t = _t(env.obs(), dev).unsqueeze(0)   # [1, obs_dim]
            out   = model.step(obs_t, obs_t)
            a     = int(out["combined"][0].argmax())
            rec_pos.append(list(env.pos))
            rec_w.append(round(float(out["w_gd"][0]), 3))
            rec_phase.append(env.phase)
            _, _, done, info = env.step(a)
    rec_pos.append(list(env.pos))
    return {"pos": rec_pos, "w": rec_w, "phase": rec_phase,
            "correct": bool(info["correct"]), "blocked": env.blocked}


def train(cfg, verbose=True):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    B   = cfg.batch_size
    # B independent environments — each gets its own rng stream
    envs = [TMazeFreeNav(cfg, np.random.default_rng(rng.integers(1 << 32)))
            for _ in range(B)]
    eval_env = TMazeFreeNav(cfg, np.random.default_rng(rng.integers(1 << 32)))

    model   = DualSystemModel(cfg).to(cfg.device)
    opt_gd  = torch.optim.Adam(model.gd_params(),  lr=cfg.lr_gd)
    opt_hab = torch.optim.Adam(model.hab_params(), lr=cfg.lr_hab)

    logs = {k: [] for k in ("episode", "combined_acc", "hab_solo_acc",
                             "gd_solo_acc", "w_gd", "da_recruit", "delay")}
    ckpt_learn = ckpt_maint = first_state = None
    train_trajs = []
    delay_advance_count = 0
    total_episodes = 0

    # Each iteration trains B episodes; loop until we hit cfg.episodes total
    iterations = (cfg.episodes + B - 1) // B
    for it in range(iterations):
        ep = total_episodes  # episode count at the START of this iteration
        total_episodes += B

        if ep <= cfg.da_warmup:
            da_lambda = 0.0
        elif ep <= cfg.da_warmup + cfg.da_ramp:
            da_lambda = cfg.da_cost_lambda * (ep - cfg.da_warmup) / cfg.da_ramp
        else:
            da_lambda = cfg.da_cost_lambda

        _train_batch(model, envs, cfg, opt_gd, opt_hab, da_lambda)

        # Trajectory logging (use eval_env for cleanliness)
        if (cfg.traj_log_every > 0
                and total_episodes % cfg.traj_log_every < B):
            traj = _record_episode(model, eval_env, cfg)
            traj["episode"] = total_episodes
            train_trajs.append(traj)

        # Periodic evaluation
        if total_episodes % cfg.eval_every < B or it == iterations - 1:
            comb = evaluate(model, eval_env, cfg, cfg.eval_trials)
            hab  = evaluate(model, eval_env, cfg, cfg.eval_trials, force_w=0.0)
            gd   = evaluate(model, eval_env, cfg, cfg.eval_trials, force_w=1.0)
            logs["episode"].append(total_episodes)
            logs["combined_acc"].append(comb["acc"])
            logs["hab_solo_acc"].append(hab["acc"])
            logs["gd_solo_acc"].append(gd["acc"])
            logs["w_gd"].append(comb["w_mean"])
            logs["da_recruit"].append(comb["da_mean"])
            logs["delay"].append(envs[0].current_delay)

            if first_state is None:
                first_state = copy.deepcopy(model.state_dict())
            if (ckpt_learn is None and comb["acc"] >= cfg.learn_combined_min
                    and hab["acc"] <= cfg.learn_habsolo_max):
                ckpt_learn = copy.deepcopy(model.state_dict())
            if hab["acc"] >= cfg.maint_solo_min and comb["acc"] >= cfg.maint_solo_min:
                ckpt_maint = copy.deepcopy(model.state_dict())

            # Delay curriculum: advance when both combined and habitual are strong
            if (comb["acc"] >= cfg.delay_advance_acc
                    and hab["acc"] >= cfg.delay_advance_acc):
                delay_advance_count += 1
            else:
                delay_advance_count = 0
            if delay_advance_count >= cfg.delay_advance_evals:
                for env in envs:
                    env.advance_delay()
                eval_env.advance_delay()
                delay_advance_count = 0
                if verbose:
                    print(f"  -> delay → {envs[0].current_delay}", flush=True)

            if verbose:
                print(f"ep {total_episodes:6d} | comb {comb['acc']:.2f} habSolo {hab['acc']:.2f} "
                      f"gdSolo {gd['acc']:.2f} | w_GD {comb['w_mean']:.2f} "
                      f"DA {comb['da_mean']:.2f} delay {envs[0].current_delay}", flush=True)

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

    return model, logs, ckpt_learn, ckpt_maint, train_trajs, envs[0].current_delay
