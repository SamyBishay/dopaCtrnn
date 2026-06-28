"""Training — vectorised over B parallel environments.

Each training iteration steps all B environments in lockstep. The batch
dimension lets PyTorch dispatch [B × n] matrix multiplies instead of
n-length vector-matrix products, saturating BLAS and cutting wall-time
by ~10-20× vs. single-env sequential training.

Two optimizers (goal-directed, habitual) are updated once per iteration
over the summed losses from all B completed episodes.
"""
import copy
import os
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

# Pin BLAS to physical-core count (see run_experiment.py for rationale).
# Idempotent: safe even when run_experiment.py has already set it.
def _physical_cores():
    n = os.environ.get("DOPA_NUM_THREADS")
    if n:
        return max(1, int(n))
    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical > 1 and logical % 2 == 0 else logical
torch.set_num_threads(_physical_cores())

from environment import TMazeFreeNav
from model import DualSystemModel
from analysis import evaluate, evaluate_vec, _t


def discounted_batch(rewards, gamma, valid):
    """Discounted returns over [T, B] without per-b Python loops.

    rewards, valid: [T, B] (valid is float/bool mask of active steps).
    Returns [T, B]. The valid mask resets the accumulator across the
    (already right-padded) inactive tail of each column.
    """
    T = rewards.shape[0]
    out = torch.zeros_like(rewards)
    R = torch.zeros(rewards.shape[1], device=rewards.device)
    vf = valid.float()
    for t in range(T - 1, -1, -1):          # T iters over [B], not B×T
        R = rewards[t] + gamma * R * vf[t]
        out[t] = R
    return out


def gae_batch(rewards, values, gamma, lam, valid):
    """Generalised Advantage Estimation over [T, B].

    rewards, values, valid: [T, B]. Bootstrap value past the last active
    step is zero (episodes run to a terminal state). Returns (adv, ret),
    each [T, B], with ret = adv + values. lam=1.0 recovers Monte-Carlo.
    """
    T = rewards.shape[0]
    vf = valid.float()
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)
    next_val = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_val * vf[t] - values[t]
        gae = delta + gamma * lam * gae * vf[t]
        adv[t] = gae
        next_val = values[t]
    ret = adv + values
    return adv, ret


class RunningNorm:
    """Rolling mean/std over a fixed window of scalar returns (for GD return
    normalisation). Mirrors the supervisor's deque-based standardisation."""
    def __init__(self, window):
        from collections import deque
        self.buf = deque(maxlen=window)

    def update(self, vals):
        self.buf.extend(vals)

    def stats(self):
        if len(self.buf) < 2:
            return 0.0, 1.0
        arr = np.fromiter(self.buf, dtype=np.float64)
        return float(arr.mean()), max(float(arr.std()), 1.0)


def _train_batch(model, venv, cfg, opt_gd, opt_hab, force_w=None,
                 ret_norm=None, ape_scale=1.0):
    """One vectorised training iteration over B parallel episodes (single venv).
    Returns (loss_gd, loss_hab, n_correct, e_rpe, mean_steps) where n_correct
    is the number of episodes in this batch that reached the correct goal arm."""
    B   = venv.B
    dev = cfg.device
    model.reset_state(B)
    venv.reset()

    # Per-step accumulators — each is a list of [B, ...] tensors
    all_pi_gd, all_w, all_da = [], [], []
    all_val, all_acts = [], []
    all_task_r, all_int_r, all_valid = [], [], []
    # Detached pi_h, kept only to reconstruct the combined GD logits (the
    # habitual policy is a no-grad input to the GD A2C term). The grad-carrying
    # pi_h is consumed per-step by the KL update, not stored.
    all_pi_h_det = []

    active = np.ones(B, dtype=bool)
    n_correct = 0
    step_counts_done = []
    kl_step_accum = None
    hab_step_count = 0
    hab_loss_total = 0.0

    for _ in range(cfg.max_episode_steps):
        obs_t      = _t(venv.obs(),     dev)      # [B, obs_dim]
        obs_hab_t  = _t(venv.obs_hab(), dev)      # [B, obs_dim_hab]
        out        = model.step(obs_t, obs_hab_t, force_w=force_w)
        acts   = Categorical(logits=out["combined"]).sample()  # [B]

        all_pi_gd.append(out["pi_gd"]); all_pi_h_det.append(out["pi_h"].detach())
        all_w.append(out["w_gd"]);      all_da.append(out["da_request"])
        all_val.append(out["value"]);   all_acts.append(acts)

        valid_np = active.copy()                  # steps where env was active
        tr, ir, done, info = venv.step(acts.cpu().numpy())
        newly_done = valid_np & done
        if newly_done.any():
            n_correct += int(info["correct"][newly_done].sum())
            step_counts_done.extend(int(venv.step_count[b]) for b in np.where(newly_done)[0])
        # rewards are zero for already-done envs by construction in venv.step
        all_task_r.append(torch.from_numpy(tr.astype(np.float32)))
        all_int_r.append(torch.from_numpy(ir.astype(np.float32)))
        all_valid.append(torch.from_numpy(valid_np))

        # Per-step habitual KL update (per-step online signal, Villet-faithful)
        if not getattr(cfg, "freeze_hab", False):
            active_mask = torch.from_numpy(valid_np.astype(np.float32)).to(dev)  # [B]
            if active_mask.any():
                pi_h_step = out["pi_h"]          # [B, n_actions], has grad
                pi_c_step = out["combined"].detach()  # [B, n_actions], no grad
                kl_step = F.kl_div(
                    F.log_softmax(pi_h_step, dim=-1),
                    F.softmax(pi_c_step, dim=-1),
                    reduction='batchmean',
                ) * ape_scale
                kl_step_accum = kl_step_accum + kl_step if kl_step_accum is not None else kl_step
                hab_step_count += 1
                if hab_step_count % cfg.hab_update_freq == 0:
                    opt_hab.zero_grad()
                    kl_step_accum.backward()
                    torch.nn.utils.clip_grad_norm_(model.hab_params(), cfg.grad_clip)
                    opt_hab.step()
                    # Detach h_hab to cut the BPTT graph: each per-step update is
                    # a fresh backward pass; without detach the freed graph causes
                    # RuntimeError on the next step's backward.
                    model.h_hab = model.h_hab.detach()
                    kl_step_accum = None
                    hab_step_count = 0
                hab_loss_total += float(kl_step.detach())

        active &= ~done

        if not active.any():
            break

    if kl_step_accum is not None and not getattr(cfg, "freeze_hab", False):
        opt_hab.zero_grad()
        kl_step_accum.backward()
        torch.nn.utils.clip_grad_norm_(model.hab_params(), cfg.grad_clip)
        opt_hab.step()
        model.h_hab = model.h_hab.detach()

    # Stack tensors: leading dim = T (steps taken)
    pi_gd_t   = torch.stack(all_pi_gd)    # [T, B, n_actions]
    pi_h_det_t = torch.stack(all_pi_h_det)  # [T, B, n_actions], no grad
    w_t       = torch.stack(all_w)         # [T, B]
    da_t    = torch.stack(all_da)
    val_t   = torch.stack(all_val)
    acts_t  = torch.stack(all_acts)      # [T, B]
    task_t  = torch.stack(all_task_r).to(dev)
    int_t   = torch.stack(all_int_r).to(dev)
    valid_t = torch.stack(all_valid).to(dev)   # [T, B] bool

    valid_f = valid_t.float()                            # [T, B]
    total_valid = valid_f.sum().clamp_min(1.0)

    # ===== goal-directed A2C, fully vectorised over B =====
    # GAE(λ) advantage and bootstrap-free returns on the per-step task reward.
    adv, returns = gae_batch(task_t, val_t.detach(), cfg.gamma,
                             cfg.gae_lambda, valid_t)       # [T, B], [T, B]

    e_rpe = float(adv[valid_t].abs().mean()) if valid_t.any() else 0.0

    # Return normalisation: standardise the *targets* with a rolling window.
    if cfg.ret_norm and ret_norm is not None:
        ret_vals = returns[valid_t].detach().cpu().numpy().tolist()
        ret_norm.update(ret_vals)
        r_mean, r_std = ret_norm.stats()
        returns_n = (returns - r_mean) / r_std
    else:
        returns_n = returns

    # Advantage normalisation (masked mean/std over active steps only).
    adv_masked = adv[valid_t].detach()
    a_mean = adv_masked.mean() if adv_masked.numel() else adv.new_zeros(())
    a_std  = adv_masked.std()  if adv_masked.numel() > 1 else adv.new_ones(())
    adv_n  = (adv - a_mean) / (a_std + 1e-8)

    c     = w_t.unsqueeze(-1) * pi_gd_t + (1 - w_t.unsqueeze(-1)) * pi_h_det_t
    dist  = Categorical(logits=c)
    logps = dist.log_prob(acts_t)                          # [T, B]
    ents  = dist.entropy()                                 # [T, B]

    expr_gate = (w_t * valid_f).sum() / total_valid        # global masked gate
    expr_gate = expr_gate.detach().clamp(0, 1)

    policy_l  = -((adv_n.detach() * logps) * valid_f).sum()
    critic_l  = cfg.value_coef * (((returns_n - val_t) ** 2) * valid_f).sum()
    entropy_l = -cfg.entropy_beta * (ents * valid_f).sum()
    total_gd  = expr_gate * (policy_l + entropy_l) + critic_l

    opt_gd.zero_grad(); total_gd.backward()
    torch.nn.utils.clip_grad_norm_(model.gd_params(), cfg.grad_clip); opt_gd.step()

    mean_steps = float(np.mean(step_counts_done)) if step_counts_done else float('nan')
    return float(total_gd.detach()), hab_loss_total, n_correct, e_rpe, mean_steps


def _record_episode(model, env, cfg):
    """Record one trajectory for the visualiser (single env, greedy)."""
    dev = cfg.device
    model.reset_state(1)
    env.reset()
    rec_pos, rec_w, rec_phase = [], [], []
    done = False; info = {"correct": False}
    with torch.no_grad():
        while not done:
            obs_t     = _t(env.obs(),     dev).unsqueeze(0)   # [1, obs_dim]
            obs_hab_t = _t(env.obs_hab(), dev).unsqueeze(0)   # [1, obs_dim_hab]
            out       = model.step(obs_t, obs_hab_t)
            a     = int(out["combined"][0].argmax())
            rec_pos.append(list(env.pos))
            rec_w.append(round(float(out["w_gd"][0]), 3))
            rec_phase.append(env.phase)
            _, _, done, info = env.step(a)
    rec_pos.append(list(env.pos))
    return {"pos": rec_pos, "w": rec_w, "phase": rec_phase,
            "correct": bool(info["correct"]), "blocked": env.blocked}


def train(cfg, verbose=True):
    _t0 = time.time()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    B   = cfg.batch_size
    # One vectorised environment of width B (replaces B separate env objects).
    from environment import TMazeVecEnv
    venv = TMazeVecEnv(cfg, np.random.default_rng(rng.integers(1 << 32)), batch_size=B)
    eval_env = TMazeFreeNav(cfg, np.random.default_rng(rng.integers(1 << 32)))

    model   = DualSystemModel(cfg).to(cfg.device)
    opt_gd  = torch.optim.Adam(model.gd_params(),  lr=cfg.lr_gd)
    opt_hab = torch.optim.Adam(model.hab_params(), lr=cfg.lr_hab)

    logs = {k: [] for k in ("episode", "combined_acc", "hab_solo_acc",
                             "gd_solo_acc", "w_gd", "da_recruit", "delay",
                             "fixed_delay_acc", "rolling_acc", "steps_to_goal")}
    rolling_buf = deque(maxlen=cfg.rolling_window)
    ckpt_learn = ckpt_maint = first_state = None
    train_trajs = []
    total_episodes = 0
    start_it = 0
    ret_norm = RunningNorm(cfg.ret_norm_window)
    hab_solo_acc = 0.0   # last evaluated habitual-solo accuracy (for APE-decay)
    delay_advance_count = 0  # consecutive evals at delay_advance_acc threshold
    # RPE-based DA gate state
    rpe_mu  = 0.0
    rpe_var = 1e-6
    da_signal = 0.0
    current_w_gd = 1.0   # starts fully open; closes once RPE criterion met
    mean_steps = float('nan')
    comb_acc     = 0.0
    villet_learn_hist = []  # track eval windows where combined_acc >= threshold
    villet_maint_count = 0  # counter for consecutive evals at maintenance threshold

    # ---- resume from a mid-run checkpoint, if present ----
    if cfg.ckpt_every > 0 and cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
        st = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt_gd.load_state_dict(st["opt_gd"])
        opt_hab.load_state_dict(st["opt_hab"])
        torch.set_rng_state(st["torch_rng"])
        rng = np.random.default_rng()
        rng.bit_generator.state = st["np_rng"]
        total_episodes      = st["total_episodes"]
        start_it            = st["iteration"] + 1
        logs                = st["logs"]
        if "rolling_acc" not in logs:          # old checkpoint compatibility
            logs["rolling_acc"] = [None] * len(logs["episode"])
        ret_norm.buf.extend(st.get("ret_norm_buf", []))
        ckpt_learn  = st.get("ckpt_learn"); ckpt_maint = st.get("ckpt_maint")
        first_state = st.get("first_state"); train_trajs = st.get("train_trajs", [])
        cur_delay = st["current_delay"]
        venv.current_delay = cur_delay
        eval_env.current_delay = cur_delay
        if verbose:
            print(f"[resume] from {cfg.ckpt_path}: ep={total_episodes} "
                  f"it={start_it} delay={cur_delay}", flush=True)

    # Each iteration trains B episodes; loop until we hit cfg.episodes total
    iterations = (cfg.episodes + B - 1) // B
    for it in range(start_it, iterations):
        ep = total_episodes  # episode count at the START of this iteration
        total_episodes += B

        # APE-decay (teacher-fade analogue): fade the action-prediction-error
        # weight once the habitual solo policy has surpassed the combined policy.
        if cfg.ape_decay:
            surplus = hab_solo_acc - comb_acc - cfg.ape_decay_margin
            if surplus > 0:
                ape_scale = max(cfg.ape_min_scale,
                                1.0 - cfg.ape_decay_rate * surplus)
            else:
                ape_scale = 1.0
        else:
            ape_scale = 1.0

        _, _, n_correct, e_rpe, mean_steps = _train_batch(
            model, venv, cfg, opt_gd, opt_hab, force_w=current_w_gd,
            ret_norm=ret_norm, ape_scale=ape_scale)

        # RPE-based DA gate: normalise RPE by running stats, pass through sigmoid
        rpe_mu  = 0.99 * rpe_mu  + 0.01 * e_rpe
        rpe_var = 0.99 * rpe_var + 0.01 * (e_rpe - rpe_mu) ** 2
        z = (e_rpe - rpe_mu) / (rpe_var ** 0.5 + 1e-6)
        if z > 0:
            da_signal = float(torch.sigmoid(torch.tensor(z)))
        else:
            da_signal = 0.999 * da_signal
        # Gate stays open until hab is competent or warmup period hasn't elapsed
        if it < cfg.rpe_warmup_iters or hab_solo_acc < cfg.early_stop_acc:
            current_w_gd = 1.0
        else:
            current_w_gd = float(torch.sigmoid(
                torch.tensor(cfg.rpe_alpha * da_signal + cfg.rpe_bias)))

        rolling_buf.extend([1] * n_correct + [0] * (B - n_correct))

        # Early stopping: 99% rolling accuracy over the last rolling_window episodes.
        if (len(rolling_buf) >= cfg.rolling_window
                and sum(rolling_buf) / len(rolling_buf) >= cfg.early_stop_acc):
            if verbose:
                racc = sum(rolling_buf) / len(rolling_buf)
                print(f"  -> early stop at ep {total_episodes}: "
                      f"rolling acc {racc:.3f} >= {cfg.early_stop_acc}", flush=True)
            break

        # Trajectory logging (use eval_env for cleanliness)
        if (cfg.traj_log_every > 0
                and total_episodes % cfg.traj_log_every < B):
            traj = _record_episode(model, eval_env, cfg)
            traj["episode"] = total_episodes
            train_trajs.append(traj)

        # Periodic evaluation
        if total_episodes % cfg.eval_every < B or it == iterations - 1:
            # Same seed for comb/hab/gd so they see identical trials (comparable).
            # Keyed to total_episodes so each eval window sees a different draw.
            ev_seed = total_episodes
            comb = evaluate_vec(model, eval_env, cfg, cfg.eval_trials, seed=ev_seed)
            hab  = evaluate_vec(model, eval_env, cfg, cfg.eval_trials, force_w=0.0, seed=ev_seed)
            gd   = evaluate_vec(model, eval_env, cfg, cfg.eval_trials, force_w=1.0, seed=ev_seed)
            comb_acc = comb["acc"]; hab_solo_acc = hab["acc"]   # for APE-decay
            r_acc = (sum(rolling_buf) / len(rolling_buf)) if rolling_buf else 0.0
            logs["episode"].append(total_episodes)
            logs["combined_acc"].append(comb["acc"])
            logs["hab_solo_acc"].append(hab["acc"])
            logs["gd_solo_acc"].append(gd["acc"])
            logs["w_gd"].append(comb["w_mean"])
            logs["da_recruit"].append(comb["da_mean"])
            logs["delay"].append(venv.current_delay)
            logs["rolling_acc"].append(r_acc)
            logs["steps_to_goal"].append(mean_steps)

            at_ceiling = venv.current_delay >= cfg.delay

            # Curriculum advance: when hab-solo exceeds threshold for delay_advance_evals
            # consecutive evals, step the delay toward cfg.delay (ceiling).
            if not at_ceiling:
                if hab["acc"] >= cfg.delay_advance_acc:
                    delay_advance_count += 1
                    if delay_advance_count >= cfg.delay_advance_evals:
                        venv.advance_delay()
                        eval_env.advance_delay()
                        delay_advance_count = 0
                        if verbose:
                            print(f"  -> delay advanced to {venv.current_delay}", flush=True)
                else:
                    delay_advance_count = 0
                at_ceiling = venv.current_delay >= cfg.delay

            # Fixed-delay eval only once at ceiling (Villet comparison pinned to cfg.delay).
            if at_ceiling:
                fd = evaluate_vec(model, eval_env, cfg, cfg.eval_trials,
                                  fixed_delay=cfg.fixed_eval_delay, seed=ev_seed)
                logs["fixed_delay_acc"].append(fd["acc"])
            else:
                logs["fixed_delay_acc"].append(None)

            if first_state is None:
                first_state = copy.deepcopy(model.state_dict())
            # Gate checkpoint captures on curriculum ceiling: measuring the handoff
            # while the delay is still advancing conflates curriculum progress with
            # the goal-directed→habitual transition (supervisor pattern, line ~1127).
            if at_ceiling:
                # ---- Villet learning checkpoint: 2 non-consecutive evals at ≥70% ----
                if ckpt_learn is None:
                    if comb["acc"] >= cfg.villet_learn_acc:
                        villet_learn_hist.append(total_episodes)
                    else:
                        villet_learn_hist = []
                    
                    # Capture when we have 2 qualifying evals with at least 1 gap between them
                    # "non-consecutive" means last - first >= 1, i.e., not adjacent indices
                    if len(villet_learn_hist) >= 2:
                        if len(villet_learn_hist) - 1 >= 1:  # indices 0, 1 → at least 1 gap
                            ckpt_learn = copy.deepcopy(model.state_dict())
                
                # ---- Villet maintenance checkpoint: 3 consecutive evals at ≥80% ----
                if (comb["acc"] >= cfg.villet_maint_acc
                        and hab["acc"] >= cfg.villet_maint_hab_min):
                    villet_maint_count += 1
                else:
                    villet_maint_count = 0
                
                if villet_maint_count >= cfg.villet_maint_days:
                    ckpt_maint = copy.deepcopy(model.state_dict())

            if verbose:
                fd_str = f" fixedDA {logs['fixed_delay_acc'][-1]:.2f}" if at_ceiling else ""
                print(f"ep {total_episodes:6d} | comb {comb['acc']:.2f} roll {r_acc:.2f} "
                      f"habSolo {hab['acc']:.2f} gdSolo {gd['acc']:.2f} | "
                      f"w_GD {comb['w_mean']:.2f} DA {comb['da_mean']:.2f} "
                      f"delay {venv.current_delay}{fd_str}", flush=True)

        # ---- resumable checkpoint (atomic write) ----
        if cfg.ckpt_every > 0 and cfg.ckpt_path and total_episodes % cfg.ckpt_every < B:
            state = {
                "model": model.state_dict(),
                "opt_gd": opt_gd.state_dict(), "opt_hab": opt_hab.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "np_rng": rng.bit_generator.state,
                "total_episodes": total_episodes, "iteration": it,
                "logs": logs,
                "ret_norm_buf": list(ret_norm.buf),
                "ckpt_learn": ckpt_learn, "ckpt_maint": ckpt_maint,
                "first_state": first_state, "train_trajs": train_trajs,
                "current_delay": venv.current_delay,
            }
            tmp = cfg.ckpt_path + ".tmp"
            torch.save(state, tmp)
            os.replace(tmp, cfg.ckpt_path)   # atomic: never leaves a half-written ckpt
            if verbose:
                print(f"  -> checkpoint @ ep {total_episodes} -> {cfg.ckpt_path}", flush=True)

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

    return model, logs, ckpt_learn, ckpt_maint, train_trajs, venv.current_delay, time.time() - _t0
