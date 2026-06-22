"""Run one seed end-to-end.

  python run_experiment.py --seed 0 --outdir results
  python run_experiment.py --seed 0 --outdir results --untrained      # H7 control
  python run_experiment.py --seed 0 --outdir results --fixed-points    # + WM fixed points
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

# Pin BLAS to the physical core count. This CPU is BLAS-3 bound; using more
# threads than physical cores (i.e. spilling onto SMT siblings) thrashes and
# REDUCES throughput. os.cpu_count() reports logical cores, so halve it when
# SMT is active. Set DOPA_NUM_THREADS to override.
def _physical_cores():
    n = os.environ.get("DOPA_NUM_THREADS")
    if n:
        return max(1, int(n))
    try:
        import multiprocessing as _mp
        logical = _mp.cpu_count()
    except Exception:
        logical = os.cpu_count() or 1
    # AMD/Intel SMT is 2-way; assume half are physical when even and > 1.
    return max(1, logical // 2) if logical > 1 and logical % 2 == 0 else logical
torch.set_num_threads(_physical_cores())

from config import Config
from environment import TMazeFreeNav, maze_layout
from model import DualSystemModel
from train import train
import analysis as A
import figures as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--fixed-points", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-gd", type=int, default=None)
    ap.add_argument("--n-hab", type=int, default=None)
    ap.add_argument("--hab-rank", type=int, default=None,
                    help="0=full-rank habitual (default); >0 = low-rank W_rec")
    ap.add_argument("--len-edge", type=int, default=None,
                    help="inner grid edge (odd, >=5); grid width = len_edge+2")
    ap.add_argument("--difficulty", type=int, default=None, choices=[0, 1, 2],
                    help="0=easy 1=medium 2=hard (stem length / grid height)")
    ap.add_argument("--gae-lambda", type=float, default=None)
    ap.add_argument("--no-ret-norm", action="store_true",
                    help="disable return normalisation")
    ap.add_argument("--ape-decay", action="store_true",
                    help="fade APE weight once habitual surpasses combined")
    ap.add_argument("--ckpt-every", type=int, default=None,
                    help="write a resumable checkpoint every N episodes (0=off)")
    ap.add_argument("--da-split", action="store_true",
                    help="split expression-gain and arbitration into independent "
                         "signals (required for E2/E3 to be non-vacuous)")
    ap.add_argument("--gate-mode", type=str, default=None,
                    choices=["expression", "scheduled"],
                    help="E2: DA-driven gate (default) vs a fixed schedule")
    ap.add_argument("--da-components", type=str, default=None,
                    choices=["both", "gain_only", "weights_only"],
                    help="E3: which DA component is active (Naudé decomposition)")
    ap.add_argument("--habit-rule", type=str, default=None,
                    choices=["value_free", "value_coupled"],
                    help="E4: value-free APE habit (default) vs reward-coupled")
    ap.add_argument("--habit-obs", type=str, default=None,
                    choices=["position_free", "allocentric"],
                    help="E5: position-free habit observation (default) vs allocentric")
    ap.add_argument("--da-gain-mode", type=str, default=None,
                    choices=["output", "recurrent"],
                    help="E9: DA gain on output readout (default) or inside recurrence (Naudé W_eff)")
    ap.add_argument("--tau-mode", type=str, default=None,
                    choices=["mixed", "uniform"],
                    help="E6/E7: mixed=fast+slow tau init (default); uniform=all same tau")
    ap.add_argument("--da-tau", action="store_true",
                    help="E6/E7/E9: enable DA-modulated effective integration tau")
    ap.add_argument("--tau-uniform", type=float, default=None,
                    help="Initial tau when --tau-mode=uniform (default 10.0)")
    ap.add_argument("--da-tau-gain", type=float, default=None,
                    help="Log-space tau shift magnitude per unit DA (default 0.5)")
    args = ap.parse_args()

    cfg = Config(seed=args.seed, device=args.device)
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.n_gd is not None:
        cfg.n_gd = args.n_gd
    if args.n_hab is not None:
        cfg.n_hab = args.n_hab
    if args.hab_rank is not None:
        cfg.hab_rank = args.hab_rank
    if args.len_edge is not None:
        cfg.len_edge = args.len_edge
    if args.difficulty is not None:
        cfg.difficulty = args.difficulty
    if args.gae_lambda is not None:
        cfg.gae_lambda = args.gae_lambda
    if args.no_ret_norm:
        cfg.ret_norm = False
    if args.ape_decay:
        cfg.ape_decay = True
    if args.da_split:
        cfg.da_split = True
    if args.gate_mode is not None:
        cfg.gate_mode = args.gate_mode
    if args.da_components is not None:
        cfg.da_components = args.da_components
    if args.habit_rule is not None:
        cfg.habit_rule = args.habit_rule
    if args.habit_obs is not None:
        cfg.habit_obs = args.habit_obs
        cfg.__post_init__()  # re-sync obs_dim_hab now that habit_obs is set
    if args.da_gain_mode is not None:
        cfg.da_gain_mode = args.da_gain_mode
    if args.tau_mode is not None:
        cfg.tau_mode = args.tau_mode
        cfg.__post_init__()
    if args.da_tau:
        cfg.da_tau = True
    if args.tau_uniform is not None:
        cfg.tau_uniform = args.tau_uniform
    if args.da_tau_gain is not None:
        cfg.da_tau_gain = args.da_tau_gain

    tag = f"seed{args.seed}" + ("_untrained" if args.untrained else "")
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    if args.ckpt_every is not None:
        cfg.ckpt_every = args.ckpt_every
    if cfg.ckpt_every > 0:
        cfg.ckpt_path = os.path.join(outdir, "train_state.pt")
    eval_env = TMazeFreeNav(cfg, np.random.default_rng(args.seed + 9_999))

    if args.untrained:
        torch.manual_seed(args.seed)
        model = DualSystemModel(cfg).to(cfg.device)
        logs = None
        ckpt_learn = ckpt_maint = copy.deepcopy(model.state_dict())
        train_trajs = []
        eval_env.current_delay = cfg.delay_max   # compare H7 at same delay as trained model
    else:
        model, logs, ckpt_learn, ckpt_maint, train_trajs, final_delay = train(cfg)
        eval_env.current_delay = final_delay   # eval at the delay reached during training
        torch.save(ckpt_learn, os.path.join(outdir, "ckpt_learning.pt"))
        torch.save(ckpt_maint, os.path.join(outdir, "ckpt_maintenance.pt"))

    results = {"seed": args.seed, "untrained": args.untrained, "config": cfg.to_dict()}

    model.load_state_dict(ckpt_maint)
    results["h1"] = A.h1_learning(model, eval_env, cfg)
    if logs is not None:
        results["logs"] = logs
        results["h2"] = A.h2_handoff(logs)
    results["h3"] = A.h3_devaluation(model, ckpt_learn, ckpt_maint, eval_env, cfg)
    results["h4"] = A.h4_lesions(model, ckpt_learn, ckpt_maint, eval_env, cfg)
    results["h5"] = A.h5_reactivation(model, ckpt_maint, eval_env, cfg)
    results["h6"] = A.h6_attractor(model, ckpt_maint, eval_env, cfg)
    if args.fixed_points:
        model.load_state_dict(ckpt_maint)
        results["fixed_points"] = A.fixed_points(model, eval_env, cfg)

    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    # ---- trajectories for the visualizer ----
    model.load_state_dict(ckpt_maint)
    eval_trajs = A.eval_trajectories(model, eval_env, cfg, cfg.traj_eval_trials,
                                     greedy=not args.untrained)
    with open(os.path.join(outdir, "trajectories.json"), "w") as f:
        json.dump({"seed": args.seed, "untrained": args.untrained,
                   "maze": maze_layout(cfg), "train": train_trajs, "eval": eval_trajs},
                  f, default=float)

    # ---- per-seed figures ----
    if logs is not None:
        F.fig1_handoff(logs, outdir, results.get("h2"))
    F.fig2_devaluation(results["h3"], outdir)
    F.fig3_lesions(results["h4"], outdir)
    F.fig4_reactivation(results["h5"], outdir, results["h3"])
    F.fig5_attractor(results["h6"], outdir)

    print(f"\n[{tag}] H1 acc={results['h1']['accuracy']:.3f} "
          f"(p={results['h1']['p_value']:.1e}) | "
          f"H5 DA-rise={results['h5']['da_request_delta']:+.3f} "
          f"({'PASS' if results['h5']['pass'] else 'FAIL'}) | "
          f"H6 decoder={results['h6']['decoder_acc']:.2f}")
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
