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
    args = ap.parse_args()

    cfg = Config(seed=args.seed, device=args.device)
    if args.episodes is not None:
        cfg.episodes = args.episodes
    if args.n_gd is not None:
        cfg.n_gd = args.n_gd
    if args.n_hab is not None:
        cfg.n_hab = args.n_hab

    tag = f"seed{args.seed}" + ("_untrained" if args.untrained else "")
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)
    eval_env = TMazeFreeNav(cfg, np.random.default_rng(args.seed + 9_999))

    if args.untrained:
        torch.manual_seed(args.seed)
        model = DualSystemModel(cfg).to(cfg.device)
        logs = None
        ckpt_learn = ckpt_maint = copy.deepcopy(model.state_dict())
        train_trajs = []
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
                   "maze": maze_layout(), "train": train_trajs, "eval": eval_trajs},
                  f, default=float)

    # ---- per-seed figures ----
    if logs is not None:
        F.fig1_handoff(logs, outdir)
    F.fig2_devaluation(results["h3"], outdir)
    F.fig3_lesions(results["h4"], outdir)
    F.fig4_reactivation(results["h5"], outdir)
    F.fig5_attractor(results["h6"], outdir)

    print(f"\n[{tag}] H1 acc={results['h1']['accuracy']:.3f} "
          f"(p={results['h1']['p_value']:.1e}) | "
          f"H5 DA-rise={results['h5']['da_request_delta']:+.3f} "
          f"({'PASS' if results['h5']['pass'] else 'FAIL'}) | "
          f"H6 decoder={results['h6']['decoder_acc']:.2f}")
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
