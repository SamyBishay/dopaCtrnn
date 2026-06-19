"""Central configuration. All hyperparameters are PINNED here (Claude's choices,
since the from-scratch build had no values to inherit). Tune freely; the protocol
explicitly treats a learning failure (H1) as a tuning problem, not a scientific one.
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- task (environment.py) ----
    delay_steps: int = 10          # protocol: report at a non-trivial delay; 0 = debug only
    max_test_steps: int = 12       # timeout in the test phase -> counts as incorrect
    sample_reward: float = 0.2     # delivered at the open arm end during forced sample
    test_reward: float = 1.0       # delivered for the correct (non-match) choice
    step_cost: float = 0.02        # per movement step
    wait_cost: float = 0.01        # per delay step
    completion_bonus: float = 0.10 # intrinsic, reaching ANY arm end (value-free signal)
    n_actions: int = 4             # 0=N 1=S 2=E 3=W

    # ---- network sizes (model.py) ----
    n_gd: int = 64                 # goal-directed CTRNN units (first half D1/phasic, second half D2/tonic)
    n_hab: int = 64                # habitual CTRNN units
    dt: float = 1.0                # Euler step for CTRNN
    tau_fast: float = 2.0          # fast units (action/decision dynamics)
    tau_slow: float = 25.0         # slow units (working memory across the delay)
    tonic_kappa: float = 0.10      # low-pass rate for tonic DA (slow timescale)
    gain_base: float = 0.5         # expression gain at DA=0  (W_eff = gain * W)
    gain_da: float = 0.5           # extra expression gain per unit of DA
    wgd_alpha: float = 6.0         # w_GD = sigmoid(alpha * DA + bias)  (learnable, this is the init)
    wgd_bias: float = -2.0         # init so w_GD -> low when DA -> 0

    # ---- training (train.py) ----
    episodes: int = 5000
    gamma: float = 0.95
    lr_gd: float = 5e-4
    lr_hab: float = 5e-4
    entropy_beta: float = 0.03
    value_coef: float = 0.5
    da_cost_lambda: float = 0.02   # penalty on da_request^2  -> "minimise its own request"
    da_warmup: int = 1500          # episodes with NO DA penalty (free acquisition)
    da_ramp: int = 1500            # episodes to linearly ramp the penalty to full (induce handoff)
    grad_clip: float = 1.0
    ape_weight: float = 1.0        # habitual imitation (action prediction error) weight
    eff_weight: float = 0.10       # habitual intrinsic-efficiency weight (NOT task reward)

    # ---- evaluation / analysis ----
    eval_every: int = 100
    eval_trials: int = 200         # trials per periodic eval
    final_trials: int = 1000       # H1 binomial test
    attractor_trials: int = 300    # trials for Fig 5 (PCA + decoder)

    # ---- trajectory visualization ----
    traj_log_every: int = 25       # log one (subsampled) training trajectory every N episodes
    traj_eval_trials: int = 60     # trajectories replayed at the maintenance checkpoint

    # checkpoint selection (phase definitions, by eval'd accuracy)
    learn_combined_min: float = 0.60   # learning phase: combined competent...
    learn_habsolo_max: float = 0.60    # ...but habitual not yet (GD is carrying)
    maint_solo_min: float = 0.80       # maintenance: habitual-solo >= this (habitual carrying)

    # ---- OPEN QUESTION #1 -------------------------------------------------
    # How is the DA-request signal trained?
    #   "a2c_coupled" (default): the DA-request is shaped by the A2C advantage
    #       (reward-coupled). This is the implemented, runnable path. It carries the
    #       circularity risk flagged in the mEmoire (the handoff is partly reward-driven).
    #   "local_pe": train the DA-request from a purely LOCAL prediction error instead.
    #       NOT implemented on purpose -- specifying that error is the open design
    #       decision, and choosing it is yours, not the code's. See model/train hooks.
    da_request_training: str = "a2c_coupled"

    seed: int = 0
    device: str = "cpu"            # these nets are tiny; CPU is fine and reproducible

    def to_dict(self):
        return asdict(self)
