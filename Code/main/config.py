"""Central configuration. All hyperparameters are pinned here.
Tune freely; H1 learning failure is a tuning problem, not scientific.
"""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- task (environment.py) ----
    len_edge: int      = 7            # inner grid edge (odd, >=5); width = len_edge+2
                                      # NOTE: len_edge=5 needs difficulty>=1 (else no stem)
    difficulty: int    = 0            # 0=easy 1=medium 2=hard — sets stem length (height)
    sig_val: float     = 0.25         # phase-signal amplitude (was 1/(COLS-1) at 5-wide)
    delay_start: int   = 5            # delay steps before choice; gives agent time to leave arm end
    delay_max: int     = 15           # max delay steps (working memory challenge)
    delay_advance_acc: float = 0.75   # accuracy threshold (combined) to lengthen delay
    delay_advance_evals: int = 3      # consecutive evals above threshold → advance
    max_episode_steps: int = 80       # hard timeout per trial (all phases)
    test_reward: float = 1.0          # reward for correct (non-match) choice
    step_cost: float   = 0.02         # per movement step (all phases)
    wait_cost: float   = 0.02         # per WAIT action — same as step_cost so WAIT isn't cheaper
    junction_bonus: float = 0.30      # task_r shaping: pre_sample→sample (reaching junction)
    arm_end_bonus: float  = 0.20      # task_r shaping: sample→delay (reaching arm end)
    completion_bonus: float = 0.10    # intrinsic, reaching ANY arm end (value-free)
    n_actions: int     = 5            # 0=N 1=S 2=E 3=W 4=WAIT
    obs_dim: int       = 6            # GD observation: [x, y, 0, sig_L, sig_R, sig_choice]
    obs_dim_hab: int   = 4            # Hab observation: [0, sig_L, sig_R, sig_choice] (no position)

    # ---- network sizes (model.py) ----
    n_gd: int    = 256                # goal-directed CTRNN units (D1/phasic + D2/tonic halves)
    n_hab: int   = 256                # habitual CTRNN units
    hab_rank: int = 0                 # 0 = full-rank habitual W (default, original behaviour);
                                      # >0 = low-rank W_rec = (m @ n.T)/n_hab, rank=hab_rank.
                                      # Makes "habit is low-dimensional" structural. Try 2-8.
    dt: float    = 1.0
    tau_fast: float  = 2.0            # fast units (action/decision)
    tau_slow: float  = 25.0           # slow units (working memory)
    tonic_kappa: float = 0.10         # low-pass rate for tonic DA
    gain_base: float   = 0.5          # expression gain at DA=0 (W_eff = gain * W)
    gain_da: float     = 0.5          # extra expression gain per unit DA
    wgd_alpha: float   = 6.0          # w_GD = sigmoid(alpha * DA + bias)
    wgd_bias: float    = -2.0         # init so w_GD → low when DA → 0

    # ---- training (train.py) ----
    episodes: int   = 32000           # total episodes; with B=128 → 250 gradient steps
    gamma: float    = 0.95
    gae_lambda: float = 0.95          # GAE(λ) for goal-directed advantage. 1.0 → plain
                                      # Monte-Carlo returns (original behaviour); <1 lowers
                                      # advantage variance at the cost of a little bias.
    ret_norm: bool  = True            # standardise GD returns by a running mean/std window
                                      # before the policy/value loss (stabilises the critic).
    ret_norm_window: int = 10000      # size of the rolling return-stat window
    lr_gd: float    = 3e-4
    lr_hab: float   = 3e-4
    entropy_beta: float = 0.05        # higher than default to prevent early policy collapse
    value_coef: float   = 0.5
    da_cost_lambda: float = 0.02      # penalty on da_request² → minimise its own request
    da_warmup: int  = 8000            # episodes with NO DA penalty (scaled ×4 for B=128)
    da_ramp: int    = 8000            # episodes to linearly ramp penalty to full
    grad_clip: float = 1.0
    ape_weight: float  = 1.0          # habitual APE (action prediction error) weight
    eff_weight: float  = 0.10         # habitual intrinsic-efficiency weight (NOT task reward)

    # APE-decay: the analogue of the supervisor's CE-teacher fade. Your habitual
    # net has no explicit teacher — it learns from action-prediction-error against
    # the COMBINED policy. APE is therefore the closest thing to a teaching signal.
    # When enabled, the APE weight decays once habitual-solo accuracy exceeds the
    # combined accuracy by ape_decay_margin, so a habit that has surpassed the
    # mixture is no longer dragged back toward it. OFF by default → original behaviour.
    ape_decay: bool    = False
    ape_decay_margin: float = 0.05    # hab_solo must exceed combined by this before decay
    ape_decay_rate: float   = 2.0     # how fast the weight falls per unit of surplus
    ape_min_scale: float    = 0.10    # floor: never fully remove APE

    # ---- checkpoint / resume (train.py) ----
    ckpt_every: int = 0               # 0 = off. >0 → write resumable state every N episodes.
    ckpt_path: str  = ""              # file path for the resumable checkpoint (set by runner)

    # ---- evaluation / analysis ----
    eval_every: int     = 100
    eval_trials: int    = 200
    final_trials: int   = 1000        # H1 binomial test
    attractor_trials: int = 300       # H6 PCA + decoder

    # ---- trajectory visualization ----
    traj_log_every: int   = 50
    traj_eval_trials: int = 60

    # checkpoint selection (by eval'd accuracy)
    learn_combined_min: float = 0.60  # learning phase: combined competent...
    learn_habsolo_max: float  = 0.60  # ...but habitual not yet
    maint_solo_min: float     = 0.80  # maintenance: habitual-solo carrying

    # ---- OPEN QUESTION #1 -------------------------------------------------
    # How is the DA-request signal trained?
    #   "a2c_coupled" (default): shaped by A2C advantage (reward-coupled).
    #   "local_pe": train from a LOCAL prediction error — not implemented;
    #       specifying that error is the open design decision.
    da_request_training: str = "a2c_coupled"

    batch_size: int = 128  # parallel environments per training iteration (vectorised BLAS)
                           # 128 amortises Python/env overhead over a large [B x n] matmul:
                           # ~1.7x ep/s vs 32 on an 8-physical-core CPU (BLAS-3 bound).

    seed: int    = 0
    device: str  = "cpu"

    def to_dict(self):
        return asdict(self)
