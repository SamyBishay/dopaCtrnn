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
    delay_start: int   = 15           # supervisor START_DELAY=15; must already stress WM at curriculum start
    delay_max: int     = 40           # supervisor MAX_DELAY=40; our 15 was far too low to test WM
    delay_advance_acc: float = 0.75   # accuracy threshold (combined) to lengthen delay
    delay_advance_evals: int = 3      # consecutive evals above threshold → advance
    max_episode_steps: int = 200      # raised from 80: delay_max=40 needs more steps per trial
    test_reward: float = 1.0          # reward for correct (non-match) choice
    step_cost: float   = 0.02         # supervisor STEP_RWD=-0.02; matches
    wait_cost: float   = 0.02         # per WAIT action — same as step_cost so WAIT isn't cheaper
    junction_bonus: float = 0.30      # task_r shaping: pre_sample→sample (reaching junction)
    arm_end_bonus: float  = 0.20      # task_r shaping: sample→delay (reaching arm end)
    completion_bonus: float = 0.10    # intrinsic, reaching ANY arm end (value-free)
    n_actions: int     = 5            # 0=N 1=S 2=E 3=W 4=WAIT
    obs_dim: int       = 6            # GD observation: [x, y, 0, sig_L, sig_R, sig_choice]
    obs_dim_hab: int   = 4            # Hab observation: [0, sig_L, sig_R, sig_choice] (no position)

    # ---- network sizes (model.py) ----
    n_gd: int    = 512                # supervisor PFC_HIDDEN=512; 256 was likely too small
    n_hab: int   = 512                # supervisor DLS_HIDDEN=512; match for fairness
    hab_rank: int = 0                 # 0 = full-rank habitual W (default, original behaviour);
                                      # >0 = low-rank W_rec = (m @ n.T)/n_hab, rank=hab_rank.
                                      # Makes "habit is low-dimensional" structural. Try 2-8.
    dt: float    = 1.0
    tau_fast: float  = 5.0            # dt/tau=0.2 → matches supervisor ALPHA=0.2 for fast units
    tau_slow: float  = 25.0           # slow units keep long WM timescale (dt/tau≈0.04)
    tonic_kappa: float = 0.10         # low-pass rate for tonic DA
    gain_base: float   = 0.5          # expression gain at DA=0 (W_eff = gain * W)
    gain_da: float     = 0.5          # extra expression gain per unit DA
    wgd_alpha: float   = 6.0          # w_GD = sigmoid(alpha * DA + bias)
    wgd_bias: float    = -2.0         # init so w_GD → low when DA → 0

    # ---- training (train.py) ----
    episodes: int   = 512000          # raised from 32k: longer curriculum needs more episodes
                                      # B=128 → 4000 gradient steps; ~same wall-time per step
    gamma: float    = 0.99            # supervisor GAMMA=0.99; 0.95 discounted too heavily over long delays
    gae_lambda: float = 0.95          # GAE(λ) for goal-directed advantage. 1.0 → plain
                                      # Monte-Carlo returns (original behaviour); <1 lowers
                                      # advantage variance at the cost of a little bias.
    ret_norm: bool  = True            # standardise GD returns by a running mean/std window
                                      # before the policy/value loss (stabilises the critic).
    ret_norm_window: int = 10000      # size of the rolling return-stat window
    lr_gd: float    = 1e-4            # supervisor LR_PFC=1e-4; our 3e-4 was too large for stable GD
    lr_hab: float   = 1e-3            # supervisor LR_DLS=1e-3; habitual benefits from faster imitation
    entropy_beta: float = 0.05        # higher than default to prevent early policy collapse
    value_coef: float   = 0.5         # supervisor VALUE_COEFF=0.5; matches
    da_cost_lambda: float = 0.02      # penalty on da_request² → minimise its own request
    da_warmup: int  = 64000           # episodes with NO DA penalty; scaled to match new episode count
    da_ramp: int    = 64000           # episodes to linearly ramp penalty to full
    grad_clip: float = 0.5            # supervisor GRAD_CLIP=0.5; tighter clipping stabilises RNN
    ape_weight: float  = 1.0          # habitual APE (action prediction error) weight
    eff_weight: float  = 0.10         # habitual intrinsic-efficiency weight (NOT task reward)

    # APE-decay: the analogue of the supervisor's CE-teacher fade. Your habitual
    # net has no explicit teacher — it learns from action-prediction-error against
    # the COMBINED policy. APE is therefore the closest thing to a teaching signal.
    # When enabled, the APE weight decays once habitual-solo accuracy exceeds the
    # combined accuracy by ape_decay_margin, so a habit that has surpassed the
    # mixture is no longer dragged back toward it. ON: mirrors supervisor CE_DECAY logic.
    ape_decay: bool    = True         # enabled; supervisor CE_DECAY is on by default
    ape_decay_margin: float = 0.05    # supervisor CE_DECAY_MARGIN=0.05; matches
    ape_decay_rate: float   = 2.0     # supervisor CE_DECAY_RATE=2.0; matches
    ape_min_scale: float    = 0.10    # supervisor CE_MIN_SCALE=0.1; matches

    # ---- checkpoint / resume (train.py) ----
    ckpt_every: int = 0               # 0 = off. >0 → write resumable state every N episodes.
    ckpt_path: str  = ""              # file path for the resumable checkpoint (set by runner)

    # ---- evaluation / analysis ----
    fixed_eval_delay: int = 40  # pinned delay for Villet-comparison eval; resolves DECIDE-DELAY-VALUE
                                # Villet: 90 s fixed delay; our steps: ceiling=40 is the hardest equiv.
                                # (maze len_edge=7, ~10-15 steps/trial; 90 s@1step/s >> our scale,
                                # so we pin to the curriculum ceiling — the hardest condition we train.)
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
