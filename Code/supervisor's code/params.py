"""
params.py  —  experiment configuration for tunl_a2c_two_area.py

Architecture
────────────
  PFC : FullRankRNN → A2C on full reward (goal-directed)
  DLS : LowRankRNN  → CE imitation of PFC + A2C on efficiency reward (habitual)

Training phases
───────────────
  1. Warmup     : PFC trains alone until acc stable at max_delay
  2. Imitation  : DLS trains (CE + efficiency RL), PFC continues training
  3. Handover   : dls_live_acc >= DLS_HANDOVER_ACC → DLS acts, PFC frozen
  4. Fallback   : dls_live_acc < DLS_FALLBACK_THRESH → PFC resumes, unfreezes

Usage
─────
  python tunl_a2c_two_area.py                        # uses ./params.py
  python tunl_a2c_two_area.py --config my_params.py
  python tunl_a2c_two_area.py --dls_rank 4 --seed 1  # CLI overrides

Run modes
─────────
  "single"  One run with DLS_RANK + MAX_DELAY defined here.
  "pairs"   One run per entry in FIXED_PAIRS = [(rank, max_delay, difficulty), ...]
  "sweep"   One run per rank in RANK_SWEEP_LIST.
"""

# ── EXPERIMENT ────────────────────────────────────────────────────────
SAVE_PATH    = "tunl_rnns/"
SEED         = 0

DIFFICULTY   = 0       # 0=easy  1=medium  2=hard  (controls grid height)
LEN_EDGE     = 7#5       # inner grid edge (must be odd, >=5)

START_DELAY  = 15      # delay at curriculum start
MAX_DELAY    = 40      # delay ceiling
N_STEPS      = 1_600_000

LESION_PFC   = False
LESION_DLS   = False

# ── NETWORK ───────────────────────────────────────────────────────────
PFC_HIDDEN   = 512
DLS_HIDDEN   = 512
DLS_RANK     = 2       # rank of DLS W_rec = m @ n^T
                       #   1 → single line attractor
                       #   2 → theoretical minimum for TUNL
                       #   4 → comfortable margin
                       #   8 → near-full flexibility

ALPHA        = 0.2#0.1     # CTRNN dt/tau
NOISE_STD    = 0.05    # recurrent noise (zeroed during eval)

# ── DLS REWARD STREAM ─────────────────────────────────────────────────
# DLS never sees food reward. It learns:
#   - WHAT to do: CE imitation of frozen PFC greedy at choice steps
#   - HOW FAST:   A2C on efficiency reward (step cost + completion bonus)

## for length_edge = 5, difficulty 2 this works
# DLS_PENALTY_ERROR    = True#False   # also penalise wrong choices via RL (optional)
# DLS_PENALTY_ERROR    = False   # also penalise wrong choices via RL (optional)
DLS_PENALTY_ERROR    = True   # also penalise wrong choices via RL (optional, difficult length)
DLS_COMPLETION_BONUS = 0.5    # terminal reward for finishing (correct or not)
NAV_CE_WEIGHT        = 1.0    # nav CE weight (reduced post-PFC-freeze)
CHOICE_WEIGHT        = 5.0    # choice-step CE weight relative to nav CE
EFFICIENCY_WEIGHT    = 0.5    # DLS A2C efficiency loss weight relative to CE
# CE imitation decays once DLS surpasses frozen PFC (weaker teacher)
CE_DECAY_MARGIN      = 0.05   # how much DLS must exceed PFC before decay starts
CE_DECAY_RATE        = 2.0    # how fast ce_scale falls per unit of surplus
CE_MIN_SCALE         = 0.1    # floor: never fully remove CE signal

# ── PFC FREEZE ────────────────────────────────────────────────────────
# PFC stops training when it reaches this quality — becomes a fixed teacher.
# DLS then fine-tunes against the stable frozen PFC policy.
PFC_FREEZE_ACC   = 0.95#85   # pfc_solo accuracy threshold
PFC_FREEZE_REACH = 0.98   # pfc choice-reach rate threshold
PFC_DEMO_BATCH   = 20     # PFC rollouts per DLS batch update
# Handover  : dls_live_acc >= DLS_HANDOVER_ACC  → DLS acts, PFC frozen
# Fallback  : dls_live_acc <  DLS_FALLBACK_THRESH → PFC resumes, unfreezes
# dls_live_acc = EMA of DLS-vs-PFC choice agreement (computed every episode)
# Handover when dls_solo_ema >= DLS_HANDOVER_FRAC * pfc_solo_acc
# Relative threshold — scales with whatever PFC achieved.
DLS_HANDOVER_FRAC    = 0.95
DLS_FALLBACK_THRESH  = 0.75 # for diff-2, len_edge-7 works;
# DLS_FALLBACK_THRESH  = 0.60 # for diff-2, len_edge-5 works

# ── TRAINING ──────────────────────────────────────────────────────────
LR_PFC       = 1e-4
LR_DLS       = 1e-3

# Warmup exits when ALL THREE are true:
#   step_count >= PFC_WARMUP_STEPS
#   ema_fast   >= WARMUP_ACC_THRESH
#   current_delay == MAX_DELAY
PFC_WARMUP_STEPS  = 50_000
WARMUP_ACC_THRESH = 0.75

# ── RUN MODE ──────────────────────────────────────────────────────────
RUN_MODE = "single"

FIXED_PAIRS = [
    (1, 10, 0),
    (2, 20, 1),
    (2, 40, 1),
    (4, 40, 1),
]
RANK_SWEEP_LIST = [1, 2, 4, 8]
EVAL_DELAYS     = [0, 5, 10, 20, 40]

# ── LOGGING / CHECKPOINTING ───────────────────────────────────────────
EVAL_EVERY        = 5_000
CHECKPOINT_EVERY  = 50_000

# ── REWARD & RL ───────────────────────────────────────────────────────
RWD               = 1.0
INC_RWD           = -1.0
STEP_RWD          = -0.02
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
VALUE_COEFF       = 0.5
GRAD_CLIP         = 0.5
TARGET_ACC        = 0.70
MAX_EPISODE_STEPS = 1000 ### for length_edge = 5 difficulty 2, this works
# MAX_EPISODE_STEPS = 200 ### for those after-warmup ones, to encourage shorter trajectories
