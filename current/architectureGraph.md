---
title: Architecture & Experiment Graphs
created: 2026-06-20
type: working-note
---

# Architecture & Experiment Graphs

## 1. Neural Network Architecture

### 1.1 Top-level model (DualSystemModel)

```mermaid
flowchart TD
    OBS["Observation\n[B, 6]\n(x, y, 0, sig_L, sig_R, sig_choice)"]

    OBS -->|"obs"| GD["GDNet\n(mPFC / DMS)\nn_gd = 256 units"]
    OBS -->|"obs2 (same)"| HAB["HabNet\n(DLS)\nn_hab = 256 units"]

    GD -->|"π_GD [B, 5]"| MIX
    GD -->|"da_request [B]"| WGDCOMP["w_GD = σ(α · da_request + bias)"]
    GD -->|"value [B]"| VALUE["Value head\n(A2C critic)"]

    HAB -->|"π_H [B, 5]"| MIX

    WGDCOMP -->|"w_GD [B]"| MIX["Policy mixer\ncombined = w_GD · π_GD + (1−w_GD) · π_H\n\nDevaluation: combined = w_GD · mot · π_GD + (1−w_GD) · π_H"]

    MIX -->|"logits [B, 5]"| ACTION["Sample action\n(0=N 1=S 2=E 3=W 4=WAIT)"]
```

### 1.2 GDNet internals

```mermaid
flowchart TD
    X["x [B, obs_dim]"]
    H_PREV["h_prev [B, n]"]
    DATONIC["da_tonic [B]\n(slow low-pass of da_request)"]

    X & H_PREV --> CTRNN["CTRNN step\ndh = −h + tanh(h)·W.T + x·W_in.T + b\nh ← h + (dt/τ)·dh\n\n(τ per-neuron, learnable;\nfirst n/2: τ_fast≈2; second n/2: τ_slow≈25)"]

    CTRNN -->|"h [B, n]"| DAREQ["da_request = σ(h · w_da + b_da)  [B]"]
    CTRNN -->|"h [B, n]"| GAIN

    DATONIC -->|"slow channel"| DACOMP
    DAREQ -->|"fast channel"| DACOMP["da_comp [B, n]\nFirst n/2 (D1-like) ← da_request  (phasic)\nSecond n/2 (D2-like) ← da_tonic   (tonic)"]

    DACOMP --> GAIN["gain = gain_base + gain_da · da_comp\nr_out = gain · tanh(h)   [B, n]"]

    GAIN -->|"r_out"| PI["π_GD = r_out · W_out.T + b_out  [B, 5]"]
    CTRNN -->|"h"| VAL["value = h · w_v + b_v  [B]"]

    DAREQ --> DATONICUPD["da_tonic ← (1−κ)·da_tonic + κ·da_request\n(κ = tonic_kappa = 0.10)"]
```

### 1.3 HabNet internals

```mermaid
flowchart TD
    X2["x [B, obs_dim]"]
    H2["h_prev [B, n]"]

    X2 & H2 --> CTRNN2["CTRNN step (same dynamics as GDNet)\ndh = −h + tanh(h)·W.T + x·W_in.T + b\nh ← h + (dt/τ)·dh"]

    CTRNN2 -->|"h [B, n]"| GO["Go path\ntanh(h) · W_go.T  [B, 5]"]
    CTRNN2 -->|"h [B, n]"| NOGO["NoGo path\ntanh(h) · W_nogo.T  [B, 5]"]

    GO & NOGO --> OPPO["π_H = Go − NoGo  [B, 5]\n(opponent actor, OpAL* style)"]
```

### 1.4 Key parameters summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| Observation | obs_dim | 6 |
| GDNet | n_gd | 256 |
| HabNet | n_hab | 256 |
| Time constants | τ_fast / τ_slow | 2.0 / 25.0 (first/second half of units) |
| DA gain | gain_base / gain_da | 0.5 / 0.5 |
| DA low-pass | tonic_kappa (κ) | 0.10 |
| Mixing sigmoid | wgd_alpha / wgd_bias | 6.0 / −2.0 (init: w_GD → low) |
| Batch size | B | 128 parallel envs |
| Training episodes | total | 32 000 |

---

## 2. Experiment Design

### 2.1 T-maze task (single trial)

```mermaid
flowchart LR
    START["START (2,2)"]
    STEM["STEM (1,2)"]
    JCT["JUNCTION (0,2)\nphase signal set:\nsig_L or sig_R = 0.25"]
    LEND["L_END (0,0)"]
    REND["R_END (0,4)"]

    START -->|"pre_sample:\nagent walks N×2"| STEM
    STEM -->|"reach junction\n→ phase: sample\n→ task_r += junction_bonus"| JCT

    JCT -->|"open side\n(blocked side impassable)"| LEND
    JCT --> REND

    LEND -->|"reach arm end\n→ phase: delay\n→ task_r += arm_end_bonus"| DELAY["DELAY\nAgent moves freely\nPhase signal decays ×0.1/step\ndelay_idx counts steps\n(delay_start=5 → delay_max=15)"]
    REND -->|"reach arm end\n→ phase: delay"| DELAY

    DELAY -->|"delay_idx ≥ current_delay\n→ phase: choice\nsig_choice = 0.25"| CHOICE["CHOICE\nBoth arms open"]
    CHOICE -->|"non-match arm\n(= blocked arm)"| CORRECT["CORRECT\n+test_reward (1.0)"]
    CHOICE -->|"match arm"| WRONG["WRONG\nno reward"]
```

### 2.2 Training loop

```mermaid
flowchart TD
    INIT["Init: DualSystemModel + 2 optimizers\n(opt_gd for GD params + α,bias;\nopt_hab for Hab params)"]

    INIT --> LOOP["Loop: 32000 episodes total\n(250 gradient steps × B=128)"]

    LOOP --> BATCH["_train_batch: B envs in lockstep\nstep all envs → collect [T,B] tensors"]

    BATCH --> GDLOSS["GD loss (A2C):\nexpr_gate × (policy_loss + entropy_loss)\n+ value_loss + da_penalty\n\nda_penalty ramps: 0 until ep 8000,\nlinear to da_cost_lambda=0.02 by ep 16000"]

    BATCH --> HABLOSS["Hab loss (value-free):\nape_weight × CrossEntropy(π_H, action)\n+ eff_weight × efficiency_A2C(intrinsic_r)"]

    GDLOSS --> OPTGD["opt_gd.step() (grad_clip=1.0)"]
    HABLOSS --> OPTPHAB["opt_hab.step() (grad_clip=1.0)"]

    OPTGD & OPTPHAB --> EVAL["Every 100 episodes:\nevaluate_vec × 3:\n  combined, force_w=0 (hab-solo), force_w=1 (gd-solo)"]

    EVAL --> CKPT["Checkpoint selection:\nckpt_learn: comb≥0.60 AND hab_solo≤0.60\nckpt_maint: hab_solo≥0.80 AND comb≥0.80"]

    EVAL --> CURRICULUM["Delay curriculum:\nif comb≥0.75 AND hab_solo≥0.75\nfor 3 consecutive evals → delay += 1\n(delay_start=5 → delay_max=15)"]
```

### 2.3 Evaluation pipeline (per seed)

```mermaid
flowchart TD
    TRAIN["train() → model, logs, ckpt_learn, ckpt_maint"]

    TRAIN --> H1["H1 — Learning\nbinomial test on 1000 trials\n(ckpt_maint, combined policy)"]
    TRAIN --> H2["H2 — Emergent handoff\nfrom training logs:\nhab_solo_acc rise vs w_GD drop timing"]
    TRAIN --> H3["H3 — Devaluation dissociation\nmot=1 vs mot=0\n@ ckpt_learn AND ckpt_maint"]
    TRAIN --> H4["H4 — Lesion × phase\nlesion∈{gd, hab, both}\n@ ckpt_learn AND ckpt_maint"]
    TRAIN --> H5["H5 — Reactivation\n@ ckpt_maint:\nda_request intact vs hab-silenced\nacc + devaluation-sensitivity"]
    TRAIN --> H6["H6 — WM attractor\n@ ckpt_maint:\nPCA of delay hidden states (h_hab)\n+ LogisticRegression decoder (CV=5)"]
    TRAIN --> H7["H7 — Untrained control\n(separate run with --untrained flag)"]
```

### 2.4 DA dynamics during training (schematic)

```
DA-request
(= w_GD)
   │
   │   ╭──────────────╮
   │   │  GD carries  │
   │   │ the policy;  │    ╭──────────────────
   │   │ DA stays high│   /  Hab competent;
   │   │              │  /   DA penalised;
   │   │              │ /    w_GD drops
   │   ╰──────────────╯/
   │                  /
   │─────────────────/─────────────────────────► episode
   0              ~ep1500                    ~ep2500
                  (hab-solo                  (w_GD drop)
                  crosses 0.8)
```
