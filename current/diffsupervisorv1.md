# Differences from Supervisor's Implementation — v1

A systematic comparison between `Code/main/` (my implementation) and `Code/supervisor's code/tunl_a2c_two_area.py`. The goal is to identify every design choice that makes my version trivially easy.

---

## 1. The sample phase is scripted — the biggest problem

**Mine:** The environment hard-codes a `_sample_path = [0, 0, 2, 2]` (or `[0, 0, 3, 3]`) and executes it automatically regardless of what the agent sends. The agent is a passenger during the most informative phase of the trial. It never needs to navigate, never needs to find the arm, never needs to touch anything. It is physically carried there.

**Supervisor's:** The agent starts at a random location anywhere in the open grid. It must independently navigate to the initiation zone (a specific cell), which triggers the sample display. Then it must navigate to the lit sample arm and touch it. Then it must navigate back to the initiation zone to begin the delay. Then it navigates to the correct choice arm. Every step of every phase is free, unconstrained, and the agent can fail or time out at any of them.

The consequence: in my code, the arm identity is never "discovered" by the agent — it is delivered to the agent on a plate, encoded directly in the observation (see §2). Working memory of a passively-received cue is categorically easier than memory of an actively-sought target.

---

## 2. The observation directly encodes the answer

**Mine (`environment.py`, `obs()`):**

```python
# Allocentric stream (22D):
#   bits 15-17: phase one-hot (sample / delay / test)
#   bits 18-19: BLOCKED arm one-hot — which side is walled off
#   bits 20-21: OPEN arm cue     — which side is open

if self.phase == "sample":
    allo[18 + (0 if self.blocked == "L" else 1)] = 1.0   # blocked
    allo[20 + (0 if self.open_side == "L" else 1)] = 1.0  # open
elif self.phase == "test":
    allo[20] = 1.0; allo[21] = 1.0   # both lit in test
```

During the sample phase, bits 18/19 tell the agent *exactly* which arm it must avoid, and bits 20/21 tell it *exactly* which arm is open. The answer is in the observation. All the network needs to do is copy that bit across the delay.

During the test phase, bits 20-21 are both 1 — so the observation gives no help at that moment, but the agent just needs to remember the one-hot that was active 10 steps ago, from a full-rank 22D observation that never contained noise or decay.

**Supervisor's (`get_vectorized_observation()`):**

```python
obs[0] = x / (self.w - 1.0)    # normalized x position
obs[1] = y / (self.h - 1.0)    # normalized y position
obs[2] = 0.0                    # prev_action (zeroed)
obs[3:6] = self.phase_signal    # 3-float signal, decays during delay
```

The phase signal encodes which arm was sampled as a 3-float vector (e.g. `[1/(w-1), 0, 0]` for L, `[0, 1/(w-1), 0]` for R). During the delay, every step it is multiplied by 0.1:

```python
if self.delay_t < self.len_delay:
    self.delay_t += 1
    self.phase_signal *= 0.10    # exponential decay toward zero
```

After just 2 delay steps the signal is at 1% of its original value. At max_delay=40 it is numerically zero. The agent must have already encoded the arm identity into its recurrent hidden state before the signal vanishes. The network cannot coast on a persistent external cue.

---

## 3. Delay length and working memory demand

| | Mine | Supervisor's |
|---|---|---|
| Delay length | 10 steps, fixed | curriculum 15 → 40 steps |
| Delay signal | constant (no decay) | decays ×0.1 per step |
| Max episode steps | ~30 total | 1000 |
| Agent during delay | sits at START, observation is constant | free to move; still counts against step cost |

At delay=40 with exponential decay, the only information available at the choice moment is what the network wrote into `h_pfc` / `h_dls` during the sample navigation. This is genuine working memory. Mine is not — it is just a 10-step buffer over a static input.

The curriculum (`current_delay` promoted when `ema_fast >= TARGET_ACC` and stable) ensures the network never faces a delay it cannot yet solve, which prevents the "collapse and restart from random" failure mode.

---

## 4. Network architecture

### Mine

- GD CTRNN: 64 units, full-rank W ∈ ℝ^{64×64}
- Habitual CTRNN: 64 units, full-rank W ∈ ℝ^{64×64}
- No recurrent noise during training

### Supervisor's

- PFC: `FullRankRNN`, 512 units, full-rank W
- DLS: `LowRankRNN`, 512 units, **rank-constrained** W_rec = m @ n^T (default rank=2)

The low-rank constraint on DLS is the primary analytical target of the study. It caps the DLS's representational capacity to exactly rank dimensions of attractor structure. Rank=2 is the theoretical minimum for a TUNL task (one dimension for arm identity, one for the delay progression). This constraint makes the DLS's geometry interpretable via SVD reparametrization:

```python
def svd_reparametrize(self):
    W = (self.m @ self.n.t()).cpu().numpy()
    u, s, vt = np.linalg.svd(W, full_matrices=False)
    # rewrite m, n as orthogonal modes weighted by sqrt(singular values)
```

The singular values of the learned W_rec are a direct readout of how many representational dimensions the DLS actually used, regardless of what rank was allocated. My code has no such constraint and no such analysis hook — my "participation ratio" is computed post-hoc and is not built into the architecture.

Recurrent noise (`noise_std=0.05`, zeroed at eval) is also present in both networks. This is standard practice for training robust CTRNNs — it prevents the network from relying on overly sharp attractors that would collapse under real-world perturbations. My code has none.

---

## 5. Training regime

### Mine: one-phase simultaneous training

Both systems train from episode 1. A DA penalty is applied to GD after episode 1500. There is no explicit staging — the habitual network is trained on action prediction error throughout.

### Supervisor's: three explicit phases

**Phase 1 — PFC warmup.** Only PFC trains via standard A2C on the full reward (+1 correct, -1 incorrect, -0.02/step). DLS receives imitation signals in parallel but does not yet act. Warmup exits only when *all three* of:
- `step_count >= PFC_WARMUP_STEPS` (50K)
- `ema_fast >= WARMUP_ACC_THRESH` (0.75)
- `current_delay == max_delay` (40)

This guarantees DLS is never imitating a bad teacher. My habitual network imitates the GD policy from episode 1, when GD is performing at chance.

**Phase 2 — Imitation + efficiency.** DLS learns KL(PFC‖DLS) confidence-weighted over all timesteps, with a 5× upweight at the terminal (choice) step. Simultaneously it runs efficiency RL (GAE on step cost + completion bonus, no food reward). PFC continues to update. A demo replay buffer stores the 500 most efficient correct PFC episodes for DLS to re-use:

```python
demo_buffer = deque(maxlen=500)
demo_use_prob = 0.3   # 30% of DLS updates use replayed demos
```

My habitual training has no replay, no confidence weighting, and weights the choice step equally with navigation steps (CE on every agent-controlled step).

**Phase 3 — Handover.** DLS takes over the environment when `dls_solo_ema >= 0.95 × pfc_solo_acc`. PFC is **frozen** — it becomes a fixed teacher. DLS now drives real trajectories and receives only terminal-step KL (navigation is free exploration) plus full efficiency RL. If DLS degrades below `0.75 × pfc_solo_acc`, the fallback triggers: PFC unfreezes and resumes acting.

My handover is soft and continuous (a sigmoid gate driven by DA penalty). The supervisor's is a hard binary switch based on measured competence. There is no DA signal in the supervisor's code at all.

---

## 6. Handover mechanism

|                                   | Mine                                       | Supervisor's                                     |
| --------------------------------- | ------------------------------------------ | ------------------------------------------------ |
| Mechanism                         | soft sigmoid gate: w_GD = σ(α·DA + bias)   | hard binary switch                               |
| Trigger                           | DA penalty ramp (schedule)                 | measured DLS solo accuracy vs. PFC solo accuracy |
| Fallback                          | none — gate can drift back but isn't reset | explicit: DLS unfreezes PFC if it degrades       |
| Emergent or designed              | emergent from pressure                     | explicit state machine                           |
| Number of eval rollouts to decide | none (continuous)                          | 100 solo episodes every 5K steps                 |

In my code the handover is emergent in the sense that no external code decides it — w_GD falls because the DA penalty makes it costly. But the schedule of that penalty is an external clock (episodes 1500–3000), not a function of DLS competence. In the supervisor's code the handover is competence-triggered: DLS only takes over when it has actually proven it can match PFC.

---

## 7. Efficiency pressure and episode termination

**Mine:** step_cost=0.02 per test-phase action. The test phase is at most ~10 steps in a tiny maze. The GD network learns a clean path very quickly. There is no post-handover episode-length pressure.

**Supervisor's:** After handover, the max episode steps is reduced from 1000 to 200 (`envs.max_episode_steps = int(MAX_EPISODE_STEPS/5)`). A completion bonus of 0.5 incentivises reaching the arm at all (random exploration in a large maze rarely finds the arm). The learning rate for DLS is also halved post-handover for stability. The paper explicitly tracks `ep_lengths` as a key readout — DLS efficiency (route compression) is expected and measured.

---

## 8. Start position

**Mine:** every episode starts at (2,2) = the fixed START cell. The agent always knows exactly where it is from observation bit 12 of the one-hot position map.

**Supervisor's:** `idx = self.rng.randint(0, len(open_y), size=N_ENVS)` — random open cell. After warmup, `force_init_loc=True` is set so it starts at the initiation cell, but during early training it is entirely random. The agent must orient itself from only its (x, y) coordinates, not from a one-hot map.

---

## 9. What "working memory" actually tests

**Mine (H6):** The delay-period hidden state of the habitual CTRNN separates by arm with a linear decoder (acc=1.00). But what is actually being tested is whether the network has carried forward the state that was established during the *scripted sample path* (where the arm was different, the walls were different, the observations were different). The network doesn't maintain memory because it has to — it maintains it because the scripted path left a persistent trace that 10 delay steps of identical input doesn't fully erase.

**Supervisor's:** The phase signal is zero at the end of a 40-step delay. The arm identity can only be in the hidden state. The low-rank DLS is forced to represent it in at most `rank` attractor dimensions. The singular values of W_rec quantify *how compressible* that memory is. This is the quantity the paper is after — not just "can the network remember," but "what is the minimum representational dimension of the memory."

---

## Summary table

| Dimension | Mine | Supervisor's |
|---|---|---|
| Sample navigation | scripted (agent not in control) | free (agent must find arm) |
| Start position | fixed | random during warmup |
| Observation dimension | 22D (allocentric) + 14D (egocentric) | 6D (position + decaying phase signal) |
| Arm cue in observation | explicit one-hot, persists through delay | decays ×0.1/step, ~0 after 2 steps |
| Delay | 10 steps, fixed | 15→40 steps, curriculum |
| External cue at choice | none (correct) | none (correct) |
| Network sizes | 64+64 units | 512+512 units |
| DLS recurrent structure | full rank | rank-constrained (rank=2 default) |
| Recurrent noise | none | 0.05 std during training |
| Training regime | simultaneous, 5K episodes | staged (warmup, imitation, handover), 1.6M steps |
| Handover | soft sigmoid (DA-scheduled) | hard binary (competence-triggered) |
| Fallback | none | yes (DLS degrades → PFC unfreezes) |
| Imitation signal | CE on every test-phase step | KL(PFC‖DLS) confidence-weighted, ×5 at choice |
| Replay buffer | none | 500-episode demo buffer (30% of DLS updates) |
| Efficiency pressure | mild step cost | strong (episode length tracked, halved after handover) |
| Task reward in DLS loss | no (correct) | no (correct) |
