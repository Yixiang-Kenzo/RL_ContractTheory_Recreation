# Experiment 2 — Coin Game (multi-agent) spec

Consolidated verbatim from the paper (Sections 5.1–5.3, Appendix D.2, Algorithm 3). This is
the build sheet for the multi-agent reproduction.

## Environment (Coin Game — Foerster et al. 2018)
- Grid: **7×7** main (Fig. 2); **3×3** smaller variant, Appendix D.2 (Fig. 4).
- **2 agents** (red, blue). Episode length **50** steps (7×7); **20** steps (3×3).
- Coins spawn randomly on the grid. Reward: **+1** own-color coin, **+0.2** other-color coin.
  (Paper uses the rllib Coin Game but **removes** the penalty a player incurs when the other
  picks up its coin.)
- State: **4-channel** grid (red pos, blue pos, red-coin pos, blue-coin pos). Actions: grid
  moves (up/down/left/right).
- It's a sequential social dilemma: selfish play under-collects joint welfare (agents grab
  each other's coins); cooperation (only take own color) maximizes social welfare.

## Principal
- Objective: **maximize social welfare with minimal payment.** Principal reward
  `rp(s,a) = (1/α) Σ_i r_i(s,a)`, **α = 0.1** (payment minimization is secondary).
- Gives an **action recommendation to each agent**; payments determined after agents act.
- Contractual Q-function trained with **VDN** (value decomposition across agents); agents'
  networks **share parameters**.

## Solution concept & the IC constraint (Eq. 5)
- Multi-agent SPE = principal subgame-perfect + agents in **Markov Perfect Equilibrium**.
- Extra **Incentive-Compatibility (dominant strategy)** requirement (Eq. 5): each agent
  prefers the recommended policy **regardless of the other agents' policies**. Robust to
  equilibrium-selection. Inspired by k-implementation (Monderer & Tennenholtz 2003).
- **Tractable approximation** (the key trick): conditioning on others' policies is
  intractable, so each agent's strategy space is reduced to a **binary variable f_i ∈ {0,1}**:
  f_i=1 → follow the principal's recommendation (and get paid); f_i=0 → "rational deviation"
  = default selfish policy ignoring contracts (same utility in equilibrium). Q^φ_i is
  conditioned on the joint f_{-i} of the others.

## Algorithm 3 (two-phase)
**Training phase.** Sample a binary vector f = (f_i) per episode. During the episode: for
i with f_i=1 the principal picks recommended action via `qθ_i` (ε-greedy); for f_i=0 the
agent acts selfishly via `Qφ_i((s,f_{-i}),·)` (ε-greedy). Store `(s,a,f,r,s')`. Updates:
- contract `b*_i(s,a_i) = max_a Qφ_i((s,1),a) − Qφ_i((s,1),a_i)`  (min implementation)
- principal target `yp_i = r_i − α·b*_i(s,a_i) + γ max_{a'_i} qθ'_i(s',a'_i)` — **update θ as VDN**
- agent target   `ya_i = r_i + γ max_{a'_i} Qφ'((s',f_{-i}),a'_i)` — **update φ as DQN**

**Validation phase.** Train fresh **black-box selfish DQNs ψ_i from scratch** in the modified
environment: each agent sees the recommendation when computing Q, `Qψ_i((s,a^p_i),a_i)`, and
is paid iff it follows the recommendation. Oracle best-response is infeasible here (too large),
hence from-scratch DQNs. Principal has no access to agents' Q-values/params.

## Hyperparameters (Appendix D.2)
- Nets: **1 conv layer** (4×4 kernels, 4→32 channels) + **2 FC layers (64)** + output; ReLU.
- **1,000,000 iterations**; 1 env interaction + 1 gradient update per iter; **batch 128**.
- Target nets hard-copied every **100** iters. LR **5e-4 → 1e-4** (exp). ε **0.4 → 0** (linear).
  **γ = 0.99**.
- **Prioritized replay** (Schaul et al. 2015): max size **100000**, α=0.4, β=0, ε=1e-7.

## Baselines
- **Selfish** = constant-proportion with proportion 0 (no contracts).
- **Optimal/cooperative** = proportion 1 (agents directly maximize social welfare).
- **Constant-proportion heuristic** (Christoffersen et al. 2023): distributes a fixed % of
  social welfare, budget-matched to what our method pays after validation.

## Figures / metrics
- **Fig. 2 (7×7)**: (a) social welfare, (b) proportion of welfare paid (~30%+), (c) accuracy
  = recommendations followed (~80–90%), (d) SPE ratio (Blue's best vs recommended when Red
  follows recommendation; ~1 = SPE), (e) IC ratio (same but Red plays its own policy; ~1 = IC).
- **Fig. 4 (3×3)**: adds a nudging panel; the paper notes **nudging is necessary** for good
  performance on the small grid (validation agents otherwise beat IC by 5–10%).
- Shaded = standard error (top plots) / min–max (bottom plots). 5 repeats, 80-episode averages.

## Build progress
- Environment `coin_game.py` ✅ (random welfare 2.52/ep on 3×3).
- Networks `networks.py` ✅ (ConvQNet 4→32 conv + 2×64 FC; agent-centric shared params;
  uniform replay for now — PER is a TODO, logged).
- Training phase `train_coin.py` ✅ Algorithm 3 lines 1-21: VDN principal, shared agent net
  conditioned on f_{-i}, contract = max-gap min-implementation, α=0.1.
  **Smoke test (3k iters, 39s): welfare 0.79→11.5, payment proportion 28.8% — matches paper's
  ~30%.** Benchmark: **~77 iters/sec** on CPU → 200k iters ≈ 45 min. 3×3 is CPU-feasible.
- Validation phase ✅ (fresh DQNs ψ, see recommendation, paid iff follow). Baselines ✅
  (`baselines.py`: selfish p=0, cooperative p=1, constant-prop p=matched). Fig-2 orchestration
  + plot ✅ (`run_fig2.py`). Nudging ✅ (abs margin = nudge_frac × per-step welfare).
- **STABILITY FIX (important):** first 40k run DIVERGED past ~4k iters (welfare collapsed
  10→2, payments exploded to 400% of welfare). The 3k smoke test missed it (stopped before
  divergence). Fix: **Huber (smooth-L1) loss + grad-clip(10) + target-net φ' for the contract**
  (was online φ). After fix: welfare stable ~12.5, payment ~30% through 10k+ iters. Applied to
  θ, φ, ψ, and baseline nets. LESSON: always run past the smoke horizon before trusting.
- TODO: Fig-2 SPE ratio (d) & IC ratio (e) panels (deferred); full run with baselines+nudging.

## Compute reality (this machine: 14-core CPU, no CUDA GPU)
1M iters × conv nets × 3 networks + a second training for validation, ×5 repeats. Estimated
**hours to >a day per full 7×7 run** on CPU. Plan: build+validate the pipeline on the **3×3**
grid at a reduced iteration budget first, then scale. Cloud GPU is an option for final numbers.
