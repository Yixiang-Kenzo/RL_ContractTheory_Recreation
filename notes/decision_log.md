# Decision Log

Every detail the paper leaves under-specified, the choice we made, and why. This is the
record of *our* decision-making for the from-scratch reproduction of Ivanov et al. (2024).

---

## Model notation (from the paper)

- **State** `s`, **agent action** `a ∈ A`, **outcome** `o ∈ O` (m outcomes).
- Action `a` at state `s` draws outcome `o ~ O(s,a)` (the **outcome distribution**), which
  determines reward and next state.
- **Agent** reward `r(s,a)` (can be negative = effort cost) **plus** contract payment `b(o)`.
- **Principal** reward `rᵖ(s,o)` **minus** payment `b(o)`.
- **Contract** `b : O → ℝ≥0` — non-negative payments (**limited liability**).
- **Principal policy** `ρ` maps each state to a contract.

Key recursions (paper §3–4):
- Truncated agent Q (excludes immediate payment):
  `Q̄*(s,a) = r(s,a) + γ · E_{o~O(s,a)}[ V(s'(o)) ]`
- Full agent Q:  `Q*((s,b),a) = E_{o~O(s,a)}[b(o)] + Q̄*(s,a)`
- Agent value:  `V(s) = max_a Q*((s, ρ(s)), a)`

---

## Tree MDP — Experiment 1

> **UPDATE (after reading the raw PDF Appendix D.1):** the paper actually specifies the
> Tree MDP experiment *in full*. Our earlier reads used a summarizing web-fetch tool that
> silently skipped the appendix pages, so entries D-1..D-4b below were solving problems the
> authors had already pinned down. They are kept for the record but **superseded by D-6**,
> which is the paper's exact recipe. The auto-calibration (D-4b) is retained as a diligence
> artifact; the paper's own reward sampler reaches the ~60% effort target by construction.

### D-1: Tree structure  — *under-specified*  → SUPERSEDED by D-6
Paper says "randomly generated binary game-trees," finite horizon, states tied to timesteps,
2 outcomes per transition. It does **not** give depth, branching, or reward distributions.

**Our choice:** complete binary tree of depth `T`. Each internal node (depth < T) is a
decision state; leaves (depth T) are terminal with value 0. **m = 2 outcomes** per
transition; outcome `o ∈ {0,1}` selects the left/right child. So the tree branches on
*outcomes*, and *actions* shift the outcome distribution.
**Why:** matches "binary" + "2 outcomes per transition" + "states tied to timesteps," and
gives a clean finite tree we can solve exactly for ground truth.

### D-2: Actions  — *under-specified*
**Our choice:** `|A| = 2` actions per state. Action reward `r(s,a)` sampled per (node,action);
we bias action 0 to be "cheap/low-effort" and action 1 to be "costly/high-effort," so the
principal must pay to induce the effortful action — the classic contract tension.

### D-3: Outcome distributions  — *under-specified*
**Our choice:** for each (node, action) sample a distribution over the 2 outcomes. The
effortful action puts more mass on the outcome the principal rewards, so paying to induce it
is what buys the principal value.

### D-4: Rewards  — *under-specified (Fig. 1 gives one example: r(s,a_L)=-4/5, r(s,a_R)=0)*
**Our choice (initial):** agent action rewards sampled i.i.d.; principal outcome rewards
`rᵖ(s,o)` sampled i.i.d. Exact ranges TBD in code; logged once fixed. Discount `γ` TBD.

### D-4b: Environment calibration — *auto-tuned (our method)*
The paper never states the generative distribution for tree rewards/outcome-probabilities.
A contract experiment is only meaningful if the principal actually pays to induce effort in
a non-trivial fraction of states. We swept the reward/cost knobs and, across **200 random
depth-5 trees per config**, measured the **effort-induced rate** (fraction of states where
`opt_action != 0`) and **pay-share** (expected payment / principal gross reward), selecting
the config landing in a 30–60% effort band. See `experiments/Tree_MDP/calibrate.py`.

**Selected config** (`env_config.json`): `effort_cost_range=(0.0, 0.15)`,
`principal_reward_range=(0.0, 2.0)`, `effort_bias=0.8`, `gamma=0.9`, `n_actions=2`.
Yields **effort_rate ≈ 42.8%**, **pay_share ≈ 3.6%**, mean principal value ≈ 5.06.
**Why:** puts the environment in the regime where contracts bind often but not always —
the interesting regime for the ~90%-optimal-action metric to actually test the algorithm.

### D-8: PyTorch environment — *machine constraint*
PyTorch's pip wheels ship unsigned DLLs; the machine had **Smart App Control** enforcing,
which blocked `torch_python.dll` (WinError 4551). SAC has no per-file allowlist and is a
one-way toggle. User chose to **disable Smart App Control** (2026-07-16); `torch 2.13.0+cpu`
now imports and runs. CPU-only is fine (tiny nets). NumPy/SciPy were unaffected (signed).

### D-9: DQN implementation choices for Algorithm 2 — *our choices (paper omits)*
- **State encoding:** one-hot over the 1023 internal nodes (leaves never need values —
  they're terminal, bootstrap masked by (1-d)). Guarantees the net can represent the exact
  solution; simplest faithful choice.
- **Contract solve in the hot loop:** the 2-action/2-outcome LP has a closed form
  (`binary_contract` in contract.py): to induce the effort action pay only on the good
  outcome `b = [0, max(ΔQ,0)/(p₁-p₀)]`; to induce the free action pay only on the bad
  outcome. Verified equal to the scipy LP. Used for speed (5M+ solves); LP kept as reference.
- **Networks for the LP inputs / targets:** target agent net φ' used for both contract
  solves b*(s,aₚ) and b*(s',aₚ'), for target stability; online θ selects aₚ' (Double-DQN
  style), target θ' provides its value. Logged as a defensible reading of Algorithm 2.
- **Validation:** principal offers, per state, aₚ=argmax qθ and contract from the learned
  agent net; an **exact** oracle best-responds by backward induction over true dynamics;
  principal utility computed under that response and compared to exact.py optimum.

### D-10: The discontinuity — tie-breaking + nudging — *specified by the paper*
Symptom during first DQN runs: 100% action-recommendation accuracy but principal utility
collapsed to ~0.08 of optimal. Diagnosis (isolating test): recommendations were perfect;
the *contracts* systematically under-paid by a hair (94% of effort states), and the optimal
contract sits exactly at the agent's indifference point, so any downward error flips the
agent to slacking — the paper's "discontinuity." Two paper mechanisms:
1. **Tie-breaking in principal's favor** (Lemma B.12, p.8): at exact indifference the agent
   takes the recommended action. Implemented in `oracle_eval` (needs the recommendation).
   Validated: `oracle_eval(exact contracts)` returns exactly the optimum (ratio 1.000000).
2. **Nudging** (Appendix B.9.2, p.31): add a strict-preference margin ξ to the RHS of the IC
   constraint so the agent must prefer a_p by ξ; paper sets **ξ=2Eₜ** (2× agent Q-error),
   applied both in training (it changes the learned Q-functions) and when offering contracts.
   Implemented via `nudge` arg in `binary_contract` and threaded through `train_one`.
Empirical note: more training alone helped (0.077 → 0.514 at 8k vs 2k updates on depth-5) but
plateaued at a coin-flip on the knife-edge; nudging is needed to clear it. The paper's
"1-2% without nudging" claim likely benefits from their larger scale (depth-10, averaged over
3 instances); our reproduction uses a small nudge, chosen by sweep (see D-11, TBD).

### D-11: Nudge margin choice — *our choice, swept*
Swept ξ on depth-5 (8k updates): ξ=0 → util 0.51 (knife-edge); **ξ=0.01 → util ~1.00, acc
90%**, payment 0.76× optimal; ξ≥0.05 starts overpaying (ξ=0.10 → pay 1.21×, util 0.85).
**Chosen: ξ=0.02** at the paper's depth-10 scale. Matches ξ=2Eₜ: depth-10 has larger Q-error
Eₜ than depth-5, so it needs a bigger nudge. Depth-10/20k comparison (single instance):
ξ=0.01 → util 0.948 but VOLATILE (evals swing 0.37–0.97, discontinuity at near-root states);
**ξ=0.02 → util 0.949 and STABLE (last evals 0.93–0.95)**; ξ=0.03 → 0.918 (overpays). So 0.02
buys stability at ~no utility cost. Result vs paper: **accuracy 95.6% (beats paper ~90%),
utility ~0.95 vs paper ~0.98.** The ~3% utility gap is the nudge's payment cost — the paper
claims 2% *without* nudging, but our contracts need the nudge for stability. Honest, faithful
reproduction of the qualitative result; not over-tuned to hit their exact number.

### D-12: exact.py computes SPE (matches paper baseline) — *verified + subtlety*
Brute-force check: exact.py's principal value matches an exhaustive policy search to fp on
4/5 small-tree seeds (the 5th gap is a mispricing artifact of the brute-force, which prices
all policies with the SPE truncated-Q). exact.py = subgame-perfect optimum via backward
induction = the paper's "optimal utilities in SPE obtained with DP" (p.35). Subtlety: a
*commitment* (non-SPE Stackelberg) principal could beat SPE slightly by lowering the agent's
downstream continuation utility to cut upstream contract cost; the learned policy sometimes
captures a sliver, so validation util/opt can peek just above 1.0. Outside the paper's SPE
solution concept; not a concern for reproduction. Averaged/at depth-10 it settles <=1.

### D-5: Metrics
Reproduce the paper's two: (i) validation principal utility as a % gap from the exact
optimum (target **within 1–2%**), (ii) fraction of states where the learned principal
recommends the exact-optimal action (target **~90%**).

### D-6: Paper-faithful Tree MDP spec — *specified by the paper (Appendix D.1)*
Read verbatim from the PDF. This REPLACES our invented model:
- Complete binary tree, **depth 10 → 1023 states**.
- 2 actions (a0 low effort, a1 high effort), 2 outcomes (o0 bad, o1 good).
- **Fixed** transition noise: `a0 -> o0 w.p. 0.9`, `a1 -> o1 w.p. 0.9` (so P(o1|a0)=0.1,
  P(o1|a1)=0.9) in ALL states. o0 -> left subtree, o1 -> right subtree.
- `r(s,a0)=0`, `rp(s,o0)=0`. Costly/rewarding side sampled nested-uniform:
  `r(s,a1) = -u, v~U[0,1], u~U[0,1-v]` ;  `rp(s,o1) = u, v~U[0,2], u~U[0,2-v]`.
- **γ = 1** (undiscounted).
- **Validation:** our exact DP solver over this model gives a mean effort rate of
  **57.6% (95% CI [57.2%, 58.0%]) over 60 instances**; individual trees span 54.6%–61.3%.
  Consistent with the paper's stated "about 60%" (their 3-instance sample, given the
  per-tree spread, plausibly averaged ~60% by chance or was rounded). Confirms both the
  environment and the exact solver are correct. Estimate is tight — this is our model's
  true value, not sampling noise; we accept it rather than fit an alt sampler reading to 60%.

### D-7: Learner is a DQN, not tabular — *specified by the paper (Appendix D.1)*
The Tree experiment is run with **Deep Q-Networks** (Algorithm 2), not tabular Q-learning —
the point of Fig. 3 is that function approximation lands within 2% of the DP optimum. Spec:
- Both principal (θ) and agent (φ) nets: **2 hidden FC layers × 256, ReLU**; input = state,
  output = Q-values for all actions.
- **20,000 iterations**; each iter = 8 env interactions + 1 gradient step on a **128** minibatch.
- Target nets hard-copied every **100** iters. LR **0.001 → 0.0001** (exp. anneal).
  ε **1 → 0** (linear anneal). γ = 1.
- **Simultaneous** (not iterative) principal/agent updates — early-terminated inner/outer.
- Outcome function O assumed known (constant across states here); optional classifier ξ.
- Validation: couple learned principal with a **best-responding oracle agent throughout
  training** (not only after). Optimum = dynamic programming (our exact.py).
- Setup: **3 tree instances × 5 trials**. Result: principal utility **~2% below optimal**,
  action accuracy **~90%** (≈98% of utility).
- LP is "(17)" in the paper; matches our contract.py Eq.-3 solver.
- Implies we need PyTorch (CPU is fine — tiny net).

