"""Auto-calibrate the TreeMDP so contracts actually bind.

A contract experiment is only meaningful if the principal pays to induce the costly
("effort") action in a non-trivial fraction of states -- not ~0% (contracts irrelevant)
and not ~100% (agent always wants effort anyway / it's free). We sweep the environment's
reward/cost knobs, and for each candidate solve MANY random trees EXACTLY, measuring:

    effort_rate  = mean fraction of internal states where opt_action != 0 (effort induced)
    pay_share    = mean (total expected payment) / (principal gross reward)   [how much it pays]

We select the config whose effort_rate is closest to the target band's centre while keeping
a healthy positive pay_share. Result is written to notes/decision_log.md-friendly output and
saved as env_config.json for the main experiment to load.
"""

from __future__ import annotations

import os
import sys
import json
import itertools

import numpy as np

# make src importable
SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, SRC)

from tree_mdp import TreeMDP          # noqa: E402
from exact import ExactSolution       # noqa: E402


def evaluate_config(cfg, depth, n_trees, seed0):
    """Average effort_rate and pay_share over ``n_trees`` random trees for one config."""
    effort_rates, pay_shares, values = [], [], []
    for k in range(n_trees):
        mdp = TreeMDP(depth=depth, rng=np.random.default_rng(seed0 + k), **cfg)
        sol = ExactSolution(mdp)
        effort_rates.append(np.mean(sol.opt_action != 0))
        gross = sum(mdp.expected_principal_reward(s, sol.opt_action[s])
                    for s in range(mdp.n_internal))
        total_pay = sol.payment.sum()
        pay_shares.append(total_pay / gross if gross > 1e-9 else 0.0)
        values.append(sol.principal_value)
    return {
        "effort_rate": float(np.mean(effort_rates)),
        "pay_share": float(np.mean(pay_shares)),
        "principal_value": float(np.mean(values)),
    }


def main():
    depth = 5
    n_trees = 200
    seed0 = 1000
    target_lo, target_hi = 0.30, 0.60
    target_mid = 0.5 * (target_lo + target_hi)

    # Sweep grid: max effort cost, effort->outcome lever, principal reward spread.
    cost_his = [0.15, 0.25, 0.35, 0.5]
    effort_biases = [0.4, 0.6, 0.8]
    principal_ranges = [(0.0, 1.0), (0.0, 2.0)]

    print(f"Sweeping over {len(cost_his)*len(effort_biases)*len(principal_ranges)} configs, "
          f"{n_trees} trees each at depth {depth}...\n")
    print(f"{'cost_hi':>7} {'bias':>5} {'p_range':>10} | {'effort':>7} {'pay_sh':>7} {'value':>7}")
    print("-" * 56)

    results = []
    for cost_hi, bias, prange in itertools.product(cost_his, effort_biases, principal_ranges):
        cfg = dict(n_actions=2, gamma=0.9,
                   effort_cost_range=(0.0, cost_hi),
                   principal_reward_range=prange,
                   effort_bias=bias)
        m = evaluate_config(cfg, depth, n_trees, seed0)
        results.append((cfg, m))
        in_band = "*" if target_lo <= m["effort_rate"] <= target_hi else " "
        print(f"{cost_hi:>7} {bias:>5} {str(prange):>10} | "
              f"{m['effort_rate']:>7.2%} {m['pay_share']:>7.2%} {m['principal_value']:>7.3f} {in_band}")

    # Pick: within band if possible, closest effort_rate to target_mid, tie-break higher pay_share.
    in_band = [(c, m) for c, m in results if target_lo <= m["effort_rate"] <= target_hi]
    pool = in_band if in_band else results
    best_cfg, best_m = min(
        pool, key=lambda cm: (abs(cm[1]["effort_rate"] - target_mid), -cm[1]["pay_share"]))

    print("\nSelected config:")
    print(json.dumps(best_cfg, indent=2))
    print("metrics:", json.dumps(best_m, indent=2))

    out = os.path.join(os.path.dirname(__file__), "env_config.json")
    with open(out, "w") as f:
        json.dump({"config": best_cfg, "metrics": best_m,
                   "calibration": {"depth": depth, "n_trees": n_trees,
                                   "target_band": [target_lo, target_hi]}}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
