"""Reproduce Figure 3 of Ivanov et al. (2024): Tree MDP learning curves.

Runs the paper's protocol -- 3 tree instances x 5 trials -- of Algorithm 2, records the
validation learning curves (principal utility, agent utility, action accuracy vs the exact
DP/SPE optimum), saves the raw data, and plots the 3-panel figure.

Usage:
    python experiments/Tree_MDP/run_fig3.py                 # full 3x5 run + plot
    python experiments/Tree_MDP/run_fig3.py --instances 3 --trials 5 --depth 10
    python experiments/Tree_MDP/run_fig3.py --plot-only      # replot from saved data
"""

from __future__ import annotations

import os
import sys
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from tree_mdp import TreeMDP           # noqa: E402
from exact import ExactSolution        # noqa: E402
from train_tree import train_one       # noqa: E402

RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(RESULTS, "fig3_data.pkl")


def run(instances, trials, depth, updates, nudge):
    os.makedirs(RESULTS, exist_ok=True)
    data = []  # one entry per instance: dict with optima + list of trial histories
    for inst in range(instances):
        mdp = TreeMDP(depth=depth, rng=np.random.default_rng(1000 + inst))
        exact = ExactSolution(mdp)
        print(f"instance {inst}: states={mdp.n_internal}  "
              f"p_opt={exact.principal_value:.3f}  a_opt={exact.V_agent[0]:.3f}", flush=True)
        entry = {"p_opt": exact.principal_value, "a_opt": float(exact.V_agent[0]), "trials": []}
        for tr in range(trials):
            h = train_one(mdp, exact, updates=updates, seed=100 * inst + tr,
                          nudge=nudge, verbose=False)
            entry["trials"].append({k: h[k] for k in ("iter", "p_ratio", "a_gap", "acc")})
            print(f"  trial {tr}: p_util/opt={h['p_ratio'][-1]:.3f}  acc={h['acc'][-1]:.1%}",
                  flush=True)
        data.append(entry)
    with open(DATA, "wb") as f:
        pickle.dump({"data": data, "meta": dict(depth=depth, updates=updates, nudge=nudge)}, f)
    print(f"saved -> {DATA}")
    return data


def plot(data=None):
    if data is None:
        with open(DATA, "rb") as f:
            data = pickle.load(f)["data"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for inst, (entry, c) in enumerate(zip(data, colors)):
        its = np.array(entry["trials"][0]["iter"])
        # stack trials
        p_util = np.array([np.array(t["p_ratio"]) * entry["p_opt"] for t in entry["trials"]])
        a_util = np.array([np.array(t["a_gap"]) + entry["a_opt"] for t in entry["trials"]])
        acc = np.array([t["acc"] for t in entry["trials"]])

        def band(ax, y, label=None):
            m, se = y.mean(0), y.std(0) / np.sqrt(y.shape[0])
            ax.plot(its, m, color=c, label=label)
            ax.fill_between(its, m - se, m + se, color=c, alpha=0.2)

        band(axes[0], p_util, f"instance {inst}")
        axes[0].axhline(entry["p_opt"], color=c, ls="--", alpha=0.7)
        band(axes[1], a_util)
        axes[1].axhline(entry["a_opt"], color=c, ls="--", alpha=0.7)
        band(axes[2], acc)

    axes[0].set(title="(a) Principal's utility", xlabel="iteration", ylabel="utility")
    axes[0].legend(fontsize=8)
    axes[1].set(title="(b) Agent's utility", xlabel="iteration", ylabel="utility")
    axes[2].set(title="(c) Accuracy of principal's action", xlabel="iteration", ylabel="fraction optimal")
    axes[2].axhline(0.9, color="gray", ls=":", alpha=0.6)
    axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Tree MDP — reproduction of Ivanov et al. (2024) Figure 3 "
                 "(solid = learned DQN, dashed = SPE optimum)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig3.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--nudge", type=float, default=0.01)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    if args.plot_only:
        plot()
    else:
        data = run(args.instances, args.trials, args.depth, args.updates, args.nudge)
        plot(data)


if __name__ == "__main__":
    main()
