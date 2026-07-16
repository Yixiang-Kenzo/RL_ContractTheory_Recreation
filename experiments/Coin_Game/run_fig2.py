"""Reproduce Figure 2 (Coin Game) of Ivanov et al. (2024) on the 3x3 grid (their Fig. 4 setting).

Pipeline:
  1. Our method: training phase (Algorithm 3) -> validation phase (fresh DQNs).
  2. Baselines (independent DQNs): selfish (p=0), cooperative/optimal (p=1),
     constant-proportion (p = our method's post-validation payment share, budget-matched).
  3. Plot: (a) social welfare, (b) proportion of welfare paid, (c) follow-accuracy.

Usage:
    python experiments/Coin_Game/run_fig2.py --iters 40000 --val-iters 40000 --nudge-frac 0.1
    python experiments/Coin_Game/run_fig2.py --plot-only
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
sys.path.insert(0, HERE)

from train_coin import train_phase, val_phase      # noqa: E402
from baselines import train_baseline               # noqa: E402

RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(RESULTS, "fig2_data.pkl")


def run(size, horizon, iters, val_iters, nudge_frac, seed):
    os.makedirs(RESULTS, exist_ok=True)
    print("== our method: training ==")
    trained = train_phase(size=size, horizon=horizon, iters=iters, seed=seed)
    print("== our method: validation ==")
    nudge = nudge_frac * trained["history"]["welfare"][-1] / horizon
    print(f"   nudge = {nudge_frac:.0%} of per-step welfare = {nudge:.4f}")
    val = val_phase(trained, iters=val_iters, nudge=nudge, seed=seed + 1)
    matched_p = float(np.clip(val["history"]["pay_prop"][-1], 0, 1))

    print(f"== baseline: selfish (p=0) ==")
    selfish = train_baseline(0.0, size=size, horizon=horizon, iters=val_iters, seed=seed + 2)
    print(f"== baseline: cooperative (p=1) ==")
    coop = train_baseline(1.0, size=size, horizon=horizon, iters=val_iters, seed=seed + 3)
    print(f"== baseline: constant-proportion (p={matched_p:.2f}) ==")
    constp = train_baseline(matched_p, size=size, horizon=horizon, iters=val_iters, seed=seed + 4)

    data = {
        "train": trained["history"], "val": val["history"],
        "selfish": selfish["history"], "coop": coop["history"], "constp": constp["history"],
        "matched_p": matched_p, "meta": dict(size=size, horizon=horizon, nudge_frac=nudge_frac),
    }
    with open(DATA, "wb") as f:
        pickle.dump(data, f)
    print(f"saved -> {DATA}")
    return data


def plot(data=None):
    if data is None:
        with open(DATA, "rb") as f:
            data = pickle.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    nf = data["meta"]["nudge_frac"]

    # (a) social welfare
    ax = axes[0]
    ax.plot(data["coop"]["iter"], data["coop"]["welfare"], color="green", ls="--", label="cooperative (optimal)")
    ax.plot(data["selfish"]["iter"], data["selfish"]["welfare"], color="red", ls="--", label="selfish")
    ax.plot(data["val"]["iter"], data["val"]["welfare"], color="blue", label="ours (validation)")
    ax.plot(data["constp"]["iter"], data["constp"]["welfare"], color="orange", label=f"constant-prop (p={data['matched_p']:.2f})")
    ax.set(title="(a) Social welfare", xlabel="iteration", ylabel="welfare/episode"); ax.legend(fontsize=8)

    # (b) proportion of welfare paid
    ax = axes[1]
    ax.plot(data["val"]["iter"], np.array(data["val"]["pay_prop"]) * 100, color="blue", label="ours (validation)")
    ax.axhline(data["matched_p"] * 100, color="orange", ls="--", label="constant-prop budget")
    ax.set(title="(b) Proportion of welfare paid", xlabel="iteration", ylabel="%"); ax.legend(fontsize=8)

    # (c) follow-accuracy
    ax = axes[2]
    ax.plot(data["val"]["iter"], np.array(data["val"]["accuracy"]) * 100, color="blue")
    ax.axhline(80, color="gray", ls=":", alpha=0.6); ax.axhline(90, color="gray", ls=":", alpha=0.6)
    ax.set(title="(c) Follow accuracy", xlabel="iteration", ylabel="% recommendations followed", ylim=(0, 101))

    for a in axes:
        a.grid(alpha=0.2)
    fig.suptitle(f"Coin Game 3x3 — reproduction of Ivanov et al. (2024) "
                 f"(nudge={nf:.0%} of welfare)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(RESULTS, "fig2.png")
    fig.savefig(out, dpi=130); print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--iters", type=int, default=40_000)
    ap.add_argument("--val-iters", type=int, default=40_000)
    ap.add_argument("--nudge-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if args.plot_only:
        plot()
    else:
        data = run(args.size, args.horizon, args.iters, args.val_iters, args.nudge_frac, args.seed)
        plot(data)


if __name__ == "__main__":
    main()
