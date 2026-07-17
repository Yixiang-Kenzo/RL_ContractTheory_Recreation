"""Diagnose the ~78% follow-rate and the contract-net faithfulness question.

Part 1 (FAITHFULNESS): train with the *faithful* online contract (Alg.3 line 14) and check
  whether Huber + grad-clip alone keep it stable, or whether the target-net contract is truly
  needed. Compare welfare trajectories.
Part 2 (HYPOTHESES): using one target-net principal, run validation across several seeds
  (H4: seed variance) and inspect each curve's tail (H1: undertraining); then measure
  compliance regret on the converged agents (H2: contract quality -- how far the contract is
  from making 'follow' the best response).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from train_coin import train_phase, val_phase, av, _onehot, _t   # noqa: E402
from coin_game import CoinGame, N_ACTIONS                        # noqa: E402


@torch.no_grad()
def compliance_regret(trained, psi, nudge, seed, episodes=60, device="cpu"):
    """For the converged validation net psi, at each visited state measure
       regret = max_a Qψ(s,ap,a) - Qψ(s,ap,follow)   (>0 => agent prefers to deviate).
    Returns (mean regret over deviating states, fraction of agent-states with regret>0)."""
    cfg = trained["cfg"]; size, horizon = cfg["size"], cfg["horizon"]
    qtheta = trained["qtheta"]
    env = CoinGame(size=size, horizon=horizon, rng=np.random.default_rng(seed))
    regrets, n_pos, n = [], 0, 0
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            ap = np.array([int(qtheta(av(obs[None], i, device))[0].argmax()) for i in range(2)])
            a = np.empty(2, dtype=int)
            for i in range(2):
                q = psi(av(obs[None], i, device), _onehot([ap[i]], N_ACTIONS, device))[0]
                a[i] = int(q.argmax())
                reg = float(q.max() - q[ap[i]])          # >0 iff a best action != recommended
                n += 1
                if reg > 1e-6:
                    n_pos += 1; regrets.append(reg)
            obs, r, done = env.step(a)
    return (float(np.mean(regrets)) if regrets else 0.0), n_pos / max(n, 1)


def main():
    size, horizon = 3, 20
    train_iters, val_iters = 40_000, 30_000
    nudge_frac = 0.1
    n_val_seeds = 3

    # ---------------- Part 1: faithfulness (online vs target contract) ----------------
    print("=" * 60)
    print("PART 1  faithfulness: does the FAITHFUL online contract stay stable?")
    print("=" * 60)
    for cnet in ("online", "target"):
        out = train_phase(size=size, horizon=horizon, iters=25_000, seed=0,
                          contract_net=cnet, eval_every=5_000, verbose=False)
        w = out["history"]["welfare"]; pp = out["history"]["pay_prop"]
        traj = "  ".join(f"{x:.1f}" for x in w)
        print(f"  contract_net={cnet:>6}: welfare traj [{traj}]  final pay_prop={pp[-1]:.0%}")
        stable = w[-1] > 0.7 * max(w) and pp[-1] < 1.0
        print(f"    -> {'STABLE' if stable else 'DIVERGED'}")

    # ---------------- Part 2: hypotheses (reuse one target principal) ----------------
    print("\n" + "=" * 60)
    print("PART 2  hypotheses: train one principal, vary validation seed")
    print("=" * 60)
    trained = train_phase(size=size, horizon=horizon, iters=train_iters, seed=0,
                          contract_net="target", verbose=False)
    per_step_w = trained["history"]["welfare"][-1] / horizon
    nudge = nudge_frac * per_step_w
    print(f"principal trained (welfare={trained['history']['welfare'][-1]:.2f}); nudge={nudge:.4f}\n")

    print(f"{'val_seed':>8} | {'follow_acc':>10} {'tail_rising?':>12} {'welfare':>8} {'pay_prop':>8}")
    print("-" * 56)
    accs = []
    for vs in range(n_val_seeds):
        v = val_phase(trained, iters=val_iters, nudge=nudge, seed=100 + vs, verbose=False)
        h = v["history"]
        acc = sum(h["accuracy"][-3:]) / 3                      # smoothed final follow-rate
        rising = h["accuracy"][-1] - h["accuracy"][-4]         # >0 => still climbing (undertrained)
        accs.append(acc)
        reg, frac = compliance_regret(trained, v["psi"], nudge, seed=999)
        print(f"{100+vs:>8} | {acc:>10.1%} {rising:>+11.1%} {sum(h['welfare'][-3:])/3:>8.2f} "
              f"{sum(h['pay_prop'][-3:])/3:>8.1%}   regret(mean/frac)={reg:.3f}/{frac:.0%}")
    accs = np.array(accs)
    print(f"\nH4 seed variance: follow_acc = {accs.mean():.1%} +/- {accs.std(ddof=1):.1%}  "
          f"(range {accs.min():.1%}-{accs.max():.1%})")
    print("Read: if mean+/-std overlaps 80%, 78% is noise. If tail still rising, undertrained.")
    print("      compliance regret = how far the contract is from making 'follow' best (H2).")


if __name__ == "__main__":
    main()
