"""Baselines for the Coin Game (Ivanov et al. 2024, Section 5.2).

All three baselines are independent DQN agents (no principal, no contracts) trained with a
reward-redistribution parameter p -- the Christoffersen et al. (2023) constant-proportion
scheme:  r_i' = (1 - p) * r_i + p * mean_j r_j.

  * p = 0   -> SELFISH baseline (agents maximize own reward).
  * p = 1   -> COOPERATIVE / optimal baseline (agents maximize social welfare).
  * p = p*  -> CONSTANT-PROPORTION heuristic, budget-matched to what our method pays.

Social welfare is always measured on the RAW rewards Sigma_i r_i.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from coin_game import CoinGame, N_ACTIONS          # noqa: E402
from networks import ConvQNet, agent_view          # noqa: E402


def _t(x, device):
    return torch.as_tensor(x, device=device)


def av(obs_batch, i, device):
    return agent_view(_t(np.asarray(obs_batch, dtype=np.float32), device), i)


def train_baseline(p, size=3, horizon=20, iters=40_000, batch=128, target_every=100,
                   lr0=5e-4, lr1=1e-4, gamma=0.99, eps0=0.4, eps1=0.0,
                   buffer_cap=100_000, prefill=5_000, eval_every=2_000, seed=0,
                   device="cpu", verbose=True):
    """Independent shared-parameter DQN agents under reward redistribution proportion p."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = CoinGame(size=size, horizon=horizon, rng=rng)
    net = ConvQNet(size, N_ACTIONS, n_extra=0).to(device)
    tgt = ConvQNet(size, N_ACTIONS, n_extra=0).to(device); tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr0)

    # simple replay of (s, a[2], r'[2], s', done) with redistributed reward
    cap = buffer_cap
    S = np.zeros((cap, 4, size, size), np.float32); SP = np.zeros((cap, 4, size, size), np.float32)
    A = np.zeros((cap, 2), np.int64); R = np.zeros((cap, 2), np.float32); D = np.zeros(cap, np.float32)
    n = i_ = 0

    def redistribute(r):
        return (1 - p) * r + p * r.mean()

    def act(obs, eps):
        a = np.empty(2, dtype=int)
        for i in range(2):
            if rng.random() < eps:
                a[i] = rng.integers(N_ACTIONS)
            else:
                with torch.no_grad():
                    a[i] = int(net(av(obs[None], i, device))[0].argmax())
        return a

    obs = env.reset()
    for _ in range(prefill):
        a = rng.integers(N_ACTIONS, size=2); sp, r, done = env.step(a)
        S[i_], SP[i_], A[i_], R[i_], D[i_] = obs, sp, a, redistribute(r), done
        i_ = (i_ + 1) % cap; n = min(n + 1, cap); obs = sp if not done else env.reset()

    hist = {"iter": [], "welfare": []}
    obs = env.reset(); t0 = time.time()
    for it in range(iters):
        frac = it / iters
        eps = eps0 + (eps1 - eps0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        for pg in opt.param_groups: pg["lr"] = lr

        a = act(obs, eps); sp, r, done = env.step(a)
        S[i_], SP[i_], A[i_], R[i_], D[i_] = obs, sp, a, redistribute(r), done
        i_ = (i_ + 1) % cap; n = min(n + 1, cap); obs = sp if not done else env.reset()

        idx = rng.integers(0, n, size=batch); done_t = _t(D[idx], device)
        loss_terms = []
        for i in range(2):
            vi, vpi = av(S[idx], i, device), av(SP[idx], i, device)
            ai = _t(A[idx, i], device)
            q_i = net(vi).gather(1, ai[:, None]).squeeze(1)
            with torch.no_grad():
                qn = tgt(vpi).max(1).values
                y = _t(R[idx, i], device) + gamma * (1 - done_t) * qn
            loss_terms.append(F.smooth_l1_loss(q_i, y))
        loss = sum(loss_terms)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if (it + 1) % target_every == 0:
            tgt.load_state_dict(net.state_dict())

        if it % eval_every == 0 or it == iters - 1:
            w = eval_welfare(net, size, horizon, device, rng)
            hist["iter"].append(it); hist["welfare"].append(w)
            if verbose:
                print(f"  [p={p:.2f}] it={it:6d} welfare={w:6.3f} eps={eps:4.2f} "
                      f"({time.time()-t0:5.1f}s)", flush=True)
    return {"net": net, "history": hist, "p": p}


@torch.no_grad()
def eval_welfare(net, size, horizon, device, rng, episodes=40):
    env = CoinGame(size=size, horizon=horizon, rng=rng)
    tot = 0.0
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            a = np.array([int(net(av(obs[None], i, device))[0].argmax()) for i in range(2)])
            obs, r, done = env.step(a); tot += float(r.sum())
    return tot / episodes


if __name__ == "__main__":
    for p, name in [(0.0, "selfish"), (1.0, "cooperative")]:
        print(f"== {name} (p={p}) ==")
        out = train_baseline(p, iters=3000, verbose=True)
        print(f"-> welfare={out['history']['welfare'][-1]:.3f}")
