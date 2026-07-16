"""Algorithm 2 — deep Q-learning of principal + agent in the single-agent Tree MDP.

Reproduces Ivanov et al. (2024) Figure 3: a DQN implementation of the contract meta-algorithm
whose principal utility should land within ~2% of the exact DP optimum, recommending the
optimal action in ~90% of states.

Run:  python experiments/Tree_MDP/train_tree.py --depth 10 --updates 20000
      python experiments/Tree_MDP/train_tree.py --smoke        # quick depth-5 sanity run
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, SRC)

from tree_mdp import TreeMDP                    # noqa: E402
from exact import ExactSolution                 # noqa: E402
from contract import binary_contract            # noqa: E402
from dqn import QNet, ReplayBuffer              # noqa: E402


# --------------------------------------------------------------------- validation
def oracle_eval(mdp, contracts, recommended, tol=1e-9):
    """Exact backward-induction eval: an oracle agent best-responds to the principal's
    offered ``contracts`` (n_internal x 2); return (principal_value, agent_value, actions).

    Standard contract-theory tie-breaking: when the agent is indifferent (within ``tol``),
    it follows the principal's ``recommended`` action -- the optimal contract pays exactly
    to the point of indifference, so this tie-break is what makes it work. A real (>tol)
    underpayment still causes a genuine deviation (the paper's discontinuity)."""
    Va = np.zeros(mdp.n_nodes)
    Vp = np.zeros(mdp.n_nodes)
    actions = np.full(mdp.n_internal, -1, dtype=int)
    g = mdp.gamma
    for s in mdp.internal_nodes():
        b = contracts[s]
        c0, c1 = mdp.child(s, 0), mdp.child(s, 1)
        full = np.empty(2)
        Qbar = np.empty(2)
        for a in range(2):
            O = mdp.outcome_probs(s, a)
            Qbar[a] = mdp.agent_reward[s, a] + g * (O[0] * Va[c0] + O[1] * Va[c1])
            full[a] = Qbar[a] + O @ b
        a_rec = int(recommended[s])
        a_star = a_rec if full[a_rec] >= full.max() - tol else int(np.argmax(full))
        actions[s] = a_star
        Va[s] = full[a_star]
        O = mdp.outcome_probs(s, a_star)
        Vp[s] = (O[0] * (mdp.principal_reward[s, 0] - b[0] + g * Vp[c0]) +
                 O[1] * (mdp.principal_reward[s, 1] - b[1] + g * Vp[c1]))
    return float(Vp[0]), float(Va[0]), actions


def principal_policy(mdp, net_p, net_a, p_o1, device, nudge=0.0):
    """The principal's current policy: recommended action + offered contract per state."""
    states = np.arange(mdp.n_internal)
    with torch.no_grad():
        qp = net_p.q_all(states, device).cpu().numpy()
        qa = net_a.q_all(states, device).cpu().numpy()   # truncated Q
    a_p = qp.argmax(1)
    b = binary_contract(qa, a_p, p_o1, nudge=nudge)
    return a_p, b


# --------------------------------------------------------------------- training
def train_one(mdp, exact, updates=20000, interactions=8, batch=128, target_every=100,
              lr0=1e-3, lr1=1e-4, buffer_cap=100_000, prefill=10_000,
              eval_every=500, seed=0, device="cpu", verbose=True, nudge=0.0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    S, A = mdp.n_internal, mdp.n_actions
    p_o1 = mdp.p_outcome1[0]                      # constant across states: [P(o1|a0), P(o1|a1)]
    g = mdp.gamma

    net_p, net_a = QNet(S, A).to(device), QNet(S, A).to(device)
    tgt_p, tgt_a = QNet(S, A).to(device), QNet(S, A).to(device)
    tgt_p.load_state_dict(net_p.state_dict())
    tgt_a.load_state_dict(net_a.state_dict())
    opt_p = torch.optim.Adam(net_p.parameters(), lr=lr0)
    opt_a = torch.optim.Adam(net_a.parameters(), lr=lr0)

    buf = ReplayBuffer(buffer_cap, rng=rng)

    def step(s, ap):
        """Apply recommended action ap at state s; return (ra, rp, o, done, sp)."""
        o = int(rng.random() < mdp.p_outcome1[s, ap])
        ra = mdp.agent_reward[s, ap]
        rp = mdp.principal_reward[s, o]
        sp = mdp.child(s, o)
        return ra, rp, o, mdp.is_leaf(sp), sp

    # pre-fill buffer with random transitions
    s = 0
    for _ in range(prefill):
        ap = int(rng.integers(A))
        ra, rp, o, done, sp = step(s, ap)
        buf.add(s, ap, ra, rp, o, done, sp)
        s = 0 if done else sp

    history = {"iter": [], "p_ratio": [], "a_gap": [], "acc": [], "acc_realized": []}
    p_opt, a_opt = exact.principal_value, float(exact.V_agent[0])
    s = 0
    t0 = time.time()
    for it in range(updates):
        frac = it / updates
        eps = 1.0 - frac                                   # linear 1 -> 0
        lr = lr0 * (lr1 / lr0) ** frac                     # exp anneal
        for opt in (opt_p, opt_a):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ---- interact ----
        for _ in range(interactions):
            if rng.random() < eps:
                ap = int(rng.integers(A))
            else:
                with torch.no_grad():
                    ap = int(net_p.q_all([s], device)[0].argmax())
            ra, rp, o, done, sp = step(s, ap)
            buf.add(s, ap, ra, rp, o, done, sp)
            s = 0 if done else sp

        # ---- learn ----
        bs, bap, bra, brp, bo, bdone, bsp = buf.sample(batch)
        ar = np.arange(batch)
        done_t = torch.as_tensor(bdone, device=device)

        qp_s = net_p.q_all(bs, device)[ar, bap]                       # qθ(s,ap)  online
        qa_s = net_a.q_all(bs, device)[ar, bap]                       # Q̄φ(s,ap)  online
        with torch.no_grad():
            ap_next = net_p.q_all(bsp, device).argmax(1).cpu().numpy()      # online arg
            qp_sp = tgt_p.q_all(bsp, device).cpu().numpy()[ar, ap_next]     # qθ'(s',ap')
            qa_s_all = tgt_a.q_all(bs, device).cpu().numpy()               # Q̄φ'(s,·)
            qa_sp_all = tgt_a.q_all(bsp, device).cpu().numpy()            # Q̄φ'(s',·)
            qa_sp_next = qa_sp_all[ar, ap_next]                          # Q̄φ'(s',ap')

        # contracts (closed form) from target agent net, with nudging margin
        b_s = binary_contract(qa_s_all, bap, p_o1, nudge=nudge)       # b*(s,ap)
        pay_realized = b_s[ar, bo]                                    # payment for realized o
        b_sp = binary_contract(qa_sp_all, ap_next, p_o1, nudge=nudge)  # b*(s',ap')
        exp_pay_sp = (1 - p_o1[ap_next]) * b_sp[:, 0] + p_o1[ap_next] * b_sp[:, 1]

        yp = torch.as_tensor(brp - pay_realized, device=device) + g * (1 - done_t) * torch.as_tensor(qp_sp, device=device)
        ya = torch.as_tensor(bra, device=device) + g * (1 - done_t) * torch.as_tensor(exp_pay_sp + qa_sp_next, device=device)

        loss_p = F.mse_loss(qp_s, yp.float())
        loss_a = F.mse_loss(qa_s, ya.float())
        opt_p.zero_grad(); loss_p.backward(); opt_p.step()
        opt_a.zero_grad(); loss_a.backward(); opt_a.step()

        if (it + 1) % target_every == 0:
            tgt_p.load_state_dict(net_p.state_dict())
            tgt_a.load_state_dict(net_a.state_dict())

        # ---- validate against exact optimum ----
        if it % eval_every == 0 or it == updates - 1:
            a_p, b = principal_policy(mdp, net_p, net_a, p_o1, device, nudge=nudge)
            vp, va, realized = oracle_eval(mdp, b, a_p)
            acc = float(np.mean(a_p == exact.opt_action))
            acc_r = float(np.mean(realized == exact.opt_action))
            history["iter"].append(it)
            history["p_ratio"].append(vp / p_opt)
            history["a_gap"].append(va - a_opt)
            history["acc"].append(acc)
            history["acc_realized"].append(acc_r)
            if verbose:
                print(f"  it={it:6d}  p_util/opt={vp/p_opt:6.3f}  "
                      f"acc={acc:5.1%}  eps={eps:4.2f}  lr={lr:.1e}  ({time.time()-t0:5.1f}s)",
                      flush=True)

    history["p_opt"], history["a_opt"] = p_opt, a_opt
    history["net_p"], history["net_a"] = net_p, net_a
    return history


# --------------------------------------------------------------------- entry
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--instances", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--nudge", type=float, default=0.01, help="strict-preference margin ξ")
    ap.add_argument("--smoke", action="store_true", help="quick depth-5, 2000-update run")
    args = ap.parse_args()

    if args.smoke:
        args.depth, args.updates = 5, 2000

    print(f"Tree MDP DQN (Algorithm 2)  depth={args.depth}  updates={args.updates}  "
          f"instances={args.instances}  trials={args.trials}")
    for inst in range(args.instances):
        mdp = TreeMDP(depth=args.depth, rng=np.random.default_rng(1000 + inst))
        exact = ExactSolution(mdp)
        print(f"\ninstance {inst}: states={mdp.n_internal}  optimal_principal_util={exact.principal_value:.3f}")
        for tr in range(args.trials):
            print(f" trial {tr}:")
            h = train_one(mdp, exact, updates=args.updates, seed=100 * inst + tr, nudge=args.nudge)
            print(f"  -> final p_util/opt={h['p_ratio'][-1]:.3f}  acc={h['acc'][-1]:.1%}")


if __name__ == "__main__":
    main()
