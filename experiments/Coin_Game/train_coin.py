"""Algorithm 3 — deep Q-learning of principal + agents in the Coin Game (multi-agent).

Two phases (Ivanov et al. 2024, Appendix D.2):
  * TRAINING: principal (VDN, net θ) learns which actions to recommend + their minimal
    IC implementation; agents (shared net φ) learn selfish Q conditioned on the other agent's
    follow-bit f_{-i}. Each episode samples f=(f_i); f_i=1 agents follow recommendations,
    f_i=0 act selfishly -- this covers the space of "who follows" for the IC approximation.
  * VALIDATION: fresh black-box agents (net ψ) trained from scratch; they see the principal's
    recommendation as input and are paid iff they follow it. Tests whether independent selfish
    learners converge to following the principal.

Contract (minimal implementation), per agent i:
    b*_i(s, a_i) = max_a Qφ_i((s, f_{-i}), a) - Qφ_i((s, f_{-i}), a_i)
i.e. pay agent i the gap between its best action and the recommended one. α scales the
principal's payment-minimization (α=0.1). Coin-Game nudge (paper) = +10% of social welfare.
"""

from __future__ import annotations

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from coin_game import CoinGame, N_ACTIONS          # noqa: E402
from networks import ConvQNet, ReplayBuffer, agent_view  # noqa: E402


def _t(x, device):
    return torch.as_tensor(x, device=device)


def av(obs_batch, i, device):
    """Agent-centric torch view (B,4,H,W) for agent i from raw obs batch (numpy or tensor)."""
    return agent_view(_t(np.asarray(obs_batch, dtype=np.float32), device), i)


# ----------------------------------------------------------------- training phase
def train_phase(size=3, horizon=20, iters=50_000, batch=128, target_every=100,
                lr0=5e-4, lr1=1e-4, gamma=0.99, alpha=0.1, eps0=0.4, eps1=0.0,
                buffer_cap=100_000, prefill=5_000, eval_every=2_000, seed=0,
                device="cpu", verbose=True, contract_net="target"):
    """contract_net: 'target' (φ', our stability fix) or 'online' (Qφ, faithful to Alg.3 line 14)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = CoinGame(size=size, horizon=horizon, rng=rng)

    qtheta = ConvQNet(size, N_ACTIONS, n_extra=0).to(device)          # principal (VDN)
    qphi = ConvQNet(size, N_ACTIONS, n_extra=1).to(device)            # agents (share params); extra = f_{-i}
    tgt_theta = ConvQNet(size, N_ACTIONS, n_extra=0).to(device)
    tgt_phi = ConvQNet(size, N_ACTIONS, n_extra=1).to(device)
    tgt_theta.load_state_dict(qtheta.state_dict())
    tgt_phi.load_state_dict(qphi.state_dict())
    opt_t = torch.optim.Adam(qtheta.parameters(), lr=lr0)
    opt_p = torch.optim.Adam(qphi.parameters(), lr=lr0)

    buf = ReplayBuffer(buffer_cap, size, rng=rng)

    def select(obs, f, eps):
        """Choose both agents' actions given follow-vector f and exploration eps."""
        acts = []
        for i in range(2):
            vt = av(obs[None], i, device)
            if f[i] == 1:  # follow: recommendation is eps-greedy over qtheta
                if rng.random() < eps:
                    a = int(rng.integers(N_ACTIONS))
                else:
                    with torch.no_grad():
                        a = int(qtheta(vt)[0].argmax())
            else:          # selfish: eps-greedy over qphi conditioned on the other's follow-bit
                extra = _t([[float(f[1 - i])]], device)
                if rng.random() < eps:
                    a = int(rng.integers(N_ACTIONS))
                else:
                    with torch.no_grad():
                        a = int(qphi(vt, extra)[0].argmax())
            acts.append(a)
        return np.array(acts)

    # prefill with random transitions (f random per episode)
    obs = env.reset(); f = rng.integers(0, 2, size=2)
    for _ in range(prefill):
        a = rng.integers(N_ACTIONS, size=2)
        sp, r, done = env.step(a)
        buf.add(obs, a, f, r, sp, done)
        obs = sp
        if done:
            obs = env.reset(); f = rng.integers(0, 2, size=2)

    history = {"iter": [], "welfare": [], "pay_prop": [], "accuracy": []}
    obs = env.reset(); f = rng.integers(0, 2, size=2)
    t0 = time.time()
    for it in range(iters):
        frac = it / iters
        eps = eps0 + (eps1 - eps0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        for opt in (opt_t, opt_p):
            for pg in opt.param_groups:
                pg["lr"] = lr

        # ---- one environment interaction ----
        a = select(obs, f, eps)
        sp, r, done = env.step(a)
        buf.add(obs, a, f, r, sp, done)
        obs = sp
        if done:
            obs = env.reset(); f = rng.integers(0, 2, size=2)

        # ---- one gradient update ----
        bs, ba, bf, br, bsp, bd = buf.sample(batch)
        done_t = _t(bd, device)
        loss_theta_terms, loss_phi_terms = [], []
        sum_q_theta = torch.zeros(batch, device=device)
        sum_yp = torch.zeros(batch, device=device)
        for i in range(2):
            vi, vpi = av(bs, i, device), av(bsp, i, device)
            ai = _t(ba[:, i], device)
            f_other = _t(bf[:, 1 - i].astype(np.float32), device).unsqueeze(1)
            ones = torch.ones_like(f_other)

            q_theta_i = qtheta(vi).gather(1, ai[:, None]).squeeze(1)      # qθ_i(s,a_i)
            q_phi_i = qphi(vi, f_other).gather(1, ai[:, None]).squeeze(1)  # Qφ_i(s,f_{-i},a_i)
            with torch.no_grad():
                qp_next = tgt_theta(vpi).max(1).values                    # max qθ'(s')
                qa_next = tgt_phi(vpi, f_other).max(1).values             # max Qφ'(s',f_{-i})
                # contract with f_{-i}=1 (Alg.3 line 14). 'online' = faithful Qφ;
                # 'target' = φ' (our stability fix). Testing which is needed.
                qphi_ones = (qphi if contract_net == "online" else tgt_phi)(vi, ones)
                b_i = (qphi_ones.max(1, keepdim=True).values - qphi_ones).gather(1, ai[:, None]).squeeze(1)
                yp_i = _t(br[:, i], device) - alpha * b_i + gamma * (1 - done_t) * qp_next
                ya_i = _t(br[:, i], device) + gamma * (1 - done_t) * qa_next

            sum_q_theta = sum_q_theta + q_theta_i          # VDN: sum over agents
            sum_yp = sum_yp + yp_i
            loss_phi_terms.append(F.smooth_l1_loss(q_phi_i, ya_i))

        loss_theta = F.smooth_l1_loss(sum_q_theta, sum_yp)  # VDN joint loss (Huber)
        loss_phi = sum(loss_phi_terms)
        opt_t.zero_grad(); loss_theta.backward()
        torch.nn.utils.clip_grad_norm_(qtheta.parameters(), 10.0); opt_t.step()
        opt_p.zero_grad(); loss_phi.backward()
        torch.nn.utils.clip_grad_norm_(qphi.parameters(), 10.0); opt_p.step()

        if (it + 1) % target_every == 0:
            tgt_theta.load_state_dict(qtheta.state_dict())
            tgt_phi.load_state_dict(qphi.state_dict())

        if it % eval_every == 0 or it == iters - 1:
            w, pp, acc = eval_training(qtheta, qphi, size, horizon, device, rng)
            history["iter"].append(it)
            history["welfare"].append(w)
            history["pay_prop"].append(pp)
            history["accuracy"].append(acc)
            if verbose:
                print(f"  it={it:6d} welfare={w:6.3f} pay_prop={pp:5.1%} "
                      f"follow_acc={acc:5.1%} eps={eps:4.2f} ({time.time()-t0:5.1f}s)", flush=True)

    return {"qtheta": qtheta, "qphi": qphi, "history": history,
            "cfg": dict(size=size, horizon=horizon, gamma=gamma, alpha=alpha)}


@torch.no_grad()
def eval_training(qtheta, qphi, size, horizon, device, rng, episodes=40):
    """Greedy eval with BOTH agents following recommendations: social welfare, payment
    proportion, and (trivially 100% here) recommendation-following -- used as the training
    curve. Payment = Σ contract for the recommended action (f_{-i}=1)."""
    env = CoinGame(size=size, horizon=horizon, rng=rng)
    tot_w, tot_pay = 0.0, 0.0
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            a = np.empty(2, dtype=int); pay = 0.0
            for i in range(2):
                vt = av(obs[None], i, device)
                ap = int(qtheta(vt)[0].argmax())
                a[i] = ap
                qp1 = qphi(vt, _t([[1.0]], device))[0]
                pay += float(qp1.max() - qp1[ap])
            obs, r, done = env.step(a)
            tot_w += float(r.sum()); tot_pay += pay
    w = tot_w / episodes
    return w, (tot_pay / tot_w if tot_w > 1e-8 else 0.0), 1.0


# ----------------------------------------------------------------- validation phase
class ValBuffer:
    """Replay for validation: stores (s, a, ap, R, s', done); ap = recommendation."""

    def __init__(self, capacity, size, rng=None):
        self.capacity = int(capacity); self.rng = np.random.default_rng(rng)
        self.s = np.zeros((capacity, 4, size, size), dtype=np.float32)
        self.sp = np.zeros((capacity, 4, size, size), dtype=np.float32)
        self.a = np.zeros((capacity, 2), dtype=np.int64)
        self.ap = np.zeros((capacity, 2), dtype=np.int64)
        self.R = np.zeros((capacity, 2), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self._n = 0; self._i = 0

    def add(self, s, a, ap, R, sp, done):
        i = self._i
        self.s[i], self.sp[i], self.a[i], self.ap[i], self.R[i], self.done[i] = s, sp, a, ap, R, done
        self._i = (i + 1) % self.capacity; self._n = min(self._n + 1, self.capacity)

    def __len__(self): return self._n

    def sample(self, batch):
        idx = self.rng.integers(0, self._n, size=batch)
        return (self.s[idx], self.a[idx], self.ap[idx], self.R[idx], self.sp[idx], self.done[idx])


def _onehot(idx, n, device):
    oh = torch.zeros((len(idx), n), device=device)
    oh[torch.arange(len(idx)), _t(idx, device)] = 1.0
    return oh


def val_phase(trained, iters=50_000, batch=128, target_every=100, lr0=5e-4, lr1=1e-4,
              gamma=0.99, eps0=0.4, eps1=0.0, buffer_cap=100_000, prefill=5_000,
              nudge=0.0, eval_every=2_000, seed=1, device="cpu", verbose=True):
    """Train fresh selfish DQNs (ψ) against the frozen principal θ and contract-basis φ."""
    cfg = trained["cfg"]; size, horizon = cfg["size"], cfg["horizon"]
    qtheta, qphi = trained["qtheta"].eval(), trained["qphi"].eval()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    env = CoinGame(size=size, horizon=horizon, rng=rng)

    psi = ConvQNet(size, N_ACTIONS, n_extra=N_ACTIONS).to(device)     # sees recommendation one-hot
    tgt_psi = ConvQNet(size, N_ACTIONS, n_extra=N_ACTIONS).to(device)
    tgt_psi.load_state_dict(psi.state_dict())
    opt = torch.optim.Adam(psi.parameters(), lr=lr0)
    buf = ValBuffer(buffer_cap, size, rng=rng)

    @torch.no_grad()
    def recommend(obs):
        return np.array([int(qtheta(av(obs[None], i, device))[0].argmax()) for i in range(2)])

    @torch.no_grad()
    def payment(obs, ap, f):
        """Contract payment for the recommended action, per agent, using actual f_{-i}."""
        pay = np.zeros(2)
        for i in range(2):
            q = qphi(av(obs[None], i, device), _t([[float(f[1 - i])]], device))[0]
            pay[i] = float(q.max() - q[ap[i]])
        return pay

    def act(obs, ap, eps):
        a = np.empty(2, dtype=int)
        for i in range(2):
            if rng.random() < eps:
                a[i] = rng.integers(N_ACTIONS)
            else:
                with torch.no_grad():
                    a[i] = int(psi(av(obs[None], i, device), _onehot([ap[i]], N_ACTIONS, device))[0].argmax())
        return a

    # prefill with random actions (nudge: extra strict-preference margin paid on following)
    obs = env.reset()
    for _ in range(prefill):
        ap = recommend(obs); a = rng.integers(N_ACTIONS, size=2)
        f = (a == ap).astype(int)
        pay = payment(obs, ap, f) + nudge
        sp, r, done = env.step(a)
        buf.add(obs, a, ap, r + f * pay, sp, done)
        obs = sp if not done else env.reset()

    history = {"iter": [], "welfare": [], "pay_prop": [], "accuracy": []}
    obs = env.reset(); t0 = time.time()
    for it in range(iters):
        frac = it / iters
        eps = eps0 + (eps1 - eps0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        for pg in opt.param_groups: pg["lr"] = lr

        ap = recommend(obs)
        a = act(obs, ap, eps)
        f = (a == ap).astype(int)
        pay = payment(obs, ap, f) + nudge
        sp, r, done = env.step(a)
        R = r + f * pay
        buf.add(obs, a, ap, R, sp, done)
        obs = sp if not done else env.reset()

        bs, ba, bap, bR, bsp, bd = buf.sample(batch)
        done_t = _t(bd, device)
        loss_terms = []
        for i in range(2):
            vi, vpi = av(bs, i, device), av(bsp, i, device)
            ai = _t(ba[:, i], device)
            q_i = psi(vi, _onehot(bap[:, i], N_ACTIONS, device)).gather(1, ai[:, None]).squeeze(1)
            with torch.no_grad():
                ap_next = qtheta(vpi).argmax(1).cpu().numpy()
                qn = tgt_psi(vpi, _onehot(ap_next, N_ACTIONS, device)).max(1).values
                y = _t(bR[:, i], device) + gamma * (1 - done_t) * qn
            loss_terms.append(F.smooth_l1_loss(q_i, y))
        loss = sum(loss_terms)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(psi.parameters(), 10.0); opt.step()
        if (it + 1) % target_every == 0:
            tgt_psi.load_state_dict(psi.state_dict())

        if it % eval_every == 0 or it == iters - 1:
            w, pp, acc = eval_validation(qtheta, qphi, psi, size, horizon, device, rng, nudge)
            history["iter"].append(it); history["welfare"].append(w)
            history["pay_prop"].append(pp); history["accuracy"].append(acc)
            if verbose:
                print(f"  [val] it={it:6d} welfare={w:6.3f} pay_prop={pp:5.1%} "
                      f"follow_acc={acc:5.1%} eps={eps:4.2f} ({time.time()-t0:5.1f}s)", flush=True)
    return {"psi": psi, "history": history}


@torch.no_grad()
def eval_validation(qtheta, qphi, psi, size, horizon, device, rng, nudge=0.0, episodes=40):
    """Greedy eval of the validation agents: social welfare, payment proportion, follow-rate."""
    env = CoinGame(size=size, horizon=horizon, rng=rng)
    tot_w, tot_pay, follows, steps = 0.0, 0.0, 0, 0
    for _ in range(episodes):
        obs = env.reset(); done = False
        while not done:
            ap = np.array([int(qtheta(av(obs[None], i, device))[0].argmax()) for i in range(2)])
            a = np.array([int(psi(av(obs[None], i, device), _onehot([ap[i]], N_ACTIONS, device))[0].argmax())
                          for i in range(2)])
            f = (a == ap).astype(int)
            pay = 0.0
            for i in range(2):
                q = qphi(av(obs[None], i, device), _t([[float(f[1 - i])]], device))[0]
                pay += f[i] * (float(q.max() - q[ap[i]]) + nudge)
            obs, r, done = env.step(a)
            tot_w += float(r.sum()); tot_pay += pay
            follows += int(f.sum()); steps += 2
    w = tot_w / episodes
    return w, (tot_pay / tot_w if tot_w > 1e-8 else 0.0), follows / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-iters", type=int, default=50_000)
    ap.add_argument("--nudge-frac", type=float, default=0.0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = ap.parse_args()
    if args.smoke:
        args.iters, args.val_iters = 3_000, 3_000

    print(f"Coin Game (Algorithm 3)  size={args.size}x{args.size}  horizon={args.horizon}  "
          f"train_iters={args.iters}  val_iters={args.val_iters}")
    print("== TRAINING PHASE ==")
    out = train_phase(size=args.size, horizon=args.horizon, iters=args.iters, seed=args.seed)
    h = out["history"]
    print(f"-> train welfare={h['welfare'][-1]:.3f}  pay_prop={h['pay_prop'][-1]:.1%}")
    print("== VALIDATION PHASE (fresh independent DQNs) ==")
    # nudge (abs, per agent-step) = fraction * per-step social welfare (paper: ~10% of welfare)
    nudge = args.nudge_frac * h["welfare"][-1] / args.horizon
    print(f"   nudge = {args.nudge_frac:.0%} of per-step welfare = {nudge:.4f}")
    v = val_phase(out, iters=args.val_iters, nudge=nudge, seed=args.seed + 1)
    vh = v["history"]
    print(f"-> val welfare={vh['welfare'][-1]:.3f}  pay_prop={vh['pay_prop'][-1]:.1%}  "
          f"follow_acc={vh['accuracy'][-1]:.1%}")


if __name__ == "__main__":
    main()
