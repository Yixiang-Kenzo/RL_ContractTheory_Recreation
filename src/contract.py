"""LP contract solver — the principal's per-state optimization (Ivanov et al., Eq. 3).

Given, at a state ``s``:
  * the agent's TRUNCATED Q-values ``Qbar[a] = Q̄(s,a)`` (utility excluding this step's payment),
  * the outcome distributions ``O[a] = O(s,a)`` (rows sum to 1),
  * a target action ``a_p`` the principal wants to induce,

find the cheapest non-negative payment vector ``b`` (limited liability) that makes ``a_p`` a
best response for the agent.

    minimize_b    E_{o~O(s,a_p)}[b(o)]                       # expected payment
    subject to    Q̄(s,a_p) + E_{O(s,a_p)}[b]  >=  Q̄(s,a') + E_{O(s,a')}[b]   for all a' != a_p
                  b(o) >= 0                                   for all outcomes o

The agent compares FULL utilities Q*(s,a) = Q̄(s,a) + E_{O(s,a)}[b], so the constraints are
exactly "the agent (weakly) prefers a_p to every alternative."
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def solve_contract(Qbar, O, a_p, tol=1e-9):
    """Cheapest limited-liability contract inducing action ``a_p``.

    Parameters
    ----------
    Qbar : array (n_actions,)   truncated Q-values Q̄(s,a).
    O    : array (n_actions, m) outcome distributions; row a is O(s,a).
    a_p  : int                  target (recommended) action.

    Returns
    -------
    b : ndarray (m,) or None
        Optimal payment vector, or None if inducing a_p is infeasible.
    expected_payment : float
        E_{o~O(s,a_p)}[b(o)] at the optimum (inf if infeasible).
    """
    Qbar = np.asarray(Qbar, dtype=float)
    O = np.asarray(O, dtype=float)
    n_actions, m = O.shape

    c = O[a_p]  # objective: minimize expected payment under the recommended action

    # IC constraints, one per alternative action a' != a_p, in scipy's A_ub @ b <= b_ub form:
    #   (O[a'] - O[a_p]) @ b  <=  Q̄(a_p) - Q̄(a')
    rows, rhs = [], []
    for ap in range(n_actions):
        if ap == a_p:
            continue
        rows.append(O[ap] - O[a_p])
        rhs.append(Qbar[a_p] - Qbar[ap])

    if rows:
        A_ub, b_ub = np.array(rows), np.array(rhs)
    else:  # single-action MDP: no constraints, cheapest contract is zero
        A_ub, b_ub = None, None

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * m, method="highs")
    if not res.success:
        return None, np.inf

    b = np.clip(res.x, 0.0, None)
    b[b < tol] = 0.0
    return b, float(O[a_p] @ b)


def binary_contract(Qbar, a_p, p_o1, nudge=0.0):
    """Closed-form solution of solve_contract for the 2-action / 2-outcome tree.

    Vectorized over a batch. For the binary case the LP is analytic:
      * To induce the effort action a1 (which favors the good outcome o1): pay ONLY on o1,
        b = [0, max(Q̄(a0) - Q̄(a1) + ξ, 0) / (p1 - p0)].
      * To induce the free action a0: pay ONLY on o0 by the symmetric amount.
    where p0 = P(o1 | a0), p1 = P(o1 | a1).  Equivalent to the scipy LP (unit-tested), but
    ~1000x faster -- Algorithm 2 solves millions of these.

    ``nudge`` (ξ >= 0) is the paper's nudging margin (Appendix B.9.2): it is added to the RHS
    of the IC constraint so the agent must *strictly* prefer a_p by ξ, making the contract
    robust to under-estimation of the required payment (the discontinuity). ξ=0 recovers the
    plain minimal contract.

    Parameters
    ----------
    Qbar : array (B, 2)   truncated Q-values [Q̄(s,a0), Q̄(s,a1)] per sample.
    a_p  : array (B,) int recommended action (0 or 1) per sample.
    p_o1 : array (2,)     [P(o1|a0), P(o1|a1)] (constant across states in this env).
    nudge : float         strict-preference margin ξ >= 0.

    Returns
    -------
    b : ndarray (B, 2)    payment [b(o0), b(o1)] per sample.
    """
    Qbar = np.asarray(Qbar, dtype=float)
    a_p = np.asarray(a_p)
    denom = float(p_o1[1] - p_o1[0])              # p1 - p0 > 0
    dQ = Qbar[:, 0] - Qbar[:, 1]                  # Q̄(a0) - Q̄(a1)
    b = np.zeros_like(Qbar)
    m1 = a_p == 1                                 # induce effort: pay on good outcome o1
    b[m1, 1] = np.maximum(dQ[m1] + nudge, 0.0) / denom
    m0 = a_p == 0                                 # induce free action: pay on bad outcome o0
    b[m0, 0] = np.maximum(-dQ[m0] + nudge, 0.0) / denom
    return b


def agent_best_response(Qbar, O, b):
    """Action the agent takes under contract ``b``: argmax_a [ Q̄(s,a) + E_{O(s,a)}[b] ]."""
    Qbar = np.asarray(Qbar, dtype=float)
    O = np.asarray(O, dtype=float)
    full = Qbar + O @ np.asarray(b, dtype=float)
    return int(np.argmax(full)), full
