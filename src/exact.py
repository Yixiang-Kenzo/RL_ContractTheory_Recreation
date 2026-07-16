"""Exact subgame-perfect solution of a TreeMDP by backward induction.

Because the environment is a finite tree, we can compute the *exact* optimum the learning
algorithm should approach. At each internal node, working from the leaves up, we:

  1. form the agent's truncated Q-values  Q̄(s,a) = r(s,a) + γ Σ_o O(s,a)[o] · V_agent(child),
  2. for every candidate recommended action a_p, solve the LP (contract.solve_contract) for
     the cheapest contract that makes a_p incentive-compatible,
  3. let the principal pick the a_p that maximizes ITS value
        V_principal(s | a_p) = E_{O(s,a_p)}[ rp(s,o) - b(o) + γ · V_principal(child) ],
  4. store that as the node's optimum and propagate V_agent, V_principal upward.

This is the subgame-perfect equilibrium (SPE): at every node both parties best-respond.
"""

from __future__ import annotations

import numpy as np

from contract import solve_contract


class ExactSolution:
    """Container for the per-node exact optimum of one TreeMDP."""

    def __init__(self, mdp):
        S = mdp.n_internal
        m = mdp.n_outcomes
        self.mdp = mdp
        self.Qbar = np.zeros((S, mdp.n_actions))     # agent truncated Q per (state, action)
        self.opt_action = np.full(S, -1, dtype=int)  # principal's recommended action a_p*(s)
        self.contract = np.zeros((S, m))             # optimal contract b*(s)
        self.V_agent = np.zeros(mdp.n_nodes)         # agent value per node (0 at leaves)
        self.V_principal = np.zeros(mdp.n_nodes)     # principal value per node (0 at leaves)
        self.payment = np.zeros(S)                   # expected payment at each state
        self._solve()

    def _solve(self):
        mdp = self.mdp
        for s in mdp.internal_nodes():               # deepest internal node first
            # --- 1. agent's truncated Q-values, using already-solved children ---
            for a in range(mdp.n_actions):
                O = mdp.outcome_probs(s, a)
                cont = O[0] * self.V_agent[mdp.child(s, 0)] + O[1] * self.V_agent[mdp.child(s, 1)]
                self.Qbar[s, a] = mdp.agent_reward[s, a] + mdp.gamma * cont

            O_all = np.stack([mdp.outcome_probs(s, a) for a in range(mdp.n_actions)])

            # --- 2 & 3. principal tries each recommended action, keeps the best ---
            best_val, best = -np.inf, None
            for a_p in range(mdp.n_actions):
                b, pay = solve_contract(self.Qbar[s], O_all, a_p)
                if b is None:
                    continue
                O = O_all[a_p]
                cont_p = O[0] * self.V_principal[mdp.child(s, 0)] + O[1] * self.V_principal[mdp.child(s, 1)]
                v_princ = mdp.expected_principal_reward(s, a_p) - pay + mdp.gamma * cont_p
                if v_princ > best_val:
                    v_agent = self.Qbar[s, a_p] + pay          # agent's full utility at s
                    best_val, best = v_princ, (a_p, b, pay, v_agent)

            # --- 4. store and propagate ---
            a_p, b, pay, v_agent = best
            self.opt_action[s] = a_p
            self.contract[s] = b
            self.payment[s] = pay
            self.V_principal[s] = best_val
            self.V_agent[s] = v_agent

    @property
    def principal_value(self):
        """Optimal principal utility at the root."""
        return float(self.V_principal[0])


if __name__ == "__main__":
    # Smoke test: solve a few random trees and print the root optimum.
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from tree_mdp import sample_tree

    for seed in range(3):
        mdp = sample_tree(depth=4, seed=seed)
        sol = ExactSolution(mdp)
        print(f"seed={seed}  depth={mdp.depth}  internal_nodes={mdp.n_internal}  "
              f"principal_value={sol.principal_value:.4f}  "
              f"mean_payment={sol.payment.mean():.4f}  "
              f"actions={np.bincount(sol.opt_action, minlength=mdp.n_actions)}")
