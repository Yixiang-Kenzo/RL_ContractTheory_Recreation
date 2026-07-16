"""Random binary game-tree MDP — Experiment 1 environment.

Reproduces the "randomly generated binary game-trees" testbed from Ivanov et al. (2024),
"Principal-Agent Reinforcement Learning" (arXiv 2407.18074).

Model (see notes/decision_log.md for the choices behind it)
-----------------------------------------------------------
* A complete binary tree of depth ``T``. Nodes are indexed 0..N-1 in breadth-first order:
  node ``i`` has children ``2i+1`` (outcome 0) and ``2i+2`` (outcome 1). Internal nodes
  (depth < T) are decision states; leaves (depth T) are terminal with value 0.
* At each internal node the agent chooses an action ``a in {0, ..., n_actions-1}``.
* Taking action ``a`` draws an OUTCOME ``o in {0, 1}`` from ``O(s, a)`` (``outcome_probs``).
  The outcome selects which child we move to AND is what the contract pays on.
* Agent reward for acting:   ``r(s, a)``      (``agent_reward``; typically <= 0, an effort cost)
* Principal reward:          ``rp(s, o)``     (``principal_reward``; depends on the OUTCOME)

The contract ``b : {0,1} -> R>=0`` pays the agent based on the realized outcome ``o``.
Agent's per-step utility:      r(s, a) + b(o)
Principal's per-step utility:  rp(s, o) - b(o)

This class is a pure data container + simulator. The exact optimum lives in ``exact.py``
and the LP contract solver in ``contract.py``.
"""

from __future__ import annotations

import numpy as np


class TreeMDP:
    """A single randomly generated binary game-tree instance."""

    def __init__(self, depth=10, gamma=1.0, rng=None, success_prob=0.9):
        """Paper-faithful binary game-tree (Ivanov et al. 2024, Appendix D.1).

        The generative model is fixed by the paper; there are no free "calibration" knobs
        here anymore (see notes/decision_log.md, entry D-6):

        * Two actions a0 (low effort), a1 (high effort). Two outcomes o0 (bad), o1 (good).
        * a0 -> o0 with prob ``success_prob`` (0.9); a1 -> o1 with prob ``success_prob``.
          So P(o1 | a0) = 1 - 0.9 = 0.1  and  P(o1 | a1) = 0.9, in ALL states.
        * a0 is costless: r(s,a0) = 0.  o0 gives the principal nothing: rp(s,o0) = 0.
        * a1 cost and o1 reward are randomly sampled with a nested-uniform scheme that makes
          the principal's reward higher on average than the agent's cost:
              r(s,a1)  = -u,  v ~ U[0,1], u ~ U[0, 1 - v]
              rp(s,o1) =  u,  v ~ U[0,2], u ~ U[0, 2 - v]
          Under this scheme the SPE incentivizes a1 in ~60% of states (a paper-stated fact
          we use to validate our exact solver).

        Parameters
        ----------
        depth : int          tree depth T (paper: 10, giving 1023 states).
        gamma : float        discount (paper: 1.0, no discounting).
        success_prob : float P(intended outcome | action) (paper: 0.9).
        """
        self.depth = int(depth)
        self.n_actions = 2
        self.gamma = float(gamma)
        self.n_outcomes = 2
        self.success_prob = float(success_prob)
        self.rng = np.random.default_rng(rng)

        # Node bookkeeping (breadth-first complete binary tree).
        self.n_nodes = 2 ** (self.depth + 1) - 1
        self.n_internal = 2 ** self.depth - 1          # nodes at depth < T
        self.first_leaf = self.n_internal              # nodes >= this index are leaves

        self._build()

    # ------------------------------------------------------------------ building
    def _build(self):
        rng = self.rng
        S = self.n_internal
        p = self.success_prob

        # Agent action rewards r(s,a): a0 free; a1 a randomly sampled negative cost.
        # r(s,a1) = -u,  v ~ U[0,1], u ~ U[0, 1 - v].
        v = rng.uniform(0.0, 1.0, size=S)
        u = rng.uniform(0.0, 1.0 - v)
        self.agent_reward = np.zeros((S, 2))
        self.agent_reward[:, 1] = -u

        # Principal outcome rewards rp(s,o): o0 gives 0; o1 a randomly sampled reward.
        # rp(s,o1) = u,  v ~ U[0,2], u ~ U[0, 2 - v].
        v2 = rng.uniform(0.0, 2.0, size=S)
        u2 = rng.uniform(0.0, 2.0 - v2)
        self.principal_reward = np.zeros((S, 2))
        self.principal_reward[:, 1] = u2

        # Outcome distribution O(s,a): P(outcome = 1 | s, a), FIXED across states.
        #   a0 -> o0 w.p. p  =>  P(o1 | a0) = 1 - p
        #   a1 -> o1 w.p. p  =>  P(o1 | a1) = p
        self.p_outcome1 = np.tile([1.0 - p, p], (S, 1))

    # ------------------------------------------------------------------ topology
    def is_leaf(self, s):
        return s >= self.first_leaf

    def child(self, s, outcome):
        """Next node when OUTCOME ``outcome in {0,1}`` occurs at internal node ``s``."""
        return 2 * s + 1 + int(outcome)

    def depth_of(self, s):
        return int(np.floor(np.log2(s + 1)))

    def internal_nodes(self):
        """Iterator over decision states, deepest first (useful for backward induction)."""
        return range(self.n_internal - 1, -1, -1)

    # ------------------------------------------------------------------ dynamics
    def outcome_probs(self, s, a):
        """O(s,a) as a length-2 vector [P(o=0), P(o=1)]."""
        p1 = self.p_outcome1[s, a]
        return np.array([1.0 - p1, p1])

    def expected_payment(self, s, a, b):
        """E_{o~O(s,a)}[b(o)] for a length-2 contract vector ``b``."""
        return float(self.outcome_probs(s, a) @ np.asarray(b, dtype=float))

    def expected_principal_reward(self, s, a):
        """E_{o~O(s,a)}[rp(s,o)] — principal's expected outcome reward before payment."""
        return float(self.outcome_probs(s, a) @ self.principal_reward[s])


def sample_tree(depth, seed, **kwargs):
    """Convenience: build one tree with an integer seed."""
    return TreeMDP(depth=depth, rng=np.random.default_rng(seed), **kwargs)
