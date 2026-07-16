"""Small DQN building blocks for Algorithm 2 (Tree MDP experiment).

Two identical-architecture networks are used: the principal's contractual Q-network qθ(s,a)
and the agent's truncated Q-network Q̄φ(s,a). Both take a one-hot state and output a Q-value
per action (paper: 2 hidden fully-connected layers of 256, ReLU).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class QNet(nn.Module):
    """MLP Q-network: one-hot state -> Q-value per action."""

    def __init__(self, n_states, n_actions, hidden=256):
        super().__init__()
        self.n_states = n_states
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, onehot):
        return self.net(onehot)

    def q_all(self, states, device=None):
        """Q-values for a batch of integer states -> tensor (B, n_actions).

        Terminal (leaf) states have index >= n_states; their Q-values are always masked out
        by (1 - done) at the call site, so we clamp the index into range for the one-hot.
        """
        idx = torch.as_tensor(states, dtype=torch.long).clamp_(0, self.n_states - 1)
        oh = torch.zeros((len(idx), self.n_states), device=device or next(self.parameters()).device)
        oh[torch.arange(len(idx)), idx] = 1.0
        return self.forward(oh)


class ReplayBuffer:
    """Fixed-size ring buffer of transitions (s, a_p, r_a, r_p, o, done, s')."""

    def __init__(self, capacity, rng=None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(rng)
        self.s = np.zeros(capacity, dtype=np.int64)
        self.ap = np.zeros(capacity, dtype=np.int64)
        self.ra = np.zeros(capacity, dtype=np.float32)
        self.rp = np.zeros(capacity, dtype=np.float32)
        self.o = np.zeros(capacity, dtype=np.int64)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.sp = np.zeros(capacity, dtype=np.int64)
        self._n = 0
        self._i = 0

    def add(self, s, ap, ra, rp, o, done, sp):
        i = self._i
        self.s[i], self.ap[i], self.ra[i] = s, ap, ra
        self.rp[i], self.o[i], self.done[i], self.sp[i] = rp, o, done, sp
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def __len__(self):
        return self._n

    def sample(self, batch_size):
        idx = self.rng.integers(0, self._n, size=batch_size)
        return (self.s[idx], self.ap[idx], self.ra[idx], self.rp[idx],
                self.o[idx], self.done[idx], self.sp[idx])
