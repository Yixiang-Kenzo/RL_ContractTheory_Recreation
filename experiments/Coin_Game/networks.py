"""Conv Q-networks + replay buffer for the Coin Game (Algorithm 3).

Networks follow Appendix D.2: 1 conv layer (4x4 kernels, 4->32 channels) + 2 FC layers (64),
ReLU. Extra scalar features (the other agent's follow-bit f_{-i}, or a one-hot recommended
action at validation) are concatenated to the flattened conv features before the FC stack.

Agents share parameters and act from an *agent-centric* view of the 4-channel state
(my-agent, other-agent, my-coin, other-coin), which makes the two symmetric players share
one network. See agent_view() below.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def agent_view(obs, i):
    """Agent-centric 4-channel view for agent i (0=red,1=blue).

    obs channels are [red-agent, blue-agent, red-coin, blue-coin]; we reorder to
    [my-agent, other-agent, my-coin, other-coin]. Batched: obs is (B,4,H,W)."""
    if i == 0:
        return obs
    idx = [1, 0, 3, 2]
    return obs[..., idx, :, :]


class ConvQNet(nn.Module):
    """4x4 conv (4->32) + 2 FC(64) + output; optional extra scalar inputs."""

    def __init__(self, size, n_actions, n_extra=0, channels=32, hidden=64):
        super().__init__()
        self.conv = nn.Conv2d(4, channels, kernel_size=4, padding=2)
        conv_out = channels * (size + 1) * (size + 1)     # padding=2, k=4 -> H+1
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(conv_out + n_extra, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        self.n_extra = n_extra

    def forward(self, obs, extra=None):
        h = self.conv(obs).flatten(1)
        if self.n_extra:
            h = torch.cat([h, extra], dim=1)
        return self.fc(h)


class ReplayBuffer:
    """Uniform replay of Coin Game transitions (s, a[2], f[2], r[2], s', done).

    NOTE: the paper uses prioritized replay (PER); we start with uniform for a working
    baseline and can upgrade later (logged as a decision)."""

    def __init__(self, capacity, size, rng=None):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(rng)
        self.s = np.zeros((capacity, 4, size, size), dtype=np.float32)
        self.sp = np.zeros((capacity, 4, size, size), dtype=np.float32)
        self.a = np.zeros((capacity, 2), dtype=np.int64)
        self.f = np.zeros((capacity, 2), dtype=np.int64)
        self.r = np.zeros((capacity, 2), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self._n = 0
        self._i = 0

    def add(self, s, a, f, r, sp, done):
        i = self._i
        self.s[i], self.sp[i] = s, sp
        self.a[i], self.f[i], self.r[i], self.done[i] = a, f, r, done
        self._i = (i + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def __len__(self):
        return self._n

    def sample(self, batch):
        idx = self.rng.integers(0, self._n, size=batch)
        return (self.s[idx], self.a[idx], self.f[idx], self.r[idx], self.sp[idx], self.done[idx])
