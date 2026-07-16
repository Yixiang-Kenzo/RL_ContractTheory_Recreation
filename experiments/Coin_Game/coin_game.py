"""The Coin Game — a 2-player sequential social dilemma (Foerster et al. 2018).

As used by Ivanov et al. (2024), with their modifications (see notes/coin_game_spec.md):
  * +1 for collecting a coin of your OWN color, +0.2 for the OTHER color.
  * NO penalty to a player whose coin is taken by the other (paper removes it).

Social dilemma: welfare is maximized when each agent takes only its own-color coins (the
correct-color collector gets +1 vs +0.2 for the wrong one). But each agent is individually
tempted to grab any reachable coin (+0.2 > 0), which destroys welfare -> the dilemma.

State: 4-channel grid tensor [red-agent, blue-agent, red-coin, blue-coin] (one coin active).
Actions per agent: 0=up 1=down 2=left 3=right, on a toroidal grid.
"""

from __future__ import annotations

import numpy as np

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right
N_ACTIONS = 4
RED, BLUE = 0, 1


class CoinGame:
    def __init__(self, size=3, horizon=20, rng=None):
        self.size = int(size)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(rng)
        self.n_agents = 2
        self.reset()

    # ------------------------------------------------------------------ dynamics
    def reset(self):
        s = self.size
        # distinct random agent positions
        cells = self.rng.choice(s * s, size=2, replace=False)
        self.agent_pos = np.array([[cells[0] // s, cells[0] % s],
                                   [cells[1] // s, cells[1] % s]])
        self._spawn_coin()
        self.t = 0
        return self.observe()

    def _spawn_coin(self):
        s = self.size
        c = self.rng.integers(s * s)
        self.coin_pos = np.array([c // s, c % s])
        self.coin_color = int(self.rng.integers(2))

    def step(self, actions):
        """actions: length-2 array of ints in {0..3}. Returns (obs, rewards[2], done)."""
        s = self.size
        self.agent_pos = (self.agent_pos + MOVES[np.asarray(actions)]) % s
        rewards = np.zeros(2, dtype=np.float32)

        on_coin = [np.array_equal(self.agent_pos[i], self.coin_pos) for i in range(2)]
        collectors = [i for i in range(2) if on_coin[i]]
        if collectors:
            # if both land on the coin, break the tie at random
            winner = collectors[0] if len(collectors) == 1 else int(self.rng.choice(collectors))
            rewards[winner] = 1.0 if winner == self.coin_color else 0.2
            self._spawn_coin()

        self.t += 1
        return self.observe(), rewards, self.t >= self.horizon

    # ------------------------------------------------------------------ encoding
    def observe(self):
        """4-channel (C, H, W) float tensor: red-agent, blue-agent, red-coin, blue-coin."""
        s = self.size
        obs = np.zeros((4, s, s), dtype=np.float32)
        obs[RED, self.agent_pos[RED, 0], self.agent_pos[RED, 1]] = 1.0
        obs[BLUE, self.agent_pos[BLUE, 0], self.agent_pos[BLUE, 1]] = 1.0
        obs[2 + self.coin_color, self.coin_pos[0], self.coin_pos[1]] = 1.0
        return obs


def social_welfare_of_random(size=3, horizon=20, episodes=200, seed=0):
    """Baseline: expected per-episode social welfare of uniformly random agents."""
    env = CoinGame(size=size, horizon=horizon, rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    tot = 0.0
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, r, done = env.step(rng.integers(N_ACTIONS, size=2))
            tot += r.sum()
    return tot / episodes


if __name__ == "__main__":
    for s, h in [(3, 20), (7, 50)]:
        w = social_welfare_of_random(size=s, horizon=h)
        print(f"CoinGame {s}x{s}, horizon {h}: random-agent social welfare/episode = {w:.3f}")
