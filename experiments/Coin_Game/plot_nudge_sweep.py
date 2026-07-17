"""Plot the nudge sweep: follow-accuracy and payment proportion vs nudge (3x3 Coin Game).

Data captured from investigate_nudge.py (a single trained principal, validation swept over
nudge levels). Shows the follow-rate is nudge-controlled and reaches the paper's 80-90% band,
at a rising payment cost -- the accuracy/efficiency tradeoff.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

# nudge_frac, welfare, pay_prop(%), follow_acc(%)   -- from investigate_nudge.py
DATA = np.array([
    [0.0, 11.06, 13.6, 62.5],
    [0.1, 11.23, 33.7, 72.6],
    [0.2, 11.48, 58.9, 85.3],
    [0.3, 11.95, 82.9, 93.2],
])


def plot(data):
    nf, w, pp, acc = data.T
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(nf, acc, "o-", color="blue", label="follow accuracy (%)")
    ax.plot(nf, pp, "s--", color="orange", label="payment (% of welfare)")
    ax.axhspan(80, 90, color="green", alpha=0.12, label="paper accuracy band (80-90%)")
    ax.set_xlabel("nudge (fraction of per-step social welfare)")
    ax.set_ylabel("%")
    ax.set_title("Coin Game 3x3 — follow-accuracy & payment vs nudge")
    ax.grid(alpha=0.25); ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(HERE, "results", "nudge_sweep.png")
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    plot(DATA)
