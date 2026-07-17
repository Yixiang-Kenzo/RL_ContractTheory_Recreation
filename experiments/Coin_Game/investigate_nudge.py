"""Diagnostic: why is the follow-accuracy ~78%? Sweep the nudge margin.

Train the principal ONCE, then run the validation phase at several nudge levels. If the
follow-rate climbs toward the paper's 80-90% as the nudge grows (at the cost of higher
payment), the ~78% is a nudge-magnitude effect, not a deeper failure.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

from train_coin import train_phase, val_phase       # noqa: E402


def main():
    size, horizon = 3, 20
    train_iters, val_iters = 40_000, 30_000
    print("== training principal once ==", flush=True)
    trained = train_phase(size=size, horizon=horizon, iters=train_iters, seed=0, verbose=False)
    w_train = trained["history"]["welfare"][-1]
    per_step_w = w_train / horizon
    print(f"trained: welfare={w_train:.2f}  per_step_welfare={per_step_w:.3f}\n", flush=True)

    print(f"{'nudge_frac':>10} {'nudge_abs':>9} | {'welfare':>7} {'pay_prop':>8} {'follow_acc':>10}")
    print("-" * 52)
    results = []
    for nf in [0.0, 0.1, 0.2, 0.3, 0.4]:
        nudge = nf * per_step_w
        v = val_phase(trained, iters=val_iters, nudge=nudge, seed=1, verbose=False)
        h = v["history"]
        # average the last 3 evals to reduce single-snapshot noise
        acc = sum(h["accuracy"][-3:]) / len(h["accuracy"][-3:])
        w = sum(h["welfare"][-3:]) / len(h["welfare"][-3:])
        pp = sum(h["pay_prop"][-3:]) / len(h["pay_prop"][-3:])
        results.append((nf, nudge, w, pp, acc))
        print(f"{nf:>10.1f} {nudge:>9.4f} | {w:>7.2f} {pp:>8.1%} {acc:>10.1%}", flush=True)

    print("\nInterpretation: if follow_acc rises toward 80-90% with larger nudge, the ~78% is a")
    print("nudge-magnitude effect (knob), and we pick the smallest nudge that reaches the target.")


if __name__ == "__main__":
    main()
