"""Does perturbation-augmented training actually help the model learn?

Runs the learning experiment in se/rsi_experiment.py and reports held-out
accuracy, the reward-hacking signature (score variance under perturbation), and
where each trained model put its weight.

Usage: python scripts/04_rsi_training_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.rsi_experiment import CHANNEL_NAMES, run_experiment


def bar(frac: float, width: int = 24) -> str:
    n = int(round(frac * width))
    return "#" * n + "." * (width - n)


def main() -> None:
    base, skep, (wb, ws) = run_experiment(n_tasks=600, m_perturb=10, corr=0.9, seed=0)

    print("=" * 78)
    print("RSI TRAINING EXPERIMENT — plain baseline vs perturbation-augmented")
    print("=" * 78)
    print("Ground truth: a fix is correct iff it addresses the bug signal.")
    print("The other 3 channels (ps phrasing, README, test order) are incidental.\n")

    print(f"{'model':<38}{'held-out acc':>13}{'reward-hack var':>17}")
    print("-" * 78)
    for r in (base, skep):
        print(f"{r.name:<38}{r.accuracy*100:>11.1f}%{r.mean_task_variance:>17.3f}")

    print("\nWhere did each model put its weight? (fraction of |weight| mass)")
    print("-" * 78)
    for r in (base, skep):
        print(f"  {r.name}")
        print(f"    on true bug signal : {bar(r.weight_on_bug)}  {r.weight_on_bug*100:4.1f}%")
        print(f"    on incidental chans: {bar(r.weight_on_spurious)}  {r.weight_on_spurious*100:4.1f}%")

    print("\nLearned weights per channel:")
    print(f"    {'channel':<14}{'plain':>10}{'skeptical':>12}")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"    {name:<14}{wb[i]:>10.2f}{ws[i]:>12.2f}")

    print("\n" + "=" * 78)
    print("TAKEAWAY")
    print("=" * 78)
    dv = base.mean_task_variance - skep.mean_task_variance
    da = (skep.accuracy - base.accuracy) * 100
    print(f"The plain baseline learned to lean on the incidental channels "
          f"({base.weight_on_spurious*100:.0f}% of its weight) —")
    print("a reward hack that looks fine on the canonical config it trained on. Under held-out")
    print(f"perturbations it swings ({base.mean_task_variance:.3f} variance) and loses accuracy.")
    print(f"Training on perturbations moved weight onto the real bug signal "
          f"({skep.weight_on_bug*100:.0f}%), cutting")
    print(f"variance by {dv:.3f} and raising held-out accuracy by {da:.1f} points.")
    print("\nSo yes: the small perturbations actually helped the model learn the right thing —")
    print("and the leftover variance is exactly what the Skeptical Evaluator's gate keys on.")


if __name__ == "__main__":
    main()
