"""Recursive self-improvement: naive Yes/No baseline vs. the perturbation evaluator.

Part 1 — run the perturbation agent on a real SWE-bench sample, score a candidate
         across every perturbation axis, and build the exact feedback message that
         gets sent back to the code-fix agent (a SUMMARY of the fix + the per-axis
         data, not the whole patch).
Part 2 — run both RSI loops for 6 rounds and plot accuracy per round. The naive loop
         reward-hacks and plateaus; the skeptical loop keeps improving.

Usage: python scripts/06_rsi_loops.py [instance_id]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.evaluator import SkepticalGate, evaluate_skeptically, safe_sites_for
from se.fixers import value_hardcoder
from se.perturbations import PerturbationAgent
from se.rsi_loop import build_feedback_message, make_tasks, run_loop
from se.swebench_data import get_instance

ROUNDS = 6
N_TASKS = 40
OUT = Path(__file__).resolve().parent.parent / "out" / "rsi_accuracy.png"


def part1_real_feedback(instance_id: str) -> None:
    print("=" * 88)
    print("1) RUN THE PERTURBATION AGENT ON A REAL SWE-BENCH SAMPLE -> BUILD FEEDBACK")
    print("=" * 88)
    inst = get_instance(instance_id)
    agent = PerturbationAgent()
    sites = safe_sites_for(inst, agent)
    gate = SkepticalGate(bar=80.0, tau=8.0)

    # A candidate partway through training that hardcoded the repro values.
    fixer = value_hardcoder()
    result = evaluate_skeptically(fixer, inst, sites, gate, seed=7)
    summary = ("special-cased the reproduction's exact input values "
               "(if inputs == (10, 5): return <memorized matrix>)")
    print()
    print(build_feedback_message(inst, summary, result))
    print("\nThis whole block — the fix summary plus the per-axis scores — is what goes")
    print("back to the agent next round. The full patch is NOT sent, only the summary.")


def part2_loops() -> None:
    print("\n" + "=" * 88)
    print(f"2) RECURSIVE SELF-IMPROVEMENT OVER {N_TASKS} TASKS, {ROUNDS} ROUNDS")
    print("=" * 88)
    tasks = make_tasks(N_TASKS, hackable_frac=0.7, seed=1)  # same tasks for both regimes
    naive = run_loop("naive", tasks, ROUNDS)
    skeptical = run_loop("skeptical", tasks, ROUNDS)

    print(f"\n{'round':<7}{'naive: believed':>17}{'naive: TRUE':>14}{'skeptical: TRUE':>18}")
    print("-" * 56)
    for r in range(ROUNDS + 1):
        print(f"{r:<7}{naive['believed'][r]:>16.0f}%{naive['true'][r]:>13.0f}%"
              f"{skeptical['true'][r]:>17.0f}%")

    print("\nThe naive loop's own evaluator believes it hit "
          f"{naive['believed'][-1]:.0f}% — but only {naive['true'][-1]:.0f}% of those")
    print("solutions actually generalize (the rest are reward hacks it froze on). The")
    print(f"skeptical loop reaches {skeptical['true'][-1]:.0f}% truly-resolved, because it never")
    print("freezes on a high-variance fix and its feedback points at the real weakness.")

    _plot(naive, skeptical)


def _plot(naive: dict, skeptical: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed — skipping plot; pip install matplotlib)")
        return

    rounds = list(range(ROUNDS + 1))
    ink, muted, grid = "#1A2420", "#77826F", "#D2D8CB"
    c_skep, c_naive, c_believe = "#1F8F58", "#A23B2E", "#8A8F98"

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=140)
    ax.plot(rounds, skeptical["true"], "-o", color=c_skep, lw=2.4, ms=6,
            label="Skeptical loop — truly resolved")
    ax.plot(rounds, naive["true"], "-s", color=c_naive, lw=2.4, ms=6,
            label="Naive loop — truly resolved")
    ax.plot(rounds, naive["believed"], "--^", color=c_believe, lw=1.8, ms=5,
            label="Naive loop — what its Yes/No check believes")

    ax.set_title("Recursive self-improvement: perturbation evaluator vs. naive Yes/No",
                 color=ink, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("RSI round", color=ink, fontsize=11)
    ax.set_ylabel("Accuracy (% of tasks truly resolved)", color=ink, fontsize=11)
    ax.set_ylim(-3, 103)
    ax.set_xticks(rounds)
    ax.grid(True, color=grid, lw=0.8, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)

    # emphasize endpoints
    for series, color in ((skeptical["true"], c_skep), (naive["true"], c_naive)):
        ax.annotate(f"{series[-1]:.0f}%", (rounds[-1], series[-1]),
                    textcoords="offset points", xytext=(8, 0), color=color,
                    fontsize=10, fontweight="bold", va="center")

    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    fig.text(0.5, 0.005,
             "Reward hacking is emergent: the same greedy agent hacks under the naive reward "
             "and generalizes under the skeptical one. Scores modeled (no Docker); channels are real.",
             ha="center", color=muted, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"\nPlot saved -> {OUT}")


def main(instance_id: str = "astropy__astropy-12907") -> None:
    part1_real_feedback(instance_id)
    part2_loops()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "astropy__astropy-12907")
