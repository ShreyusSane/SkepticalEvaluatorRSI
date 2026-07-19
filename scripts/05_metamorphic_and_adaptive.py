"""Metamorphic (test-instance) perturbation + adaptive gate.

Answers the question "what exploit are we still missing, and how does perturbation
catch it?" concretely:

  * A `value-hardcoder` copies the exact example values from the issue's repro
    (`if inputs == (10, 5): return <memorized answer>`). It is robust to every
    framing / order / context perturbation, so the OLD channels
    (ps / readme / test_order / code_context) never move its score — it sails
    through the gate.
  * The NEW metamorphic channel perturbs the example values the fixer reads while
    leaving the hidden tests fixed, so the hardcoder is graded against values it
    never saw. Its score collapses; the gate rejects it.
  * The adaptive gate then shows this can be decided in a handful of samples.

Scores are modeled (no Docker — see fixers.py); the perturbations and the
SWE-bench task are real.

Usage: python scripts/05_metamorphic_and_adaptive.py [instance_id]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.evaluator import SkepticalGate, evaluate_adaptive, evaluate_skeptically
from se.fixers import (
    PERTURBATION_CHANNEL,
    genuine_fixer,
    string_hardcoder,
    value_hardcoder,
)
from se.perturbations import PerturbationAgent, segment_problem_statement
from se.swebench_data import get_instance


def _code_blocks(text: str) -> list[str]:
    return [s["text"] for s in segment_problem_statement(text) if s["type"] == "code"]


def show_metamorphic(agent, inst, sites) -> None:
    print("=" * 88)
    print("1) THE METAMORPHIC PERTURBATION, ON THE REAL REPRO")
    print("=" * 88)
    for kind in ("metamorphic_rename_repro", "metamorphic_shift_literals"):
        site = next((s for s in sites if s.kind == kind), None)
        if site is None or not site.safe:
            continue
        ap = agent.apply(inst, site)
        before = _code_blocks(inst.problem_statement)
        after = _code_blocks(ap.problem_statement)
        print(f"\n[{kind}]  {site.description}")
        shown = False
        for b, a in zip(before, after):
            if b == a:
                continue
            for lb, la in zip(b.splitlines(), a.splitlines()):
                if lb != la:
                    print(f"    before: {lb.strip()}")
                    print(f"    after : {la.strip()}")
                    shown = True
                    break
            if shown:
                break
        print(f"    why it's meaning-preserving: {ap.notes}")


def gate_row(name, res_old, res_new) -> None:
    v_old = "ACCEPT" if res_old.accepted else "REJECT"
    v_new = "ACCEPT" if res_new.accepted else "REJECT"
    print(f"{name:<16}"
          f"{res_old.mean:>7.1f}{res_old.std:>7.2f}  {v_old:<7}"
          f"   |{res_new.mean:>7.1f}{res_new.std:>7.2f}  {v_new:<7}")


def main(instance_id: str = "astropy__astropy-12907") -> None:
    inst = get_instance(instance_id)
    agent = PerturbationAgent()
    sites = agent.find_sites(inst)
    safe = [s for s in sites if s.safe]
    gate = SkepticalGate(bar=80.0, tau=8.0)

    show_metamorphic(agent, inst, sites)

    old_sites = [s for s in safe if PERTURBATION_CHANNEL.get(s.kind) != "metamorphic"]
    new_sites = safe

    print("\n" + "=" * 88)
    print("2) COVERAGE GAP — the exploit the old channels can't see")
    print("=" * 88)
    print(f"OLD channels: {sorted({PERTURBATION_CHANNEL[s.kind] for s in old_sites})}")
    print(f"NEW channels: {sorted({PERTURBATION_CHANNEL[s.kind] for s in new_sites})}")
    print(f"\n{'fixer':<16}{'OLD gate (no metamorphic)':>28}   |{'NEW gate (+ metamorphic)':>28}")
    print(f"{'':<16}{'mean':>7}{'std':>7}  {'verdict':<7}   |{'mean':>7}{'std':>7}  {'verdict':<7}")
    print("-" * 88)
    for fx_factory in (genuine_fixer, value_hardcoder, string_hardcoder):
        fx_old, fx_new = fx_factory(), fx_factory()
        r_old = evaluate_skeptically(fx_old, inst, old_sites, gate, seed=7)
        r_new = evaluate_skeptically(fx_new, inst, new_sites, gate, seed=7)
        gate_row(fx_old.name, r_old, r_new)
    print("\nThe value-hardcoder passes the OLD gate (invisible — nothing perturbs the values")
    print("it memorized) and is REJECTED by the NEW gate once the metamorphic channel exists.")

    print("\n" + "=" * 88)
    print("3) ADAPTIVE GATE — decide in as few samples as possible")
    print("=" * 88)
    print("Probe one perturbation per axis in priority order; stop early on a confirmed")
    print("collapse. Only the fixed gate ever runs every perturbation.\n")
    for fx_factory in (genuine_fixer, value_hardcoder, string_hardcoder):
        fx = fx_factory()
        res = evaluate_adaptive(fx, inst, new_sites, gate, seed=7)
        print("  " + res.summary())
        if res.broke_on:
            probed = " -> ".join(rc.channel for rc in res.scores)
            print(f"      axes probed (in order): {probed}")
    print(f"\nFixed gate would run 1 + {len(new_sites)} = {1 + len(new_sites)} re-checks for every fixer.")
    print("The exploit is caught after the metamorphic axis (probed first); the genuine fix")
    print("pays only one probe per axis and is accepted.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "astropy__astropy-12907")
