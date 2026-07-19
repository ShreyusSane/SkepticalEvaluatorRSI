"""Run the Perturbation Agent on a real SWE-bench instance and show the
concrete, grounded perturbation sites it finds — plus a couple applied examples.

Usage:
    python scripts/02_run_perturbation_agent.py [instance_id]
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.perturbations import PerturbationAgent
from se.swebench_data import get_instance
from se.util import parse_diff


def main(instance_id: str = "astropy__astropy-12907") -> None:
    inst = get_instance(instance_id)
    print("=" * 78)
    print(f"INSTANCE  {inst.instance_id}   ({inst.repo} @ {inst.base_commit[:10]})")
    print("=" * 78)

    gold = parse_diff(inst.patch)
    tests = parse_diff(inst.test_patch)
    print("\nOFF-LIMITS (edited by gold fix or hidden tests — never perturb these):")
    for f, r in {**gold.ranges, **tests.ranges}.items():
        print(f"    {f}  lines {r}")

    agent = PerturbationAgent()  # no LLM -> deterministic perturbations only
    sites = agent.find_sites(inst)

    print(f"\nCONCRETE PERTURBATION SITES FOUND: {len(sites)}")
    print("-" * 78)
    for s in sites:
        print(s)

    safe = [s for s in sites if s.safe]
    print("-" * 78)
    print(f"{len(safe)}/{len(sites)} sites pass the static meaning-preservation check.\n")

    # Show two applied examples end to end.
    show = [s for s in safe if s.kind in ("ps_reorder_paragraphs", "readme_reorder_sections",
                                          "test_reorder", "readme_inject_note")]
    for s in show[:3]:
        ap = agent.apply(inst, s)
        print("#" * 78)
        print(f"APPLIED: {s.kind}  ->  {ap.notes}")
        print("#" * 78)
        if s.kind.startswith("ps_"):
            print(textwrap.indent(ap.problem_statement[:500], "    "))
        elif ap.file_overrides:
            for path, content in ap.file_overrides.items():
                print(f"    [{path}] first 500 chars:")
                print(textwrap.indent(content[:500], "    "))
        elif ap.evaluator_hints:
            h = ap.evaluator_hints
            print(f"    test file : {h['test_file']}")
            print(f"    original  : {h['original_order'][:6]}{' ...' if len(h['original_order'])>6 else ''}")
            print(f"    perturbed : {h['perturbed_order'][:6]}{' ...' if len(h['perturbed_order'])>6 else ''}")
            print(f"    F2P (must still pass regardless of order): {h['f2p']}")
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "astropy__astropy-12907")
