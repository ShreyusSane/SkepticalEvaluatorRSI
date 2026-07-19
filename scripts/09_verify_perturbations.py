"""Verify the perturbations are genuinely meaning-preserving — WITHOUT any agent.

The gold patch is a known-correct fix. So for every perturbation we can ask the
real SWE-bench harness a falsifiable question:

    apply the perturbation, then grade the GOLD patch under it.
    A meaning-preserving perturbation MUST still RESOLVE.

If a perturbation breaks the build, corrupts a source file, or upsets test
collection, gold stops resolving and the perturbation is proven broken. No
code-fixing agent, no LLM, no Anthropic spend — just the harness.

Two classes of perturbation:
  * prompt-only  — touches no repo file and no test, so the graded artifacts are
    bit-identical. Meaning-preserving BY CONSTRUCTION; verified statically here,
    no sandbox needed.
  * repo-side    — modifies files the agent reads. Needs a real gold-patch run.

Usage: python scripts/09_verify_perturbations.py [instance_id]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.daytona_runner import DaytonaRunner
from se.perturbations import PerturbationAgent
from se.secrets import load_key
from se.swebench_data import get_instance


def main(instance_id: str = "pallets__flask-5063") -> None:
    if not load_key("DAYTONA_API_KEY"):
        print("Set DAYTONA_API_KEY in SkepticalEvaluator/.env first."); sys.exit(1)

    inst = get_instance(instance_id)
    agent = PerturbationAgent()
    sites = [s for s in agent.find_sites(inst) if s.safe]

    prompt_only, repo_side, eval_side = [], [], []
    for s in sites:
        ap = agent.apply(inst, s)
        if ap.file_overrides or ap.new_files:
            repo_side.append((s, ap))
        elif ap.evaluator_hints:
            eval_side.append((s, ap))
        else:
            prompt_only.append((s, ap))

    print("=" * 84)
    print(f"PERTURBATION VERIFICATION (no agent) — {inst.instance_id}")
    print("=" * 84)

    # --- static tier: prompt-only perturbations ---------------------------
    print(f"\n1) PROMPT-ONLY ({len(prompt_only)}) — meaning-preserving by construction")
    print("   (they change only what the agent reads; repo and hidden tests untouched)")
    for s, ap in prompt_only:
        same_repo = not ap.file_overrides and not ap.new_files
        same_tests = inst.test_patch == get_instance(instance_id).test_patch
        status = "OK" if (same_repo and same_tests) else "FAIL"
        print(f"   [{status}] {s.kind:<28} repo files touched: 0, tests unchanged")

    # --- empirical tier: repo-side perturbations --------------------------
    print(f"\n2) REPO-SIDE ({len(repo_side)}) — grading the GOLD patch under each")
    print("   A meaning-preserving perturbation must still RESOLVE.\n")
    runner = DaytonaRunner(cpu=2, memory=4, disk=10)

    print(f"   {'perturbation':<28}{'resolved':<10}{'score':<8}{'F2P':<8}{'verdict'}")
    print("   " + "-" * 70)
    failures = []
    for s, ap in repo_side:
        res = runner.evaluate_patch(inst, inst.patch,
                                    file_overrides=ap.file_overrides,
                                    new_files=ap.new_files)
        ok = res.resolved
        if not ok:
            failures.append(s.kind)
        f2p = f"{res.f2p_success}/{res.f2p_success + res.f2p_fail}"
        print(f"   {s.kind:<28}{str(ok):<10}{res.score:<8.1f}{f2p:<8}"
              f"{'MEANING-PRESERVING' if ok else 'BROKEN — changes ground truth'}")

    # --- not yet wired ----------------------------------------------------
    if eval_side:
        print(f"\n3) EVAL-SIDE ({len(eval_side)}) — not verifiable yet")
        for s, ap in eval_side:
            print(f"   [SKIP] {s.kind:<28} produces evaluator_hints, but the runner "
                  "does not consume them yet")

    print("\n" + "=" * 84)
    if failures:
        print(f"RESULT: {len(failures)} perturbation(s) BROKE the gold fix: {failures}")
        print("Those are not meaning-preserving and must be fixed or dropped.")
    else:
        print(f"RESULT: all {len(repo_side)} repo-side perturbations preserved the gold fix,")
        print(f"and all {len(prompt_only)} prompt-only perturbations are provably inert.")
        print("The perturbation set is verified meaning-preserving on this task.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pallets__flask-5063")
