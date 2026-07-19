"""The real run: A-e on t and on t0…tn, graded by the official SWE-bench harness.

This is the experiment the whole project is built around:

    t          = the original task
    t0 … tn    = meaning-preserving perturbations of it (already verified in
                 scripts/09 — grading GOLD under each still resolves at 100.0)
    A-e        = a real LLM code-fixing agent

Run A-e on every variant, grade each patch for real, and read the pass/fail vector:

  * passes everywhere            -> genuine fix; the gate must ACCEPT (no false positive)
  * passes t, fails the t_i's    -> it keyed on something incidental; the gate REJECTS,
                                    and the failing axis names what it leaned on

Because every t_i is verified meaning-preserving, a failure there cannot be blamed
on the perturbation — the only remaining explanation is that A-e did not generalise.

PAID: one agent run + one graded run per variant (~2 sandboxes each) plus Anthropic
tokens for the agent loop. Use --adaptive to probe one perturbation per axis instead
of all of them, and --model claude-sonnet-5 to cut the agent cost.

Usage:
  python scripts/11_agent_across_perturbations.py [instance_id] [--adaptive] [--model M]
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.agent import SimpleSweAgent
from se.daytona_fixer import DaytonaFixer
from se.daytona_runner import DaytonaRunner
from se.evaluator import (
    SkepticalGate,
    evaluate_adaptive,
    evaluate_skeptically,
    safe_sites_for,
)
from se.patch_inspect import summarize
from se.secrets import load_key
from se.swebench_data import get_instance


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    instance_id = args[0] if args else "pallets__flask-5063"
    model = next((a.split("=", 1)[1] for a in flags if a.startswith("--model=")),
                 "claude-opus-4-8")
    adaptive = "--adaptive" in flags

    for key in ("DAYTONA_API_KEY", "ANTHROPIC_API_KEY"):
        if not load_key(key):
            print(f"{key} missing — add it to SkepticalEvaluator/.env"); sys.exit(1)

    inst = get_instance(instance_id)
    fixer = DaytonaFixer(
        runner=DaytonaRunner(cpu=2, memory=4, disk=10),
        agent=SimpleSweAgent(model=model, max_steps=14),
    )
    sites = safe_sites_for(inst, fixer.pert)
    gate = SkepticalGate(bar=80.0, tau=8.0)

    print("=" * 92)
    print(f"A-e ACROSS t AND t0…tn — {inst.instance_id} ({inst.repo})")
    print("=" * 92)
    print(f"agent model : {model}")
    print(f"variants    : t + {len(sites)} perturbations "
          f"({'adaptive: one per axis' if adaptive else 'all'})")
    print("every perturbation was verified meaning-preserving in scripts/09\n")

    if adaptive:
        res = evaluate_adaptive(fixer, inst, sites, gate)
    else:
        res = evaluate_skeptically(fixer, inst, sites, gate)

    print("\n" + "=" * 92)
    print("PASS/FAIL VECTOR (real SWE-bench grading)")
    print("=" * 92)
    print(f"{'variant':<28}{'axis':<14}{'score':>7}   {'resolved':<10}{'inspector'}")
    print("-" * 92)
    for rc in res.scores:
        findings = fixer.inspections.get((inst.instance_id, rc.kind), [])
        insp = summarize(findings) if findings else "clean"
        resolved = "yes" if rc.score >= 99.9 else "no"
        label = "t (original)" if rc.kind == "canonical" else rc.kind
        print(f"{label:<28}{rc.channel:<14}{rc.score:>7.1f}   {resolved:<10}{insp}")

    values = [rc.score for rc in res.scores]
    canonical = res.scores[0].score
    perturbed = values[1:]
    print("-" * 92)
    print(f"t = {canonical:.1f}   |   t0..tn mean = "
          f"{statistics.mean(perturbed) if perturbed else float('nan'):.1f}   "
          f"sigma = {res.std:.2f}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(res.summary())
    if res.accepted:
        print("\nA-e generalised: it survived every meaning-preserving variant.")
        print("This is the control case — the gate does NOT punish an honest agent.")
    else:
        broke = res.failure_bundle.get("broke_on_channels") or ["(see worst rechecks)"]
        print(f"\nA-e did NOT generalise. It leaned on: {broke}")
        print("t passed but its perturbed twins did not, and every twin was verified")
        print("meaning-preserving — so the perturbation cannot be blamed. A single")
        print("pass/fail score on t alone would have certified this patch.")


if __name__ == "__main__":
    main(sys.argv[1:])
