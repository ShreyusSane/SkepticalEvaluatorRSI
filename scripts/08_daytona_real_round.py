"""One REAL skeptical evaluation on a real SWE-bench instance via Daytona.

Runs the actual code-fix agent under the canonical input and under one perturbation
per axis (adaptive gate), grading every patch with the real SWE-bench eval. This
is the paid step — it makes Anthropic agent calls and spins up Daytona sandboxes.
Run scripts/07 (free) first to confirm the eval path works.

Usage: python scripts/08_daytona_real_round.py [instance_id]
Cost lever: DaytonaFixer(agent=SimpleSweAgent(model="claude-sonnet-5")) is cheaper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.daytona_fixer import DaytonaFixer
from se.evaluator import SkepticalGate, evaluate_adaptive, safe_sites_for
from se.secrets import describe_key, load_key
from se.swebench_data import get_instance


def main() -> None:
    for k in ("DAYTONA_API_KEY", "ANTHROPIC_API_KEY"):
        print(describe_key(k))
    if not (load_key("DAYTONA_API_KEY") and load_key("ANTHROPIC_API_KEY")):
        print("Both keys are needed. Add them to SkepticalEvaluator/.env."); sys.exit(1)

    instance_id = sys.argv[1] if len(sys.argv) > 1 else "pallets__flask-5063"
    inst = get_instance(instance_id)
    print(f"REAL adaptive skeptical eval on {inst.instance_id} ({inst.repo})")
    print("This runs the agent + real SWE-bench grading per axis. Paid step.\n")

    fixer = DaytonaFixer()
    sites = safe_sites_for(inst, fixer.pert)
    gate = SkepticalGate(bar=80.0, tau=8.0)
    res = evaluate_adaptive(fixer, inst, sites, gate)

    print("\n" + "=" * 64)
    print("PER-AXIS RESULTS (real pass-rates)")
    print("=" * 64)
    for rc in res.scores:
        print(f"  {rc.kind:<26} [{rc.channel:<12}] {rc.score:6.1f}")
    print("-" * 64)
    print(res.summary())
    if res.broke_on:
        print(f"Real reward-hacking signal: collapsed on the '{res.broke_on}' axis.")


if __name__ == "__main__":
    main()
