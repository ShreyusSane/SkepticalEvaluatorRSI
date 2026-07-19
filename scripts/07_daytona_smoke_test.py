"""Zero-Anthropic-cost pipeline check for the Daytona real-eval path.

Creates a sandbox from a real SWE-bench instance image, evaluates the GOLD patch
(must resolve) and an EMPTY patch (must not). If both hold, the Daytona + SWE-bench
eval path is correct and we can trust the real numbers. Uses a light repo by
default so the image fits Daytona's 10GB sandbox disk.

Prereqs: pip install daytona swebench ; set DAYTONA_API_KEY.
Usage: python scripts/07_daytona_smoke_test.py [instance_id]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se.daytona_runner import DaytonaRunner, instance_image
from se.secrets import describe_key, load_key
from se.swebench_data import get_instance, load_instances

# Light, pure-Python repos whose images comfortably fit a 10GB sandbox.
LIGHT_REPOS = {"pallets/flask", "psf/requests", "pytest-dev/pytest",
               "sqlfluff/sqlfluff", "marshmallow-code/marshmallow"}


def pick_light_instance() -> str:
    for offset in range(0, 300, 100):
        for inst in load_instances(offset=offset, length=100):
            if inst.repo in LIGHT_REPOS:
                return inst.instance_id
    return "astropy__astropy-12907"  # fallback (heavier image)


def main() -> None:
    print(describe_key("DAYTONA_API_KEY"))
    if not load_key("DAYTONA_API_KEY"):
        print("Add it to SkepticalEvaluator/.env (gitignored) or the environment.")
        sys.exit(1)

    instance_id = sys.argv[1] if len(sys.argv) > 1 else pick_light_instance()
    inst = get_instance(instance_id)
    print(f"Instance : {inst.instance_id}  ({inst.repo})")
    print(f"Image    : {instance_image(inst)}")
    print("Running smoke test (gold patch must resolve; empty patch must not)...\n")

    runner = DaytonaRunner(cpu=2, memory=4, disk=10)
    gold, empty = runner.smoke_test(inst)

    print("\n" + "=" * 60)
    print("gold  :", gold.summary())
    print("empty :", empty.summary())
    ok = gold.resolved and not empty.resolved
    print("=" * 60)
    print("SMOKE TEST", "PASSED — the real eval path is trustworthy." if ok
          else "FAILED — inspect output above (image fit / swebench version / disk).")


if __name__ == "__main__":
    main()
