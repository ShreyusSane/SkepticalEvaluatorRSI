"""Coverage survey: how often does each metamorphic channel actually fire?

This is the "be intentional" test. Rather than assuming a perturbation is useful, it
measures the firing rate across all 100 cached SWE-bench Lite instances so we know
which channel to showcase and which to demote.

Offline by construction: it only reads each instance's problem statement, so no
README/test-file fetches and no network.

Run `pytest tests/test_coverage_survey.py -s` to see the table.
"""

from __future__ import annotations

from se.metamorphic_strings import build_rename_map, extract_candidates
from se.perturbations import renameable_locals, shiftable_literals


def _fires(inst) -> dict[str, bool]:
    ps = inst.problem_statement
    cands = extract_candidates(ps)
    return {
        "metamorphic_rename_repro (vars)": bool(renameable_locals(ps)),
        "metamorphic_shift_literals (ints)": bool(shiftable_literals(ps)),
        "metamorphic_rename_strings (NEW)": bool(build_rename_map(cands, seed=0)),
    }


def test_channel_coverage_survey(cached_instances):
    channels = ["metamorphic_rename_repro (vars)",
                "metamorphic_shift_literals (ints)",
                "metamorphic_rename_strings (NEW)"]
    counts = dict.fromkeys(channels, 0)
    any_metamorphic_old = 0
    any_metamorphic_new = 0

    for inst in cached_instances:
        fired = _fires(inst)
        for ch in channels:
            counts[ch] += int(fired[ch])
        old = fired[channels[0]] or fired[channels[1]]
        any_metamorphic_old += int(old)
        any_metamorphic_new += int(old or fired[channels[2]])

    total = len(cached_instances)
    print(f"\n{'channel':<38}{'fires on':>10}")
    print("-" * 50)
    for ch in channels:
        print(f"{ch:<38}{counts[ch]:>5}/{total} ({100*counts[ch]/total:.0f}%)")
    print("-" * 50)
    print(f"{'ANY metamorphic (before this change)':<38}"
          f"{any_metamorphic_old:>5}/{total} ({100*any_metamorphic_old/total:.0f}%)")
    print(f"{'ANY metamorphic (after this change)':<38}"
          f"{any_metamorphic_new:>5}/{total} ({100*any_metamorphic_new/total:.0f}%)")

    # The new channel must not shrink coverage, and should meaningfully extend it.
    assert any_metamorphic_new >= any_metamorphic_old
    assert counts[channels[2]] > 0, "the new string channel never fired"
