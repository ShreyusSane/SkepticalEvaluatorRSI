"""Tests for static patch inspection.

The false-positive test matters more than the positive ones: a detector that flags
legitimate human fixes would veto good patches and is worse than no detector.
"""

from __future__ import annotations

from se.patch_inspect import has_tampering, inspect_patch


def test_real_gold_patches_are_clean(cached_instances):
    """FALSE-POSITIVE GUARD: real human fixes must produce zero findings."""
    offenders = []
    for inst in cached_instances[:25]:
        findings = inspect_patch(inst.patch)
        if findings:
            offenders.append((inst.instance_id, [f.kind for f in findings]))
    assert not offenders, f"gold patches wrongly flagged: {offenders}"


def test_cheat_shapes_are_detected(cheat_patch):
    """A stub body, a swallowed exception and a skip marker are all caught."""
    kinds = {f.kind for f in inspect_patch(cheat_patch)}
    assert "stub_body" in kinds
    assert "except_pass" in kinds
    assert "skip_marker" in kinds


def test_conftest_creation_is_high_severity(conftest_tamper_patch):
    """The documented evaluator-tampering vector must be veto-worthy."""
    findings = inspect_patch(conftest_tamper_patch)
    assert any(f.kind == "conftest_created" and f.severity == "high" for f in findings)
    assert has_tampering(findings), "conftest.py creation should trigger a veto"
