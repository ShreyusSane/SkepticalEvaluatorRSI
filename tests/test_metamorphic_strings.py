"""Tests for the string-metamorphic perturbation primitives."""

from __future__ import annotations

from se.metamorphic_strings import (
    build_rename_map,
    extract_candidates,
    is_safe_to_rename,
    rename_example_values,
)
from se.perturbations import renameable_locals, shiftable_literals


def test_safety_filter_rejects_load_bearing_values():
    """Values whose meaning is fixed by the language/protocol must never change."""
    for unsafe in ("utf-8", "GET", "", "/", "a", ".env", "application/json",
                   "pip install python-dotenv", "1.2.3"):
        assert not is_safe_to_rename(unsafe), f"{unsafe!r} should be rejected"
    for safe in ("admin", "test.local", "admin.test.local", "my_service"):
        assert is_safe_to_rename(safe), f"{safe!r} should be accepted"


def test_rename_is_consistent_and_leaves_no_original_values(flask_instance):
    """Every occurrence is rewritten — prose, quoted literals and compound domains —
    so the report stays internally coherent and nothing hardcodable survives."""
    original = flask_instance.problem_statement
    new, mapping = rename_example_values(original, seed=0)

    # nothing hardcodable survives
    for stale in ("admin.test.local", "test.test.local", "test.local",
                  "subdomain='admin'", "subdomain='test'"):
        assert stale not in new, f"{stale!r} survived the rename"

    # distinct source values never collide onto the same replacement
    assert len(set(mapping.values())) == len(mapping)

    # structure that is NOT an example value is preserved
    assert "app.register_blueprint" in new, "Python API call was corrupted"
    assert "admin_blueprint.home" in new, "endpoint name was corrupted"
    assert ".local" in new, "TLD should be preserved so it still reads as a domain"
    assert "pip install python-dotenv" in new, "a real shell command was renamed"


def test_flask_gap_is_closed(flask_instance):
    """Regression test for the gap that motivated this work: flask-5063 previously
    produced ZERO metamorphic candidates because its repro values are strings."""
    # the old int/variable primitives still find nothing here — that was the gap
    assert renameable_locals(flask_instance.problem_statement) == []
    assert shiftable_literals(flask_instance.problem_statement) == []

    # the new string primitive finds the real example values
    cands = extract_candidates(flask_instance.problem_statement)
    assert "admin" in cands["quoted"]
    assert "test.local" in cands["domains"]
    assert build_rename_map(cands, seed=0), "expected a non-empty rename map"

    # meaning preservation: the graded artifacts are untouched
    before_patch, before_tests = flask_instance.patch, flask_instance.test_patch
    rename_example_values(flask_instance.problem_statement, seed=0)
    assert flask_instance.patch == before_patch
    assert flask_instance.test_patch == before_tests


def test_astropy_still_works(astropy_instance):
    """Non-regression: the int/variable channel that already worked must keep working."""
    assert renameable_locals(astropy_instance.problem_statement) == ["cm"]
    assert shiftable_literals(astropy_instance.problem_statement), "expected int literals"
