"""Fetch real SWE-bench instances without the heavy Docker evaluation harness.

We only need the *data* (repo id, base commit, problem statement, gold patch,
test patch, and the FAIL_TO_PASS / PASS_TO_PASS test lists). The HuggingFace
datasets-server REST API serves this as plain JSON rows, so there is no
dependency on the `datasets` library or on Docker here.

A SWE-bench instance is one GitHub issue + the commit it was filed against +
the gold fix + the tests that verify the fix. That is exactly the input a
code-fixing agent consumes, which is what the Perturbation Agent operates on.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "princeton-nlp/SWE-bench_Lite"

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SweInstance:
    """One SWE-bench task. Field names mirror the dataset columns."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str  # the gold solution diff
    test_patch: str  # diff that adds/updates the verifying tests
    fail_to_pass: list[str]  # tests that must flip fail -> pass
    pass_to_pass: list[str]  # tests that must stay passing
    hints_text: str = ""
    version: str = ""
    environment_setup_commit: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def repo_url(self) -> str:
        return f"https://github.com/{self.repo}.git"

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SweInstance":
        def _as_list(v: Any) -> list[str]:
            if isinstance(v, list):
                return [str(x) for x in v]
            if isinstance(v, str) and v.strip():
                try:
                    parsed = json.loads(v)
                    return [str(x) for x in parsed]
                except json.JSONDecodeError:
                    return [v]
            return []

        return cls(
            instance_id=row["instance_id"],
            repo=row["repo"],
            base_commit=row["base_commit"],
            problem_statement=row.get("problem_statement", ""),
            patch=row.get("patch", ""),
            test_patch=row.get("test_patch", ""),
            fail_to_pass=_as_list(row.get("FAIL_TO_PASS")),
            pass_to_pass=_as_list(row.get("PASS_TO_PASS")),
            hints_text=row.get("hints_text", "") or "",
            version=str(row.get("version", "")),
            environment_setup_commit=row.get("environment_setup_commit", "") or "",
            raw=row,
        )


def _cache_path(dataset: str, split: str, offset: int, length: int) -> Path:
    slug = dataset.replace("/", "__")
    return CACHE_DIR / f"{slug}__{split}__{offset}_{length}.json"


def fetch_rows(
    dataset: str = DEFAULT_DATASET,
    split: str = "test",
    offset: int = 0,
    length: int = 20,
    config: str = "default",
    use_cache: bool = True,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """Return raw dataset rows, cached to disk so we do not re-hit the API."""
    length = min(length, 100)  # datasets-server caps page size at 100
    cache = _cache_path(dataset, split, offset, length)
    if use_cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    params = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    resp = requests.get(DATASETS_SERVER, params=params, timeout=timeout)
    resp.raise_for_status()
    rows = [entry["row"] for entry in resp.json().get("rows", [])]
    cache.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def load_instances(
    dataset: str = DEFAULT_DATASET,
    split: str = "test",
    offset: int = 0,
    length: int = 20,
    use_cache: bool = True,
) -> list[SweInstance]:
    rows = fetch_rows(dataset, split, offset, length, use_cache=use_cache)
    return [SweInstance.from_row(r) for r in rows]


def get_instance(
    instance_id: str,
    dataset: str = DEFAULT_DATASET,
    split: str = "test",
    search_span: int = 300,
    page: int = 100,
) -> SweInstance:
    """Find a single instance by id, paging through the dataset as needed."""
    for offset in range(0, search_span, page):
        for inst in load_instances(dataset, split, offset, page):
            if inst.instance_id == instance_id:
                return inst
    raise KeyError(f"instance {instance_id!r} not found in first {search_span} rows")


if __name__ == "__main__":
    insts = load_instances(length=3)
    for inst in insts:
        print(f"{inst.instance_id:30s} {inst.repo:20s} "
              f"F2P={len(inst.fail_to_pass)} P2P={len(inst.pass_to_pass)} "
              f"ps={len(inst.problem_statement)}chars")
