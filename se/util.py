"""Small utilities: unified-diff parsing and lightweight repo file access.

We deliberately avoid a full `git clone` of large repos (astropy, django, ...).
The Perturbation Agent only needs to *read* a handful of files (README, the
test file, a couple of modules) to find concrete places to perturb, so we fetch
them individually from GitHub raw at the exact base commit and cache them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "filecache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"

# ---------------------------------------------------------------------------
# Unified diff parsing
# ---------------------------------------------------------------------------

_DIFF_GIT = re.compile(r"^diff --git a/(.+?) b/(.+?)$")
_MINUS = re.compile(r"^--- (?:a/)?(.+?)\s*$")
_PLUS = re.compile(r"^\+\+\+ (?:b/)?(.+?)\s*$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class TouchMap:
    """Files (and original-file line ranges) touched by a diff.

    ``ranges[path]`` is a list of inclusive ``(start, end)`` line spans in the
    *original* file. A code perturbation is only meaning-preserving for the
    patched fix if it does not overlap any of these.
    """

    files: set[str] = field(default_factory=set)
    ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def touches(self, path: str) -> bool:
        return path in self.files

    def overlaps(self, path: str, start: int, end: int) -> bool:
        for s, e in self.ranges.get(path, []):
            if start <= e and s <= end:
                return True
        return False


def parse_diff(diff_text: str) -> TouchMap:
    """Parse a unified diff into the set of files and original line ranges it edits."""
    tm = TouchMap()
    current: str | None = None
    for line in diff_text.splitlines():
        m = _DIFF_GIT.match(line)
        if m:
            current = m.group(2)
            tm.files.add(current)
            tm.ranges.setdefault(current, [])
            continue
        m = _PLUS.match(line)
        if m and m.group(1) != "/dev/null":
            current = m.group(1)
            tm.files.add(current)
            tm.ranges.setdefault(current, [])
            continue
        m = _HUNK.match(line)
        if m and current is not None:
            old_start = int(m.group(1))
            old_len = int(m.group(2) or "1")
            if old_len == 0:
                # pure insertion: mark the seam around the insertion point
                tm.ranges[current].append((old_start, old_start + 1))
            else:
                tm.ranges[current].append((old_start, old_start + old_len - 1))
    return tm


def diff_target_files(diff_text: str) -> list[str]:
    return sorted(parse_diff(diff_text).files)


# ---------------------------------------------------------------------------
# Lightweight repo file access (GitHub raw, cached)
# ---------------------------------------------------------------------------

README_CANDIDATES = [
    "README.rst", "README.md", "README.txt", "README",
    "readme.rst", "readme.md", "docs/README.rst",
]


class RepoAccess:
    """Fetch individual files from a repo at a fixed commit, cached to disk."""

    def __init__(self, timeout: int = 30, offline: bool = False):
        self.timeout = timeout
        self.offline = offline

    def _cache_path(self, repo: str, commit: str, path: str) -> Path:
        slug = f"{repo}__{commit[:12]}__{path}".replace("/", "__")
        return CACHE_DIR / slug

    def get_file(self, repo: str, commit: str, path: str) -> str | None:
        cache = self._cache_path(repo, commit, path)
        if cache.exists():
            text = cache.read_text(encoding="utf-8", errors="replace")
            return None if text == "\x00MISSING" else text
        if self.offline:
            return None
        url = RAW_URL.format(repo=repo, commit=commit, path=path)
        try:
            resp = requests.get(url, timeout=self.timeout)
        except requests.RequestException:
            return None
        if resp.status_code == 200:
            cache.write_text(resp.text, encoding="utf-8")
            return resp.text
        if resp.status_code == 404:
            cache.write_text("\x00MISSING", encoding="utf-8")
        return None

    def find_readme(self, repo: str, commit: str) -> tuple[str, str] | None:
        for cand in README_CANDIDATES:
            content = self.get_file(repo, commit, cand)
            if content:
                return cand, content
        return None
