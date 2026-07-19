"""String-level metamorphic perturbation: rename the *example values* in a bug
report while leaving the hidden tests untouched.

Why this exists: our integer/variable metamorphic perturbations found nothing on
`pallets__flask-5063`, because that report's hardcodable values are strings —
`subdomain='admin'`, `SERVER_NAME: 'test.local'`, `admin.test.local`. An agent that
copies those values into a special case passes the hidden tests; rewrite the report
to use different values and the copier is graded against values it never saw, while
a fix that actually reads Flask's routing table is unaffected.

Design note (driven by the real data): on flask-5063 the values live in the *prose*,
not in fenced code blocks — the code blocks only hold `flask routes` terminal output
whose one quoted literal is `"pip install python-dotenv"`, which must NOT be renamed.
So extraction spans the whole report, and we rename only two surgical categories:

  1. the contents of quoted literals  ('admin', 'test.local')
  2. domain-like dotted tokens        (admin.test.local, test.test.local)

Bare English words in prose are deliberately left alone — renaming a standalone
"test" would mangle ordinary sentences for no gain.
"""

from __future__ import annotations

import random
import re

# --- what must never be renamed -------------------------------------------

_UNSAFE_EXACT = {
    # HTTP verbs / protocol tokens
    "get", "post", "put", "patch", "delete", "head", "options", "http", "https",
    # encodings / mime / formats
    "utf-8", "utf8", "ascii", "latin-1", "json", "xml", "html", "csv", "yaml",
    "application/json", "text/html",
    # common structural words in tables/output
    "endpoint", "methods", "rule", "domain", "none", "true", "false", "null",
}
# file extensions and dotfiles we must not treat as domains
_CODE_EXTS = {"py", "txt", "json", "yaml", "yml", "cfg", "ini", "toml", "md", "rst",
              "env", "flaskenv", "lock", "log", "sh", "js", "ts", "html", "css"}
# Real TLD-ish suffixes. A dotted token only counts as a DOMAIN if its last part is
# one of these — otherwise `app.register_blueprint` and `admin_blueprint.home` (Python
# attribute access) get mangled into nonsense, which corrupts the example code.
_TLDS = {"local", "localhost", "com", "org", "net", "io", "dev", "co", "uk", "edu",
         "gov", "example", "test", "app", "xyz", "info", "biz"}

_QUOTED = re.compile(r"'([^'\n]*)'|\"([^\"\n]*)\"")
_DOMAIN = re.compile(r"(?<![\w.-])([A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)+)(?![\w-])")

# replacement pool, chosen to look like plausible alternatives (shape-preserving)
_NAME_POOL = ["staff", "beta", "portal", "shop", "intranet", "billing", "reports",
              "console", "gateway", "studio"]


def is_safe_to_rename(s: str) -> bool:
    """Reject load-bearing values; accept user-chosen example names."""
    v = s.strip()
    if not v or len(v) < 2:
        return False
    if " " in v or "/" in v:          # commands, paths, mime types
        return False
    if v.lower() in _UNSAFE_EXACT:
        return False
    if v.startswith(".") or v.startswith("-"):  # .env, .flaskenv, flags
        return False
    if any(ch.isdigit() for ch in v) and re.fullmatch(r"[\d.]+", v):  # version numbers
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", v):
        return False
    # a dotted token whose last part is a code extension is a filename, not a domain
    if "." in v and v.rsplit(".", 1)[-1].lower() in _CODE_EXTS:
        return False
    return True


def _is_domain_like(s: str) -> bool:
    """A dotted token is a DOMAIN only if its last part is a real TLD. This is what
    keeps `app.register_blueprint(...)` and `admin_blueprint.home` (Python attribute
    access) from being rewritten into nonsense."""
    if not _DOMAIN.fullmatch(s) or not is_safe_to_rename(s):
        return False
    return s.rsplit(".", 1)[-1].lower() in _TLDS


def extract_candidates(text: str) -> dict[str, list[str]]:
    """Find renameable example values across the WHOLE report (prose + code)."""
    quoted, domains = set(), set()
    for m in _QUOTED.finditer(text):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        if val is not None and is_safe_to_rename(val):
            (domains if _is_domain_like(val) else quoted).add(val)
    for m in _DOMAIN.finditer(text):
        val = m.group(1)
        if _is_domain_like(val):
            domains.add(val)
    return {"quoted": sorted(quoted), "domains": sorted(domains)}


def build_rename_map(candidates: dict[str, list[str]], seed: int = 0) -> dict[str, str]:
    """Map atomic names -> replacements, deterministically and shape-preservingly.

    Domains are renamed part-by-part (admin.test.local -> staff.demo.local) so the
    result still reads like a domain, and every occurrence stays consistent."""
    rng = random.Random(f"metamorphic-strings:{seed}")
    atoms: list[str] = []
    for d in candidates["domains"]:
        # Only the LAST part is the TLD. Everything before it is a user-chosen name
        # and is renameable — `test` in `test.local` is a subdomain, not a TLD.
        for part in d.split(".")[:-1]:
            if part not in atoms and is_safe_to_rename(part):
                atoms.append(part)
    for q in candidates["quoted"]:
        if q not in atoms:
            atoms.append(q)

    pool = _NAME_POOL[:]
    rng.shuffle(pool)
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for atom in atoms:
        # distinct atoms must never collide onto the same replacement, and a
        # replacement must not already appear as another candidate value
        while pool and (pool[-1] in used or pool[-1] in atoms):
            pool.pop()
        replacement = pool.pop() if pool else f"{atom}x"
        mapping[atom] = replacement
        used.add(replacement)
    return mapping


def _rename_domain(domain: str, mapping: dict[str, str]) -> str:
    """Rename every part except the final TLD, so the result still reads as a domain."""
    parts = domain.split(".")
    return ".".join([mapping.get(p, p) for p in parts[:-1]] + [parts[-1]])


def apply_rename(text: str, mapping: dict[str, str]) -> str:
    """Rewrite quoted literals and domain-like tokens consistently.

    Uses placeholders so a value that maps onto another candidate can never be
    replaced twice (the classic A->B then B->C cascade bug)."""
    if not mapping:
        return text

    placeholders: dict[str, str] = {}

    def _ph(value: str) -> str:
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = value
        return key

    # 1. domain-like tokens, longest first so admin.test.local wins over test.local
    def sub_domain(m: re.Match) -> str:
        tok = m.group(1)
        if not _is_domain_like(tok):   # leave Python attribute access alone
            return tok
        renamed = _rename_domain(tok, mapping)
        return _ph(renamed) if renamed != tok else tok

    out = _DOMAIN.sub(sub_domain, text)

    # 2. quoted literal contents
    def sub_quoted(m: re.Match) -> str:
        raw = m.group(0)
        val = m.group(1) if m.group(1) is not None else m.group(2)
        if val is None or not is_safe_to_rename(val):
            return raw
        quote = raw[0]
        new = _rename_domain(val, mapping) if "." in val else mapping.get(val, val)
        if new == val:
            return raw
        return _ph(f"{quote}{new}{quote}")

    out = _QUOTED.sub(sub_quoted, out)

    for key, value in placeholders.items():
        out = out.replace(key, value)
    return out


def rename_example_values(text: str, seed: int = 0) -> tuple[str, dict[str, str]]:
    """Convenience: extract -> map -> apply. Returns (new_text, mapping)."""
    cands = extract_candidates(text)
    mapping = build_rename_map(cands, seed)
    return apply_rename(text, mapping), mapping
