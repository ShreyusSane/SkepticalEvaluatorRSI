"""Load API keys without ever putting them in source, chat, or logs.

Resolution order for any key name:
  1. the process environment
  2. `.env` at the project root
  3. `server/.env` (legacy layout)

Values are read at call time and passed straight to the relevant SDK client. They
are never printed — `describe_key()` reports only whether a key was found and,
at most, a masked fingerprint so you can tell two keys apart.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_CANDIDATES = [_PROJECT_ROOT / ".env", _PROJECT_ROOT / "server" / ".env"]


def load_key(name: str) -> str | None:
    """Return the value of `name` from the environment or a project .env file."""
    val = os.environ.get(name)
    if val:
        return val.strip()
    pattern = re.compile(rf"\s*(?:export\s+)?{re.escape(name)}\s*=\s*(.+?)\s*$")
    for env_path in _ENV_CANDIDATES:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = pattern.match(line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    return None


def describe_key(name: str) -> str:
    """Safe, loggable status for a key — never reveals the secret."""
    val = load_key(name)
    if not val:
        return f"{name}: NOT FOUND (set it in the environment or {_PROJECT_ROOT / '.env'})"
    return f"{name}: found (…{val[-4:]}, {len(val)} chars)"


def require_key(name: str) -> str:
    val = load_key(name)
    if not val:
        raise RuntimeError(
            f"{name} not found. Put it in the environment or in "
            f"{_PROJECT_ROOT / '.env'} as `{name}=...` (that file is gitignored)."
        )
    return val
