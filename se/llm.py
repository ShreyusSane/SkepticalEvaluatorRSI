"""Optional LLM helper for the semantic-paraphrase perturbations.

Deterministic perturbations (reorderings, noise injection, comment insertion)
need no model and run offline. The paraphrase perturbations — reword the issue
or the README while preserving meaning — need a language model. This module
wraps the Anthropic SDK and is imported lazily, so nothing here runs unless a
paraphrase perturbation is actually requested.

Key resolution: the Anthropic SDK reads ANTHROPIC_API_KEY from the environment
first; if it is not set we fall back to reading it from a `.env` at the project
root (create `SkepticalEvaluator/.env` with `ANTHROPIC_API_KEY=...`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

# The paraphrase task is simple; per the API guidance the default is the most
# capable model, overridable. For high-volume runs pass model="claude-haiku-4-5".
DEFAULT_MODEL = "claude-opus-4-8"

def _load_api_key() -> Optional[str]:
    """Kept for backwards compatibility; the shared loader lives in se/secrets.py."""
    from .secrets import load_key
    return load_key("ANTHROPIC_API_KEY")


def make_llm(model: str = DEFAULT_MODEL, max_tokens: int = 2000) -> Callable[[str], str]:
    """Return a `prompt -> completion` function backed by the Anthropic API."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use paraphrase perturbations") from e

    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found (checked env and the project .env). "
            "Set the env var or create SkepticalEvaluator/.env; paraphrase "
            "perturbations need a key, deterministic perturbations do not."
        )

    client = anthropic.Anthropic(api_key=api_key)

    def call(prompt: str) -> str:
        # No thinking needed for a paraphrase; omitting it keeps latency/cost low.
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return call


def llm_available() -> bool:
    return _load_api_key() is not None
