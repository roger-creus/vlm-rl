"""COT action parser: regex last-match + whitelist."""

from __future__ import annotations

import re

_ACTION_RE = re.compile(r"ACTION:\s*([A-Z_]+)")


def parse_action_cot(text: str, action_names: list[str]) -> int | None:
    """Extract the trailing ACTION from a VLM generation.

    The model sometimes emits multiple ``ACTION: X`` lines during its
    chain-of-thought; take the last one (that's the committed choice).
    Returns ``None`` on no-match or not-in-whitelist so the caller can
    fall back to a uniform sample.
    """
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    last = matches[-1].strip()
    if last not in action_names:
        return None
    return action_names.index(last)
