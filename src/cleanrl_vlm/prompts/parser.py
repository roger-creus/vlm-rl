"""COT action parser: regex last-match + whitelist (reviewer M2 + M3)."""

from __future__ import annotations

import re

_ACTION_RE = re.compile(r"ACTION:\s*([A-Z_]+)")


def parse_action_cot(text: str, action_names: list[str]) -> int | None:
    """Extract the trailing ACTION from a VLM generation.

    Strategy (reviewer M3):
    - Regex ``r"ACTION:\\s*([A-Z_]+)"``.
    - Take the **last** match in the text (model sometimes emits multiple).
    - Whitelist against ``action_names``.
    - Return ``None`` on no-match / not-in-whitelist; caller samples uniformly.
    """
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    last = matches[-1].strip()
    if last not in action_names:
        return None
    return action_names.index(last)
