"""Per-ALE-game action-meaning lookup.

ALE-v5 envs return `Discrete(N)` action spaces; the parser needs the
textual action name the VLM emits (e.g. ``ACTION: RIGHT``). We keep a
static map here rather than calling ``env.unwrapped.get_action_meanings()``
at import time — that would force gym.make to succeed for every entry,
which pulls in Stella + the full ALE ROM loader.

Keys are the gymnasium-registered env ids (``ALE/<Game>-v5``).
"""

from __future__ import annotations

action_tables: dict[str, list[str]] = {
    # Pong: full 6-action set. NOOP + FIRE + 4 paddle moves.
    "ALE/Pong-v5": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
}
