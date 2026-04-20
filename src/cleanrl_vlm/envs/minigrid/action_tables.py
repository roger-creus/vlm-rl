"""Per-MiniGrid-scenario action-name lookup.

MiniGrid's native action space is ``Discrete(7)`` with a fixed ordering:
``[left, right, forward, pickup, drop, toggle, done]``. We expose the
uppercase names here so the parser regex (``r"ACTION:\\s*([A-Z_]+)"``)
captures them cleanly.

For navigation-only envs like ``MiniGrid-Empty-*`` the first three
actions suffice; the prompt nudges the VLM toward those while keeping
the full native action set available (master-spec §4: "MiniGrid
discrete actions are native").
"""

from __future__ import annotations

_NATIVE_ACTIONS = ["LEFT", "RIGHT", "FORWARD", "PICKUP", "DROP", "TOGGLE", "DONE"]

action_tables: dict[str, list[str]] = {
    "MiniGrid-Empty-5x5-v0": _NATIVE_ACTIONS,
}
