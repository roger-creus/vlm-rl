"""Per-scenario ViZDoom button-name lookup.

Adding a new ViZDoom scenario requires (a) a new entry here, (b) wiring in
`factories.py`, and (c) a prompt template under
`src/cleanrl_vlm/prompts/templates/vizdoom/<slug>/`.

Keys match the gymnasium-registered env IDs from `vizdoom.gymnasium_wrapper`
(v1 series; older v0 names like `VizdoomCorridor-v0` are deprecated).
"""

from __future__ import annotations

action_tables: dict[str, list[str]] = {
    "VizdoomBasic-v1": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
    "VizdoomDeadlyCorridor-v1": [
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
    ],
    "VizdoomDefendLine-v1": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
}
