"""Per-scenario ViZDoom button-name lookup.

Adding a new ViZDoom scenario requires (a) a new entry here, (b) wiring in
`factories.py`, and (c) a prompt template under
`src/cleanrl_vlm/prompts/templates/vizdoom/<slug>/`.
"""

from __future__ import annotations

action_tables: dict[str, list[str]] = {
    "VizdoomBasic-v0": ["MOVE_LEFT", "MOVE_RIGHT", "ATTACK"],
    "VizdoomCorridor-v0": [
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "ATTACK",
        "MOVE_FORWARD",
        "MOVE_BACKWARD",
        "TURN_LEFT",
        "TURN_RIGHT",
    ],
    "VizdoomDefendLine-v0": ["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
}
