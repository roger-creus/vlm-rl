"""Unified env-id → action-name table lookup across env families.

Trainers import this single map; the per-family tables
(``vizdoom/action_tables.py``, ``atari/action_tables.py``) stay
co-located with their factories for discoverability.
"""

from __future__ import annotations

from cleanrl_vlm.envs.atari.action_tables import action_tables as _atari_tables
from cleanrl_vlm.envs.vizdoom.action_tables import action_tables as _vizdoom_tables

action_tables: dict[str, list[str]] = {**_vizdoom_tables, **_atari_tables}
