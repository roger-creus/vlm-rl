"""Accelerate / DeepSpeed config loading."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_accelerator_config(config_path: str | Path, num_processes: int) -> dict:
    """Load an accelerate YAML.

    If ``num_processes == 1``, emit a startup log line noting the sharding
    strategy is ignored (reviewer m11).
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    sharding = cfg.get("distributed_type") or cfg.get("deepspeed_config") or "unknown"
    if num_processes == 1:
        log.info("sharding=%s (ignored at num_processes=1)", sharding)
    return cfg
