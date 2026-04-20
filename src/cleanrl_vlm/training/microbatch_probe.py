"""Startup microbatch auto-probe (reviewer M7)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)


def probe_microbatch(try_batch_fn: Callable[[int], bool], cap: int = 64) -> int:
    """Double microbatch size until it fails (OOM) or hits ``cap``.

    Returns the largest size that succeeded, clamped to at least 1.
    """
    last_good = 0
    size = 1
    while size <= cap:
        ok = try_batch_fn(size)
        if not ok:
            break
        last_good = size
        size *= 2
    return max(1, last_good)


def record_microbatch_probe(run_dir: Path, per_gpu_microbatch: int, target_batch_floor: int) -> None:
    """Write ``runs/<name>/microbatch_probe.json`` per reviewer M7."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "per_gpu_microbatch": per_gpu_microbatch,
        "target_batch_floor": target_batch_floor,
    }
    (run_dir / "microbatch_probe.json").write_text(json.dumps(payload, indent=2))
    log.info("microbatch probe: %s", payload)
