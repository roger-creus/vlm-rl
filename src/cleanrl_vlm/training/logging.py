"""Three-sink logging: Rich dashboard + CSV + W&B shim."""

from __future__ import annotations

import csv
import importlib
import logging
from pathlib import Path
from typing import Any

# Full metrics schema. Order matters for the CSV header.
CSV_COLUMNS = [
    "global_step",
    "iteration",
    "total_env_steps",
    "wall_s",
    "loss_total",
    "loss_clip",
    "loss_clip_unclipped",
    "loss_value",
    "loss_entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "grad_norm_global",
    "loss_scale",
    "lr",
    "action_entropy_avg",
    "action_parse_fail_rate",
    "ep_return_mean",
    "ep_return_std",
    "ep_return_min",
    "ep_return_max",
    "ep_return_n",
    "ep_length_mean",
    "ep_length_std",
    "rollout_wall_s",
    "train_wall_s",
    "generate_wall_s",
    "lora_weight_norm_actor",
    "lora_weight_norm_critic",
    "adapter_sync_wall_s",
    "gen_truncated_rate",
    "inv_1_status",
    "inv_3_status",
    "inv_4_status",
    "inv_5_status",
    "inv_6_status",
    "inv_9_status",
    "inv_10_status",
    "inv_11_status",
    "inv_13_status",
]


class CsvWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        self._fh = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict[str, Any]) -> None:
        filled = {k: row.get(k, "") for k in CSV_COLUMNS}
        self._writer.writerow(filled)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def wandb_init(run_name: str, project: str, config: dict[str, Any], enabled: bool):
    if not enabled:
        return None
    wandb = importlib.import_module("wandb")

    return wandb.init(project=project, name=run_name, config=config)


class RichDashboard:
    """Rich-console live dashboard; auto-off in headless / non-TTY."""

    def __init__(self, run_name: str, enabled: bool = True) -> None:
        self.run_name = run_name
        self.enabled = enabled and self._is_tty()

    @staticmethod
    def _is_tty() -> bool:
        import sys

        return sys.stdout.isatty()

    def update(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        logging.getLogger(__name__).info(
            "step=%s ret=%s grad=%s",
            row.get("global_step"),
            row.get("ep_return_mean"),
            row.get("grad_norm_global"),
        )
