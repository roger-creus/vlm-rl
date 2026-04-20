"""tier1 @gpu — 10-iter short run of ``algos/ppo_cot.py`` on VizdoomBasic."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.gpu]


def test_ppo_cot_short_run(tmp_path: Path):
    if not (os.environ.get("GPU_AVAILABLE") or os.path.exists("/dev/nvidia0")):
        pytest.skip("no GPU present")

    # 10 iterations * num_envs=2 * num_steps=8 = 160 env steps.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    cmd = [
        sys.executable,
        "-m",
        "algos.ppo_cot",
        "--env-id",
        "VizdoomBasic-v0",
        "--num-envs",
        "2",
        "--num-steps",
        "8",
        "--total-timesteps",
        "160",
        "--max-new-tokens",
        "64",
        "--num-minibatches",
        "2",
        "--update-epochs",
        "1",
        "--checkpoint-interval",
        "10",
    ]
    r = subprocess.run(cmd, capture_output=True, env=env, cwd=".", timeout=1800)
    assert r.returncode == 0, r.stderr.decode()[-4000:]

    runs = list(Path("runs").glob("ppo_cot__VizdoomBasic-v0__*"))
    assert runs, "no run directory produced"
    run = max(runs, key=lambda p: p.stat().st_mtime)
    csv_path = run / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    body = csv_path.read_text().splitlines()
    assert len(body) > 1, "no metrics rows"
    # Header contains reviewer-required columns.
    header = body[0].split(",")
    for col in ["gen_truncated_rate", "lora_weight_norm_actor", "inv_4_status"]:
        assert col in header
