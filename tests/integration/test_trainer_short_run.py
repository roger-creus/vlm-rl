"""tier1 @gpu — minimal end-to-end smoke of ``algos/ppo_cot.py`` on VizdoomBasic.

Runs *one iteration* (num_envs=1, num_steps=2 → batch_size=2) with a tiny
``max_new_tokens`` so the full rollout → PPO update → checkpoint flow executes
in a few minutes. Stdout streams to a log file so hangs surface immediately.

Correctness is limited here: we assert only that the subprocess completes
cleanly, writes a runs/ directory, and the CSV header has the reviewer-required
columns. "Genuinely learning" is a Task 23 concern with a longer run.
"""

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

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    cmd = [
        sys.executable,
        "-m",
        "algos.ppo_cot",
        "--env-id",
        "VizdoomBasic-v1",
        "--num-envs",
        "1",
        "--num-steps",
        "2",
        "--total-timesteps",
        "2",
        "--max-new-tokens",
        "16",
        "--num-minibatches",
        "1",
        "--update-epochs",
        "1",
        "--checkpoint-interval",
        "1",
    ]

    log_path = tmp_path / "trainer.log"
    with log_path.open("w") as log:
        r = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=".",
            timeout=1200,
        )

    if r.returncode != 0:
        # Surface the tail of the log on failure.
        tail = log_path.read_text()[-4000:]
        pytest.fail(f"trainer exited {r.returncode}. Log tail:\n{tail}")

    runs = list(Path("runs").glob("ppo_cot__VizdoomBasic-v1__*"))
    assert runs, "no run directory produced"
    run = max(runs, key=lambda p: p.stat().st_mtime)
    csv_path = run / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    body = csv_path.read_text().splitlines()
    assert len(body) > 1, "no metrics rows"
    header = body[0].split(",")
    for col in ["gen_truncated_rate", "lora_weight_norm_actor", "inv_4_status"]:
        assert col in header

    # Inv-4 single-path drift at epoch=0 / minibatch=0 must be green: the
    # re-score path runs under the same actor adapter on the same cached
    # full_ids as the rollout — before any gradient step, ratio must be 1
    # within fp16 noise. Red here means the re-score forward has fp16
    # divergence vs the rollout forward (e.g., padding-induced drift in a
    # batched re-score vs batch=1 rollout).
    row = dict(zip(header, body[1].split(","), strict=True))
    assert row["inv_4_status"] == "green", (
        f"Inv-4 single-path parity red on iter 1. "
        f"approx_kl={row.get('approx_kl')} (mean); re-score path has diverged from rollout."
    )
