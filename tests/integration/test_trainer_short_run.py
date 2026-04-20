"""tier1 @gpu — minimal end-to-end smoke of ``algos/ppo_cot.py``.

Runs *one iteration* (num_envs=1, num_steps=2 → batch_size=2) with a tiny
``max_new_tokens`` so the full rollout → PPO update → checkpoint flow executes
in a few minutes. Stdout streams to a log file so hangs surface immediately.

Parametrized across Tier-1 envs to exercise the env-registry dispatcher and
per-env prompt templates end-to-end. Correctness is limited here: we assert
the subprocess completes cleanly, writes a runs/ directory, Inv-4 stays green
on the first minibatch, and gradients are finite. "Genuinely learning" is a
Tier-2 campaign concern.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.tier1, pytest.mark.gpu]


TIER1_ENVS = [
    pytest.param("VizdoomBasic-v1", "configs/envs/VizdoomBasic-v1.yaml", id="vizdoom_basic"),
    pytest.param("ALE/Pong-v5", "configs/envs/ALE-Pong-v5.yaml", id="ale_pong"),
]


@pytest.mark.parametrize(("env_id", "env_config"), TIER1_ENVS)
def test_ppo_cot_short_run(tmp_path: Path, env_id: str, env_config: str):
    if not (os.environ.get("GPU_AVAILABLE") or os.path.exists("/dev/nvidia0")):
        pytest.skip("no GPU present")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(".").resolve())
    cmd = [
        sys.executable,
        "-m",
        "algos.ppo_cot",
        "--env-id",
        env_id,
        "--env-config",
        env_config,
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
        tail = log_path.read_text()[-4000:]
        pytest.fail(f"trainer exited {r.returncode}. Log tail:\n{tail}")

    env_slug = env_id.replace("/", "-")
    runs = list(Path("runs").glob(f"ppo_cot__{env_slug}__*"))
    assert runs, f"no run directory produced for {env_id}"
    run = max(runs, key=lambda p: p.stat().st_mtime)
    csv_path = run / "metrics.csv"
    assert csv_path.exists(), "metrics.csv missing"
    body = csv_path.read_text().splitlines()
    assert len(body) > 1, "no metrics rows"
    header = body[0].split(",")
    for col in ["gen_truncated_rate", "lora_weight_norm_actor", "inv_4_status"]:
        assert col in header

    # Read the LAST metrics row (CsvWriter opens in append mode — re-running
    # the trainer under the same run_name adds rows rather than truncating,
    # which matches the §10 resume story but means the latest run's data is
    # always at the tail).
    row = dict(zip(header, body[-1].split(","), strict=True))
    assert row["inv_4_status"] == "green", (
        f"[{env_id}] Inv-4 single-path parity red on iter 1. "
        f"approx_kl={row.get('approx_kl')} (mean); re-score path has diverged from rollout."
    )
    # Gradients must be finite. Qwen3.5's Gated DeltaNet backward produces
    # NaNs under FP16 without the flash-linear-attention fast path; the
    # BF16 default (amendment 2026-04-20-bf16-default-for-qwen3.5.md)
    # avoids this.
    grad_norm = row.get("grad_norm_global", "")
    assert grad_norm and grad_norm.lower() not in {
        "nan",
        "inf",
        "-inf",
    }, f"[{env_id}] grad_norm_global not finite: {grad_norm!r}."
