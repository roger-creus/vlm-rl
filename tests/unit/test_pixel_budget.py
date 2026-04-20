"""Pixel budget derives from the env's observation_space by default.

Single-frame envs (frame_stack.n=1) should get a processor budget equal to
H*W of the actual obs — no silent upscale, no silent downscale. YAML may
override by uncommenting ``processor_min_pixels`` / ``processor_max_pixels``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from gymnasium.vector import AsyncVectorEnv

from cleanrl_vlm.envs.registry import make_env


def _native_pixels_for(env_id: str, cfg_path: str) -> tuple[int, int]:
    """Spin up 1 env instance, return (H*W, H*W*C) from single_observation_space."""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    envs = AsyncVectorEnv([make_env(env_id, cfg, seed=0, idx=0, run_name="_unit_pixels")])
    shape = envs.single_observation_space.shape
    envs.close()
    assert shape is not None
    return int(shape[0]) * int(shape[1]), int(shape[0]) * int(shape[1]) * int(shape[2])


def test_vizdoom_basic_single_frame_native_pixels():
    # 240x320x3 = 76800 px
    pixels, pixels_with_c = _native_pixels_for("VizdoomBasic-v1", "configs/envs/VizdoomBasic-v1.yaml")
    assert pixels == 240 * 320 == 76800
    assert pixels_with_c == 76800 * 3


def test_ale_pong_single_frame_native_pixels():
    # 210x160x3 = 33600 px
    pixels, pixels_with_c = _native_pixels_for("ALE/Pong-v5", "configs/envs/ALE-Pong-v5.yaml")
    assert pixels == 210 * 160 == 33600
    assert pixels_with_c == 33600 * 3


def test_vizdoom_basic_yaml_has_no_hardcoded_pixel_budget():
    """Assert the pixel-budget entries are commented out (adaptive default).

    Leaving them hardcoded re-introduces the drift risk between actual obs
    dims and what the processor sees (silent upscale / downscale).
    """
    raw = Path("configs/envs/VizdoomBasic-v1.yaml").read_text()
    cfg = yaml.safe_load(raw)
    assert "processor_min_pixels" not in cfg, "pixel budget should be commented out; adaptive path handles it"
    assert "processor_max_pixels" not in cfg


def test_ale_pong_yaml_has_no_hardcoded_pixel_budget():
    raw = Path("configs/envs/ALE-Pong-v5.yaml").read_text()
    cfg = yaml.safe_load(raw)
    assert "processor_min_pixels" not in cfg
    assert "processor_max_pixels" not in cfg
