"""Verify that the scaffolded package and its submodules import cleanly.

Follows master-spec §0: binary correctness (either imports work or they don't).
"""

import importlib

import pytest

SUBMODULES = [
    "cleanrl_vlm",
    "cleanrl_vlm.envs",
    "cleanrl_vlm.models",
    "cleanrl_vlm.prompts",
    "cleanrl_vlm.rollout",
    "cleanrl_vlm.training",
    "cleanrl_vlm.research",
]


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodule_imports(name: str) -> None:
    mod = importlib.import_module(name)
    assert mod is not None, f"importlib returned None for {name!r}"


def test_version_string() -> None:
    import cleanrl_vlm

    assert isinstance(cleanrl_vlm.__version__, str)
    assert cleanrl_vlm.__version__ == "0.0.1"
