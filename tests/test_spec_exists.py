"""Verify the master spec file exists and is non-empty.

Motivation: the spec is the single source of truth for the /loop agent; if
it goes missing, every subsequent cycle is compromised. This test is a
cheap tripwire.
"""

from pathlib import Path

SPEC_PATH = Path("docs/superpowers/specs/2026-04-19-cleanrl-vlm-masterplan.md")


def test_spec_exists() -> None:
    assert SPEC_PATH.exists(), f"master spec missing at {SPEC_PATH}"


def test_spec_non_empty() -> None:
    assert (
        SPEC_PATH.stat().st_size > 1000
    ), f"master spec suspiciously small ({SPEC_PATH.stat().st_size} bytes); expected > 1000"


def test_spec_has_autonomy_section() -> None:
    content = SPEC_PATH.read_text(encoding="utf-8")
    assert "## §13. Autonomous operation contract" in content
    assert "Never stop" in content
    assert "Never ask the user anything" in content


def test_claude_md_carries_autonomy_section() -> None:
    claude_md = Path("CLAUDE.md")
    assert claude_md.exists(), "CLAUDE.md missing"
    content = claude_md.read_text(encoding="utf-8")
    assert "AUTONOMOUS OPERATION CONTRACT" in content
    assert "Never stop" in content
    assert "Never ask the user anything" in content
