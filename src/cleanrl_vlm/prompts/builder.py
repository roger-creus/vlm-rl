"""Prompt template assembly + chat-template application."""

from __future__ import annotations

from pathlib import Path


class PromptBuilder:
    """Loads actor/critic prompt templates for a given ``env_id`` + action names."""

    def __init__(
        self,
        env_id: str,
        action_names: list[str],
        templates_root: Path | None = None,
    ) -> None:
        self.env_id = env_id
        self.action_names = list(action_names)
        self.templates_root = templates_root or (Path(__file__).parent / "templates")
        slug = self._env_id_to_slug(env_id)
        self.actor_template = (self.templates_root / slug / "actor.txt").read_text()
        self.critic_template = (self.templates_root / slug / "critic.txt").read_text()

    @staticmethod
    def _env_id_to_slug(env_id: str) -> str:
        if env_id == "VizdoomBasic-v0":
            return "vizdoom/basic"
        if env_id == "VizdoomCorridor-v0":
            return "vizdoom/corridor"
        if env_id == "VizdoomDefendLine-v0":
            return "vizdoom/defend_line"
        raise KeyError(f"No prompt template slug for {env_id!r}")

    def actor_prompt(self) -> str:
        return self.actor_template

    def critic_prompt(self) -> str:
        return self.critic_template
