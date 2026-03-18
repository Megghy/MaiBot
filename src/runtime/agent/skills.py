from __future__ import annotations

from dataclasses import dataclass, field

from src.runtime.agent.models import SkillProfile


@dataclass(slots=True)
class SkillRegistry:
    skills: dict[str, SkillProfile] = field(default_factory=dict)

    def register(self, skill: SkillProfile, *, overwrite: bool = False) -> SkillProfile:
        if skill.name in self.skills and not overwrite:
            return self.skills[skill.name]
        self.skills[skill.name] = skill
        return skill

    def get(self, name: str) -> SkillProfile | None:
        return self.skills.get(name)

    def all(self) -> dict[str, SkillProfile]:
        return self.skills.copy()

    def clear(self) -> None:
        self.skills.clear()


skill_registry = SkillRegistry()


def get_skill_registry() -> SkillRegistry:
    return skill_registry
