from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.plugin_system.base.base_tool import BaseTool
from src.runtime.agent.models import AgentProfile
from src.runtime.agent.tool_spec import (
    ToolSpec,
    build_tool_spec_from_memory_tool,
    build_tool_spec_from_plugin_class,
)


@dataclass(slots=True)
class ToolRegistry:
    specs: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec, *, overwrite: bool = False) -> ToolSpec:
        if spec.name in self.specs and not overwrite:
            return self.specs[spec.name]
        self.specs[spec.name] = spec
        return spec

    def register_many(self, specs: Iterable[ToolSpec], *, overwrite: bool = False) -> list[ToolSpec]:
        return [self.register(spec, overwrite=overwrite) for spec in specs]

    def register_plugin_tool_class(
        self,
        tool_class: type[BaseTool],
        *,
        overwrite: bool = False,
        tags: set[str] | None = None,
        visible_to_profiles: set[str] | None = None,
    ) -> ToolSpec:
        return self.register(
            build_tool_spec_from_plugin_class(
                tool_class,
                tags=tags,
                visible_to_profiles=visible_to_profiles,
            ),
            overwrite=overwrite,
        )

    def register_memory_tool(
        self,
        memory_tool: Any,
        *,
        overwrite: bool = False,
        tags: set[str] | None = None,
        visible_to_profiles: set[str] | None = None,
    ) -> ToolSpec:
        return self.register(
            build_tool_spec_from_memory_tool(
                memory_tool,
                tags=tags,
                visible_to_profiles=visible_to_profiles,
            ),
            overwrite=overwrite,
        )

    def get(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def all(self) -> dict[str, ToolSpec]:
        return self.specs.copy()

    def resolve_for_profile(self, profile: AgentProfile | None = None) -> list[ToolSpec]:
        specs = [spec for spec in self.specs.values() if spec.enabled]
        if profile is None:
            return sorted(specs, key=lambda spec: spec.name)

        if profile.blocked_tools:
            specs = [spec for spec in specs if spec.name not in profile.blocked_tools]

        if profile.allowed_tool_tags:
            specs = [spec for spec in specs if spec.tags & profile.allowed_tool_tags]

        if profile.name:
            specs = [
                spec
                for spec in specs
                if not spec.visible_to_profiles or profile.name in spec.visible_to_profiles or profile.skill_name in spec.visible_to_profiles
            ]

        return sorted(specs, key=lambda spec: spec.name)

    def get_tool_definitions(self, profile: AgentProfile | None = None) -> list[dict[str, Any]]:
        return [spec.to_tool_definition() for spec in self.resolve_for_profile(profile)]

    def clear(self) -> None:
        self.specs.clear()


agent_tool_registry = ToolRegistry()


def get_agent_tool_registry() -> ToolRegistry:
    return agent_tool_registry
