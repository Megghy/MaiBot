from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.llm_models.payload_content.tool_option import ToolParamType
from src.runtime.agent.models import ToolExecutionContext, ToolExecutionResult, ToolParamSpec
from src.runtime.agent.tool_spec import ToolSpec


@dataclass(slots=True)
class McpToolDescriptor:
    name: str
    description: str
    parameters: list[ToolParamSpec] = field(default_factory=list)
    tags: set[str] = field(default_factory=lambda: {"mcp"})
    timeout_seconds: float = 20.0


class McpAdapter:
    def __init__(self, executor) -> None:
        self.executor = executor

    def build_tool_spec(self, descriptor: McpToolDescriptor) -> ToolSpec:
        async def _execute(context: ToolExecutionContext, args: dict[str, Any]) -> ToolExecutionResult:
            result = await self.executor(descriptor.name, context, args)
            if isinstance(result, ToolExecutionResult):
                return result
            return ToolExecutionResult(tool_name=descriptor.name, content=result)

        return ToolSpec(
            name=descriptor.name,
            description=descriptor.description,
            params_schema=descriptor.parameters,
            executor=_execute,
            source="mcp",
            tags=set(descriptor.tags or {"mcp"}),
            timeout_seconds=descriptor.timeout_seconds,
        )

    def build_many(self, descriptors: Iterable[McpToolDescriptor]) -> list[ToolSpec]:
        return [self.build_tool_spec(descriptor) for descriptor in descriptors]


def build_param(
    name: str,
    param_type: ToolParamType,
    description: str,
    *,
    required: bool = True,
    enum_values: list[str] | None = None,
) -> ToolParamSpec:
    return ToolParamSpec(
        name=name,
        param_type=param_type,
        description=description,
        required=required,
        enum_values=enum_values,
    )
