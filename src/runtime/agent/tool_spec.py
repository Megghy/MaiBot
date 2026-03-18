from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger
from src.llm_models.payload_content.tool_option import ToolParamType
from src.plugin_system.base.base_tool import BaseTool
from src.runtime.agent.models import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolFailMode,
    ToolParamSpec,
    ToolSource,
)


logger = get_logger("agent_tool_spec")

ToolExecutor = Callable[[ToolExecutionContext, dict[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    params_schema: list[ToolParamSpec]
    executor: ToolExecutor
    source: ToolSource
    tags: set[str] = field(default_factory=set)
    enabled: bool = True
    visible_to_profiles: set[str] = field(default_factory=set)
    timeout_seconds: float = 15.0
    fail_mode: ToolFailMode = "soft_error"

    def to_tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                (
                    param.name,
                    param.param_type,
                    param.description,
                    param.required,
                    param.enum_values,
                )
                for param in self.params_schema
            ],
        }

    async def execute(
        self,
        context: ToolExecutionContext,
        args: dict[str, Any] | None = None,
        *,
        call_id: str | None = None,
    ) -> ToolExecutionResult:
        try:
            raw_result = await self.executor(context, args or {})
            return normalize_tool_result(self.name, raw_result, call_id=call_id)
        except Exception as exc:
            if self.fail_mode == "raise":
                raise
            logger.error(f"工具 {self.name} 执行失败: {exc}")
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                call_id=call_id,
                error=str(exc),
                content=f"工具{self.name}执行失败: {exc}",
            )


def normalize_tool_result(tool_name: str, raw_result: Any, *, call_id: str | None = None) -> ToolExecutionResult:
    if isinstance(raw_result, ToolExecutionResult):
        if raw_result.call_id is None:
            raw_result.call_id = call_id
        return raw_result

    if isinstance(raw_result, dict):
        return ToolExecutionResult(
            tool_name=tool_name,
            content=raw_result.get("content", ""),
            success=raw_result.get("success", True),
            call_id=call_id,
            error=raw_result.get("error"),
            metadata={k: v for k, v in raw_result.items() if k not in {"content", "success", "error"}},
        )

    return ToolExecutionResult(
        tool_name=tool_name,
        content=raw_result,
        success=True,
        call_id=call_id,
    )


def _build_param_specs(
    params: list[tuple[str, ToolParamType, str, bool, list[str] | None]],
) -> list[ToolParamSpec]:
    return [
        ToolParamSpec(
            name=name,
            param_type=param_type,
            description=description,
            required=required,
            enum_values=enum_values,
        )
        for name, param_type, description, required, enum_values in params
    ]


def build_tool_spec_from_plugin_class(
    tool_class: type[BaseTool],
    *,
    source: ToolSource = "plugin",
    tags: set[str] | None = None,
    visible_to_profiles: set[str] | None = None,
    timeout_seconds: float = 15.0,
) -> ToolSpec:
    tool_info = tool_class.get_tool_info()

    async def executor(context: ToolExecutionContext, args: dict[str, Any]) -> Any:
        tool = tool_class(plugin_config=context.plugin_config, chat_stream=context.chat_stream)
        return await tool.execute(args)

    metadata_tags = tool_info.metadata.get("tool_tags", []) if isinstance(tool_info.metadata, dict) else []
    return ToolSpec(
        name=tool_info.name,
        description=tool_info.tool_description or tool_info.description,
        params_schema=_build_param_specs(tool_info.tool_parameters),
        executor=executor,
        source=source,
        tags=set(tags or metadata_tags),
        enabled=bool(tool_info.enabled),
        visible_to_profiles=set(visible_to_profiles or []),
        timeout_seconds=timeout_seconds,
    )


def build_tool_spec_from_memory_tool(
    memory_tool: Any,
    *,
    tags: set[str] | None = None,
    visible_to_profiles: set[str] | None = None,
    timeout_seconds: float = 15.0,
) -> ToolSpec:
    params_schema: list[ToolParamSpec] = []
    type_mapping = {
        "string": ToolParamType.STRING,
        "integer": ToolParamType.INTEGER,
        "int": ToolParamType.INTEGER,
        "float": ToolParamType.FLOAT,
        "boolean": ToolParamType.BOOLEAN,
        "bool": ToolParamType.BOOLEAN,
    }
    for param in getattr(memory_tool, "parameters", []):
        params_schema.append(
            ToolParamSpec(
                name=param.get("name", ""),
                param_type=type_mapping.get(str(param.get("type", "string")).lower(), ToolParamType.STRING),
                description=param.get("description", ""),
                required=bool(param.get("required", False)),
                enum_values=param.get("enum"),
            )
        )

    async def executor(context: ToolExecutionContext, args: dict[str, Any]) -> Any:
        params = dict(args)
        if context.chat_id and "chat_id" not in params:
            params["chat_id"] = context.chat_id
        return await memory_tool.execute(**params)

    return ToolSpec(
        name=str(memory_tool.name),
        description=str(memory_tool.description),
        params_schema=params_schema,
        executor=executor,
        source="memory",
        tags=set(tags or {"memory"}),
        enabled=True,
        visible_to_profiles=set(visible_to_profiles or []),
        timeout_seconds=timeout_seconds,
    )
