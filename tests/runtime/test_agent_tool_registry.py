from __future__ import annotations

import pytest

from src.llm_models.payload_content.tool_option import ToolParamType
from src.plugin_system.base.base_tool import BaseTool
from src.runtime.agent import AgentMode, AgentProfile, ToolExecutionContext, ToolRegistry


class FakePluginTool(BaseTool):
    name = "plugin_lookup"
    description = "Lookup plugin data"
    parameters = [("query", ToolParamType.STRING, "query text", True, None)]
    available_for_llm = True

    async def execute(self, function_args: dict[str, str]) -> dict[str, str]:
        return {
            "type": "plugin_result",
            "content": f"plugin:{self.chat_id}:{function_args['query']}",
        }


class FakeMemoryTool:
    name = "query_memory"
    description = "Query memory entries"
    parameters = [{"name": "query", "type": "string", "description": "query text", "required": True}]

    async def execute(self, query: str, chat_id: str) -> str:
        return f"memory:{chat_id}:{query}"


@pytest.mark.asyncio
async def test_tool_registry_adapts_plugin_and_memory_tools(fake_chat_stream):
    registry = ToolRegistry()
    registry.register_plugin_tool_class(FakePluginTool, tags={"plugin"})
    registry.register_memory_tool(FakeMemoryTool(), tags={"memory"})

    profile = AgentProfile(
        name="memory-profile",
        skill_name="memory_retrieval",
        agent_mode=AgentMode.TOOL_LOOP,
        model_task="tool_use",
        allowed_tool_tags={"plugin", "memory"},
    )
    definitions = registry.get_tool_definitions(profile)

    assert [item["name"] for item in definitions] == ["plugin_lookup", "query_memory"]

    context = ToolExecutionContext(chat_id="chat-1", chat_stream=fake_chat_stream)
    plugin_spec = registry.get("plugin_lookup")
    memory_spec = registry.get("query_memory")

    assert plugin_spec is not None
    assert memory_spec is not None

    plugin_result = await plugin_spec.execute(context, {"query": "hello"}, call_id="call-1")
    memory_result = await memory_spec.execute(context, {"query": "world"}, call_id="call-2")

    assert plugin_result.success is True
    assert plugin_result.content == "plugin:stream-1:hello"
    assert memory_result.success is True
    assert memory_result.content == "memory:chat-1:world"
