from __future__ import annotations

import pytest

from src.llm_models.payload_content.tool_option import ToolParamType
from src.runtime.agent import McpAdapter, McpToolDescriptor, ToolExecutionContext, ToolRegistry
from src.runtime.agent.mcp_adapter import build_param


@pytest.mark.asyncio
async def test_mcp_adapter_registers_tools_into_unified_registry():
    async def fake_executor(tool_name, context, args):
        assert tool_name == "mcp_search"
        assert context.chat_id == "chat-1"
        return f"mcp:{args['query']}"

    adapter = McpAdapter(fake_executor)
    descriptor = McpToolDescriptor(
        name="mcp_search",
        description="Search over MCP source",
        parameters=[
            build_param("query", ToolParamType.STRING, "query text"),
        ],
    )
    registry = ToolRegistry()
    registry.register(adapter.build_tool_spec(descriptor))

    spec = registry.get("mcp_search")
    assert spec is not None
    context = ToolExecutionContext(chat_id="chat-1")
    result = await spec.execute(context, {"query": "hello"})

    assert result.success is True
    assert result.content == "mcp:hello"
