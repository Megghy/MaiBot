from __future__ import annotations

from src.runtime.agent.models import (
    AgentMode,
    AgentProfile,
    McpPolicy,
    MemoryPolicy,
    OutputMode,
    SkillProfile,
)
from src.runtime.agent.skills import get_skill_registry


DEFAULT_SKILL_DEFINITIONS = (
    {
        "name": "reply_default",
        "description": "Default reply generation profile",
        "agent_mode": AgentMode.SINGLE_PASS,
        "model_task": "replyer",
        "enabled_context_blocks": ["identity", "context", "memory", "output_contract"],
        "allowed_tool_tags": {"reply", "knowledge"},
        "output_mode": OutputMode.REPLY_TEXT,
        "memory_policy": MemoryPolicy.LIGHT,
        "mcp_policy": McpPolicy.OFF,
        "max_iterations": 1,
    },
    {
        "name": "planner_default",
        "description": "Planner action selection profile",
        "agent_mode": AgentMode.SINGLE_PASS,
        "model_task": "planner",
        "enabled_context_blocks": ["policy", "task", "context", "output_contract"],
        "allowed_tool_tags": {"action", "planner"},
        "output_mode": OutputMode.ACTION_SELECTION,
        "memory_policy": MemoryPolicy.OFF,
        "mcp_policy": McpPolicy.OFF,
        "max_iterations": 1,
    },
    {
        "name": "memory_retrieval",
        "description": "Multi-iteration memory retrieval profile",
        "agent_mode": AgentMode.TOOL_LOOP,
        "model_task": "tool_use",
        "enabled_context_blocks": ["task", "context", "memory", "output_contract"],
        "allowed_tool_tags": {"memory"},
        "output_mode": OutputMode.QA_ANSWER,
        "memory_policy": MemoryPolicy.FULL,
        "mcp_policy": McpPolicy.OFF,
        "max_iterations": 4,
        "terminal_tools": {"found_answer", "not_enough_info"},
    },
    {
        "name": "knowledge_qa",
        "description": "Knowledge QA profile with MCP-enabled tool access",
        "agent_mode": AgentMode.TOOL_LOOP,
        "model_task": "tool_use",
        "enabled_context_blocks": ["policy", "task", "context", "tool_observations", "output_contract"],
        "allowed_tool_tags": {"knowledge", "mcp"},
        "output_mode": OutputMode.QA_ANSWER,
        "memory_policy": MemoryPolicy.OFF,
        "mcp_policy": McpPolicy.ALLOW_LIST,
        "max_iterations": 4,
    },
    {
        "name": "tool_only",
        "description": "Tool selection only profile",
        "agent_mode": AgentMode.SINGLE_PASS,
        "model_task": "tool_use",
        "enabled_context_blocks": ["task", "context", "output_contract"],
        "allowed_tool_tags": {"tool", "memory", "knowledge", "mcp"},
        "output_mode": OutputMode.TOOL_ONLY,
        "memory_policy": MemoryPolicy.OFF,
        "mcp_policy": McpPolicy.AUTO,
        "max_iterations": 1,
    },
)


def _build_skill_profile(skill_definition: dict) -> SkillProfile:
    return SkillProfile(
        name=skill_definition["name"],
        description=skill_definition.get("description", ""),
        agent_mode=skill_definition.get("agent_mode", AgentMode.SINGLE_PASS),
        model_task=skill_definition.get("model_task", "replyer"),
        enabled_context_blocks=list(skill_definition.get("enabled_context_blocks", [])),
        allowed_tool_tags=set(skill_definition.get("allowed_tool_tags", set())),
        blocked_tools=set(skill_definition.get("blocked_tools", set())),
        max_iterations=skill_definition.get("max_iterations", 1),
        timeout_seconds=skill_definition.get("timeout_seconds", 30.0),
        output_mode=skill_definition.get("output_mode", OutputMode.REPLY_TEXT),
        memory_policy=skill_definition.get("memory_policy", MemoryPolicy.OFF),
        mcp_policy=skill_definition.get("mcp_policy", McpPolicy.OFF),
        allow_plain_text_fallback=skill_definition.get("allow_plain_text_fallback", True),
        terminal_tools=set(skill_definition.get("terminal_tools", set())),
    )


def init_default_skills() -> None:
    registry = get_skill_registry()
    for skill_definition in DEFAULT_SKILL_DEFINITIONS:
        registry.register(_build_skill_profile(skill_definition))


def build_agent_profile(skill_name: str, **overrides) -> AgentProfile:
    registry = get_skill_registry()
    if not registry.all():
        init_default_skills()
    skill = registry.get(skill_name)
    if skill is None:
        raise KeyError(f"Skill profile '{skill_name}' not found")
    return AgentProfile.from_skill(skill, overrides=overrides)
