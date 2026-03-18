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


DEFAULT_SKILLS = (
    SkillProfile(
        name="reply_default",
        description="Default reply generation profile",
        agent_mode=AgentMode.SINGLE_PASS,
        model_task="replyer",
        enabled_context_blocks=["identity", "context", "memory", "output_contract"],
        allowed_tool_tags={"reply", "knowledge"},
        output_mode=OutputMode.REPLY_TEXT,
        memory_policy=MemoryPolicy.LIGHT,
        mcp_policy=McpPolicy.OFF,
        max_iterations=1,
    ),
    SkillProfile(
        name="planner_default",
        description="Planner action selection profile",
        agent_mode=AgentMode.SINGLE_PASS,
        model_task="planner",
        enabled_context_blocks=["policy", "task", "context", "output_contract"],
        allowed_tool_tags={"action", "planner"},
        output_mode=OutputMode.ACTION_SELECTION,
        memory_policy=MemoryPolicy.OFF,
        mcp_policy=McpPolicy.OFF,
        max_iterations=1,
    ),
    SkillProfile(
        name="memory_retrieval",
        description="Multi-iteration memory retrieval profile",
        agent_mode=AgentMode.TOOL_LOOP,
        model_task="tool_use",
        enabled_context_blocks=["task", "context", "memory", "output_contract"],
        allowed_tool_tags={"memory"},
        output_mode=OutputMode.QA_ANSWER,
        memory_policy=MemoryPolicy.FULL,
        mcp_policy=McpPolicy.OFF,
        max_iterations=4,
        terminal_tools={"found_answer", "not_enough_info"},
    ),
    SkillProfile(
        name="knowledge_qa",
        description="Knowledge QA profile with MCP-enabled tool access",
        agent_mode=AgentMode.TOOL_LOOP,
        model_task="tool_use",
        enabled_context_blocks=["policy", "task", "context", "tool_observations", "output_contract"],
        allowed_tool_tags={"knowledge", "mcp"},
        output_mode=OutputMode.QA_ANSWER,
        memory_policy=MemoryPolicy.OFF,
        mcp_policy=McpPolicy.ALLOW_LIST,
        max_iterations=4,
    ),
    SkillProfile(
        name="tool_only",
        description="Tool selection only profile",
        agent_mode=AgentMode.SINGLE_PASS,
        model_task="tool_use",
        enabled_context_blocks=["task", "context", "output_contract"],
        allowed_tool_tags={"tool", "memory", "knowledge", "mcp"},
        output_mode=OutputMode.TOOL_ONLY,
        memory_policy=MemoryPolicy.OFF,
        mcp_policy=McpPolicy.AUTO,
        max_iterations=1,
    ),
)


def init_default_skills() -> None:
    registry = get_skill_registry()
    for skill in DEFAULT_SKILLS:
        registry.register(skill)


def build_agent_profile(skill_name: str, **overrides) -> AgentProfile:
    registry = get_skill_registry()
    if not registry.all():
        init_default_skills()
    skill = registry.get(skill_name)
    if skill is None:
        raise KeyError(f"Skill profile '{skill_name}' not found")
    return AgentProfile.from_skill(skill, overrides=overrides)
