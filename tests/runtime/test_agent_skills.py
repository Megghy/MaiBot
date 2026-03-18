from __future__ import annotations

from src.runtime.agent import AgentMode, OutputMode, build_agent_profile, get_skill_registry, init_default_skills


def test_build_agent_profile_uses_default_skill_registry():
    registry = get_skill_registry()
    registry.clear()
    init_default_skills()

    profile = build_agent_profile("planner_default")

    assert profile.skill_name == "planner_default"
    assert profile.agent_mode == AgentMode.SINGLE_PASS
    assert profile.output_mode == OutputMode.ACTION_SELECTION
    assert "planner" in profile.allowed_tool_tags
