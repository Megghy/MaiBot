from __future__ import annotations

import pytest

from src.chat.planner_actions.planner_runtime import PreparedPlannerExecution, execute_planner_llm


class FakePlannerLLM:
    async def generate_response_async(self, *, prompt, tools, system_prompt=None):
        del prompt, tools, system_prompt
        return '{"name":"no_reply","arguments":{"reason":"继续观察"}}', (None, "fake-model", None)


@pytest.mark.asyncio
async def test_execute_planner_llm_supports_json_fallback():
    reasoning, actions = await execute_planner_llm(
        FakePlannerLLM(),
        PreparedPlannerExecution(prompt="prompt", message_id_list=[], filtered_actions={}),
        available_actions={},
        loop_start_time=7.0,
        log_prefix="[test]",
    )

    assert reasoning
    assert len(actions) == 1
    assert actions[0].action_type == "no_reply"
    assert "继续观察" in (actions[0].reasoning or "")
    assert actions[0].action_data["loop_start_time"] == 7.0
