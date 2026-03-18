from __future__ import annotations

import pytest

from src.chat.brain_chat.brain_planner import BrainPlanner
from src.chat.planner_actions.planner_runtime import PreparedPlannerExecution


class FakePlannerLLM:
    async def generate_response_async(self, *, prompt, tools, system_prompt=None):
        del prompt, tools, system_prompt
        return '{"name":"no_reply","arguments":{"reason":"先观察"}}', (None, None, None)


@pytest.mark.asyncio
async def test_brain_planner_uses_shared_json_fallback():
    planner = BrainPlanner.__new__(BrainPlanner)
    planner.log_prefix = "[test]"
    planner.planner_llm = FakePlannerLLM()

    actions = await planner._execute_main_planner(
        prompt="prompt",
        message_id_list=[],
        filtered_actions={},
        available_actions={},
        loop_start_time=12.5,
    )

    assert len(actions) == 1
    assert actions[0].action_type == "no_reply"
    assert "先观察" in (actions[0].reasoning or "")
    assert actions[0].action_data["loop_start_time"] == 12.5


@pytest.mark.asyncio
async def test_brain_planner_plan_delegates_to_prepared_execution():
    planner = BrainPlanner.__new__(BrainPlanner)
    prepared_plan = PreparedPlannerExecution(prompt="prompt", message_id_list=[], filtered_actions={})

    async def prepare_plan_context(available_actions):
        assert available_actions == {"reply": "ok"}
        return prepared_plan

    async def execute_prepared_plan(*, prepared_plan, available_actions, loop_start_time):
        assert prepared_plan.prompt == "prompt"
        assert available_actions == {"reply": "ok"}
        assert loop_start_time == 3.0
        return ["done"]

    planner.prepare_plan_context = prepare_plan_context
    planner.execute_prepared_plan = execute_prepared_plan

    result = await planner.plan({"reply": "ok"}, loop_start_time=3.0)

    assert result == ["done"]
