from __future__ import annotations

from types import SimpleNamespace

from src.chat.planner_actions.planner_prompting import build_reply_stats_block, format_time_delta
from src.chat.planner_actions.planner_selection import (
    filter_actions_by_activation_type,
    select_actions_for_prompt,
)
from src.chat.planner_actions.planner_tools import parse_single_tool_call, try_parse_json_fallback
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.llm_models.payload_content.tool_option import ToolCall
from src.plugin_system.base.component_types import ActionActivationType, ActionInfo, ComponentType


def _action_info(
    name: str,
    *,
    activation_type: ActionActivationType = ActionActivationType.ALWAYS,
    activation_keywords: list[str] | None = None,
    metadata: dict | None = None,
    parallel_action: bool = False,
) -> ActionInfo:
    return ActionInfo(
        name=name,
        component_type=ComponentType.ACTION,
        description=f"{name} desc",
        activation_type=activation_type,
        activation_keywords=activation_keywords or [],
        metadata=metadata or {},
        parallel_action=parallel_action,
    )


def test_filter_actions_by_activation_type_respects_never_and_keyword(monkeypatch):
    monkeypatch.setattr("src.chat.planner_actions.planner_selection.random.random", lambda: 0.0)
    actions = {
        "always": _action_info("always"),
        "never": _action_info("never", activation_type=ActionActivationType.NEVER),
        "keyword": _action_info(
            "keyword",
            activation_type=ActionActivationType.KEYWORD,
            activation_keywords=["触发词"],
        ),
        "random": _action_info(
            "random",
            activation_type=ActionActivationType.RANDOM,
            metadata={"planner_priority": 1.0},
        ),
    }

    filtered = filter_actions_by_activation_type(actions, "这里有触发词")

    assert set(filtered) == {"always", "keyword"}


def test_select_actions_for_prompt_uses_score_and_limit():
    actions = {
        "high": _action_info("high", metadata={"planner_priority": 3}),
        "keyword": _action_info(
            "keyword",
            activation_type=ActionActivationType.KEYWORD,
            activation_keywords=["重要"],
            metadata={"planner_priority": 2},
        ),
        "low": _action_info("low", activation_type=ActionActivationType.RANDOM),
    }

    selected = select_actions_for_prompt(actions, "这条消息很重要", max_dynamic_action_tools=2)

    assert set(selected.keys()) == {"keyword", "high"}


def test_parse_single_tool_call_replaces_message_ids_in_reasoning():
    message = SimpleNamespace(processed_plain_text="你好呀", display_message="你好呀")
    tool_call = ToolCall(
        call_id="call-1",
        func_name="reply",
        args={"target_message_id": "m1", "reason": "回应 [m1]"},
    )

    actions = parse_single_tool_call(tool_call, [("m1", message)], [], log_prefix="[test]")

    assert len(actions) == 1
    assert actions[0].action_type == "reply"
    assert "回应 [你好呀]" in actions[0].reasoning
    assert "目标消息: 你好呀" in actions[0].reasoning


def test_try_parse_json_fallback_supports_arguments_payload():
    parsed = try_parse_json_fallback(
        '{"name":"no_reply","arguments":{"reason":"先观察"}}',
        {},
        [],
        "planner reasoning",
        log_prefix="[test]",
    )

    assert parsed is not None
    assert parsed[0].action_type == "no_reply"
    assert "先观察" in (parsed[0].reasoning or "")


def test_build_reply_stats_block_and_time_delta():
    plan_log = [
        (
            "reason",
            0.0,
            [ActionPlannerInfo(action_type="reply", reasoning="ok", action_data={}, action_message=None)],
        )
    ]

    assert format_time_delta(3661) == "1小时1分钟"
    assert "你还没有回复过消息" in build_reply_stats_block([], window_seconds=600)
    assert "已经回复了1条消息" in build_reply_stats_block(plan_log, window_seconds=10**10)
