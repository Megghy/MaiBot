from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from src.chat.replyer.prompt.replyer_prompt import (
    BASE_REPLY_SAFETY_POLICY,
    build_chat_customization_block,
    build_chat_history_block,
    build_group_identity_block,
    build_group_reply_task,
    build_planner_hint_block,
    build_reply_memory_block,
    build_reply_style_block,
    detect_reply_scene,
)
from src.config.api_ada_configs import TaskConfig
from src.config.config import model_config
from src.llm_models.exceptions import NetworkConnectionError, RespNotOkException
from src.llm_models.utils_model import LLMRequest
from src.runtime.agent import ContextAssembler, PromptComposer, build_agent_profile


@dataclass(frozen=True, slots=True)
class LiveReplyScenario:
    case_id: str
    current_time: str
    dialogue_lines: tuple[str, ...]
    target: str
    sender: str = "yhggeM"
    reply_reason: str = ""
    stale_keywords: tuple[str, ...] = ()


SCENARIOS = [
    LiveReplyScenario(
        case_id="greeting_ignores_stale_meal_context",
        current_time="当前时间：2026-03-19 20:18:00",
        dialogue_lines=(
            "[12:30] yhggeM: 中午吃啥",
            "[12:31] 麦麦(你): 咖喱或者黄焖呗",
            "[15:45] yhggeM: 我中午吃的咖喱鸡饭 还行",
            "[20:18] yhggeM: 晚上好麦麦",
        ),
        target="晚上好麦麦",
        stale_keywords=("咖喱", "黄焖", "鸡饭"),
    ),
    LiveReplyScenario(
        case_id="direct_question_dinner",
        current_time="当前时间：2026-03-19 20:21:00",
        dialogue_lines=(
            "[20:20] 路人甲: 这会儿食堂估计都快关了",
            "[20:21] yhggeM: 晚饭吃什么好",
        ),
        target="晚饭吃什么好",
    ),
    LiveReplyScenario(
        case_id="light_touch_check_in",
        current_time="当前时间：2026-03-19 20:25:00",
        dialogue_lines=(
            "[20:24] 群友A: 这群今天怎么这么安静",
            "[20:25] yhggeM: 打卡",
        ),
        target="打卡",
    ),
    LiveReplyScenario(
        case_id="short_ack_arrived",
        current_time="当前时间：2026-03-19 20:28:00",
        dialogue_lines=(
            "[20:25] 群友B: 到了没",
            "[20:28] yhggeM: 到了",
        ),
        target="到了",
    ),
    LiveReplyScenario(
        case_id="casual_update_not_overconstrained",
        current_time="当前时间：2026-03-19 20:35:00",
        dialogue_lines=(
            "[20:31] 群友C: 外面风好大",
            "[20:32] 麦麦(你): 我刚还想出门来着",
            "[20:35] yhggeM: 我先去洗澡 一会儿再说",
        ),
        target="我先去洗澡 一会儿再说",
    ),
]


def _build_single_model_task(task_name: str, model_name: str) -> TaskConfig:
    base_task = model_config.model_task_config.get_task(task_name)
    return TaskConfig(
        model_list=[model_name],
        max_tokens=base_task.max_tokens,
        temperature=base_task.temperature,
        frequency_penalty=base_task.frequency_penalty,
        presence_penalty=base_task.presence_penalty,
    )


def _build_group_reply_prompts(scenario: LiveReplyScenario) -> tuple[str, str]:
    reply_scene = detect_reply_scene(target=scenario.target, reply_reason=scenario.reply_reason)
    assembled = ContextAssembler().from_parts(
        system_parts={
            "policy": BASE_REPLY_SAFETY_POLICY,
            "identity": build_group_identity_block(
                "你是麦麦，女大学生，说话口语化，群聊里正常接话，不装懂也不端着。",
                "麦麦",
            ),
            "style": build_reply_style_block(
                "\n".join(
                    [
                        "说话像真人在线聊天。",
                        "别把每句话都回得太工整。",
                        "允许偶尔轻轻跑偏，但别像随机生成。",
                    ]
                )
            ),
        },
        user_parts={
            "memory": build_reply_memory_block(
                knowledge_prompt="",
                memory_retrieval="",
                person_memory_block="",
                relation_info_block="",
                extra_info_block="",
                expression_habits_block="",
                actions_info="",
            ),
            "chat_history": build_chat_history_block(
                time_block=scenario.current_time,
                dialogue_prompt="\n".join(scenario.dialogue_lines),
            ),
            "reply_target": f"现在{scenario.sender}说的：{scenario.target}。引起了你的注意",
            "planner_hint": build_planner_hint_block("正常接一下当前这句，别被太早之前的话带跑。"),
            "mood_hint": "当前状态：随意聊天，不用端着。",
            "chat_customization": build_chat_customization_block(chat_prompt_block="", keywords_reaction_prompt=""),
            "task": build_group_reply_task(reply_scene=reply_scene),
        },
    )

    profile = build_agent_profile("reply_default")
    composer = PromptComposer()
    prepared = composer.prepare(assembled.blocks, profile)
    composed = composer.compose(prepared.blocks)
    return composed.system_prompt, composed.user_prompt


def _assert_reply_shape(reply: str) -> None:
    assert reply.strip(), "回复不能为空"
    assert "Thinking Process" not in reply
    assert "```" not in reply

    for line in reply.splitlines():
        assert not re.match(r"^\s*[-*]\s", line), f"不应输出项目符号: {reply!r}"
        assert not re.match(r"^\s*\d+\.\s", line), f"不应输出编号列表: {reply!r}"
        assert not re.match(r"^\s*(回复|分析|解释|角色)\s*[:：]", line), f"不应输出说明性前缀: {reply!r}"


@pytest.mark.live_model
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.case_id for scenario in SCENARIOS])
async def test_live_replyer_group_scenarios(live_replyer_model_name: str, scenario: LiveReplyScenario):
    system_prompt, user_prompt = _build_group_reply_prompts(scenario)
    request = LLMRequest(
        model_set=_build_single_model_task("replyer", live_replyer_model_name),
        request_type=f"live_test.replyer.{scenario.case_id}",
    )

    try:
        response, (reasoning, used_model, tool_calls) = await request.generate_response_async(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=160,
        )
    except NetworkConnectionError as exc:
        pytest.skip(f"live provider unavailable for {live_replyer_model_name}: {exc}")
    except RespNotOkException as exc:
        if exc.status_code in {429, 500, 503}:
            pytest.skip(f"live provider unavailable for {live_replyer_model_name}: {exc}")
        raise

    print(f"\n[model={used_model}] case={scenario.case_id}")
    print(f"target: {scenario.target}")
    print(f"reply: {response}")
    if reasoning.strip():
        print(f"reasoning_len: {len(reasoning)}")

    assert used_model == live_replyer_model_name
    assert tool_calls in (None, [])
    _assert_reply_shape(response)

    for keyword in scenario.stale_keywords:
        assert keyword not in response, f"问候场景不该翻旧账带出过期关键词: {keyword} | reply={response!r}"
