from datetime import datetime
from typing import List

import pytest

from src.chat.message_receive.message import SessionMessage
from src.chat.replyer.maisaka_generator_base import BaseMaisakaReplyGenerator
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.llm_models.payload_content.context_item import ContextItemBuilder, RoleType, get_item_text
from src.maisaka.context.messages import (
    ComplexSessionMessage,
    ContentContextMessage,
    InternalContextMessage,
    LLMContextMessage,
    ModelOutputContextMessage,
    ReferenceMessage,
    SessionBackedMessage,
    ToolResultMediaMessage,
    ToolResultMessage,
)
from src.maisaka.context.compaction import ContextSummaryMessage


def _build_mixed_history() -> List[LLMContextMessage]:
    timestamp = datetime(2026, 10, 19, 12)
    history: List[LLMContextMessage] = []
    for index, (message_type, source, text) in enumerate([
        (SessionBackedMessage, "user", "真实用户发言"),
        (SessionBackedMessage, "guided_reply", "机器人已发送回复"),
        (ComplexSessionMessage, "user", "真实转发消息"),
    ]):
        sequence = MessageSequence([TextComponent(text)])
        original = SessionMessage(message_id=str(index), timestamp=timestamp, platform="test")
        history.append(message_type(
            raw_message=sequence,
            visible_text=text,
            timestamp=timestamp,
            message_id=str(index),
            original_message=original,
            source_kind=source,
        ))
        if isinstance(history[-1], ComplexSessionMessage):
            history[-1].prompt_text = text
    for source in (
        "subagent_result", "plugin_proactive:example", "focus_at_wakeup",
        "focus_cooldown_wakeup", "focus_switch", "internal", "future_internal_type",
    ):
        history.append(InternalContextMessage(
            raw_message=MessageSequence([TextComponent(source)]),
            visible_text=source,
            timestamp=timestamp,
            source_kind=source,
        ))
    history.extend([
        ContentContextMessage(
            raw_message=MessageSequence([TextComponent("非 IM 内容")]),
            visible_text="非 IM 内容", timestamp=timestamp,
        ),
        ToolResultMediaMessage(
            raw_message=MessageSequence([TextComponent("工具媒体")]),
            visible_text="工具媒体", timestamp=timestamp, tool_call_id="call-1",
        ),
        ContextSummaryMessage(content="记忆摘要", timestamp=timestamp, remaining_uses_value=None),
        ReferenceMessage(content="参考资料", timestamp=timestamp),
        ToolResultMessage(
            content="工具结果", timestamp=timestamp, tool_call_id="call-1", logical_turn_id="turn-1",
        ),
        ModelOutputContextMessage(
            output_item=ContextItemBuilder().set_role(RoleType.Assistant).add_text_content("Planner 内部输出").build(),
        ),
    ])
    for message_type, text in (
        (SessionBackedMessage, "[事件-群消息表情回应] 用户对消息表达了赞"),
        (SessionBackedMessage, "平台通知：戳一戳、撤回、禁言、入群退群"),
        (ComplexSessionMessage, "未来平台的复杂通知"),
    ):
        original = SessionMessage(message_id=f"notice-{len(history)}", timestamp=timestamp, platform="test")
        original.is_notify = True
        notification = message_type(
            original_message=original,
            message_id=original.message_id,
            timestamp=timestamp,
            raw_message=MessageSequence([TextComponent(text)]),
            visible_text=text,
        )
        if isinstance(notification, ComplexSessionMessage):
            notification.prompt_text = text
        history.append(notification)
    return history


def test_common_filter_only_keeps_real_chat_messages() -> None:
    history = _build_mixed_history()
    assert [
        message for message in history
        if BaseMaisakaReplyGenerator._should_keep_replyer_history_message(message)
    ] == history[:3]


@pytest.mark.parametrize("retro", [False, True], ids=["normal", "retro"])
@pytest.mark.parametrize("visual", [False, True], ids=["text", "visual"])
def test_request_history_excludes_internal_messages(monkeypatch, retro: bool, visual: bool) -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    history = _build_mixed_history()
    if retro:
        # 替换模板与展示依赖，保留真实的复古入口过滤和图文上下文组装。
        monkeypatch.setattr(generator, "_select_retro_prompt_name", lambda **kwargs: "test")
        monkeypatch.setattr(generator, "_build_personality_prompt", lambda: "测试人设")
        monkeypatch.setattr(generator, "_build_retro_template_context", lambda **kwargs: {
            "dialogue_prompt": "\n".join(message.processed_plain_text for message in kwargs["chat_history"]),
        })
        monkeypatch.setattr(generator, "_load_prompt", lambda name, **kwargs: kwargs["dialogue_prompt"], raising=False)
        monkeypatch.setattr(
            generator, "_render_retro_history_line", lambda message, history_by_id: message.processed_plain_text,
        )
        # 图片预处理会读取真实消息媒体；此测试不涉及图片缩放。
        monkeypatch.setattr(generator, "_prepare_retro_visual_history", lambda messages: (messages, {}))
        items = generator._build_retro_request_messages(
            chat_history=history, reply_message=None, reply_reason="", enable_visual_message=visual,
        )
    else:
        items = generator._build_history_messages(history, enable_visual_message=visual)

    text = "\n".join(get_item_text(item) for item in items)
    for message in history[:3]:
        assert message.processed_plain_text in text
    for message in history[3:]:
        assert message.processed_plain_text not in text
    # 过滤不能删除或改写 Planner 共享历史里的内部上下文。
    assert history[3].source_kind == "subagent_result"
    assert len(history) == 19
    assert all(message.original_message.is_notify for message in history[-3:])



def test_retro_dialogue_renders_relative_time_and_early_history_cutoff() -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    history: List[LLMContextMessage] = [
        ContentContextMessage(
            raw_message=MessageSequence([TextComponent("蛋白: 发点色图")]),
            visible_text="蛋白: 发点色图",
            timestamp=datetime(2026, 4, 15, 12, 0, 0),
            message_id="m-early",
            source_kind="user",
        ),
        ContentContextMessage(
            raw_message=MessageSequence([TextComponent("飞鱼丸: 哈哈哈哈")]),
            visible_text="飞鱼丸: 哈哈哈哈",
            timestamp=datetime(2026, 4, 15, 12, 30, 0),
            message_id="m-recent",
            source_kind="user",
        ),
    ]

    rendered = generator._build_retro_dialogue_block(history)
    assert "(30分钟前)" in rendered
    assert "(刚刚)" in rendered
    assert "中间间隔了 30 分钟无人发言" in rendered
    assert "以上是 15 分钟前的早期聊天记录，若与当前话头无关请勿翻旧账提及" in rendered


@pytest.mark.parametrize("visual", [False, True], ids=["text", "visual"])
def test_retro_request_splits_reference_dialogue_and_instruction_blocks(monkeypatch, visual: bool) -> None:
    generator = object.__new__(BaseMaisakaReplyGenerator)
    history = _build_mixed_history()[:2]
    monkeypatch.setattr(generator, "_select_retro_prompt_name", lambda **kwargs: "test")
    monkeypatch.setattr(generator, "_build_personality_prompt", lambda: "测试人设")
    monkeypatch.setattr(
        generator,
        "_build_retro_template_context",
        lambda **kwargs: {
            "extra_info_block": "【关联长期记忆】\n测试记忆内容\n",
            "expression_habits_block": "",
            "dialogue_prompt": "\n".join(msg.processed_plain_text for msg in kwargs["chat_history"]),
        },
    )
    monkeypatch.setattr(
        generator,
        "_load_prompt",
        lambda name, **kwargs: (
            f"{kwargs['extra_info_block']}{kwargs['expression_habits_block']}\n\n"
            f"你正在qq群里聊天:\n当前时间：12:00:00\n{kwargs['dialogue_prompt']}\n\n"
            "现在用户说的：真实用户发言 引起了你的注意。\n把握当前语境直接接话。\n现在，你说："
        ),
        raising=False,
    )
    monkeypatch.setattr(
        generator, "_render_retro_history_line", lambda message, history_by_id: message.processed_plain_text
    )
    monkeypatch.setattr(generator, "_prepare_retro_visual_history", lambda messages: (messages, {}))

    items = generator._build_retro_request_messages(
        chat_history=history,
        reply_message=None,
        reply_reason="",
        enable_visual_message=visual,
    )

    assert len(items) == 4
    assert items[0].role == RoleType.System
    assert get_item_text(items[0]) == "测试人设"
    assert items[1].role == RoleType.User
    assert "【关联长期记忆】" in get_item_text(items[1])
    assert "你正在qq群里聊天" not in get_item_text(items[1])
    assert items[2].role == RoleType.User
    assert "你正在qq群里聊天:" in get_item_text(items[2])
    assert "真实用户发言" in get_item_text(items[2])
    assert "引起了你的注意" not in get_item_text(items[2])
    assert items[3].role == RoleType.User
    assert "引起了你的注意" in get_item_text(items[3])
    assert "现在，你说：" in get_item_text(items[3])


def test_replyer_time_blocks_include_weekday_and_period(monkeypatch) -> None:
    from src.common.utils.math_utils import format_prompt_datetime

    fixed_now = datetime(2026, 11, 13, 17, 4, 39)
    assert format_prompt_datetime(fixed_now) == "2026-11-13 17:04:39（星期五，傍晚）"

    generator = object.__new__(BaseMaisakaReplyGenerator)
    monkeypatch.setattr(generator, "_select_temporary_reply_style", lambda: "")
    monkeypatch.setattr(generator, "_select_reply_style", lambda: "自然口语")
    monkeypatch.setattr(generator, "_build_reply_attachment_prompt", lambda **kwargs: "")
    monkeypatch.setattr(
        "src.chat.replyer.retro_prompt.format_prompt_datetime",
        lambda dt=None: format_prompt_datetime(fixed_now),
    )
    monkeypatch.setattr(
        "src.chat.replyer.maisaka_generator_base.format_prompt_datetime",
        lambda dt=None: format_prompt_datetime(fixed_now),
    )

    retro_ctx = generator._build_retro_template_context(
        chat_history=[],
        reply_message=None,
        reply_reason="",
        reply_reference="",
        expression_habits="",
        reply_requirements="",
    )
    assert retro_ctx["time_block"] == "当前时间：2026-11-13 17:04:39（星期五，傍晚）"

    final_user_msg = generator._build_final_user_message(chat_history=[], reply_message=None)
    assert final_user_msg.startswith("当前时间：2026-11-13 17:04:39（星期五，傍晚）")

