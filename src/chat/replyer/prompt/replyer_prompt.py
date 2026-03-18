from __future__ import annotations

from src.chat.utils.prompt_builder import Prompt


BASE_REPLY_SAFETY_POLICY = "请不要输出违法违规内容，不要输出色情、暴力、政治相关内容，如有敏感内容请规避。"

BASE_REPLY_FORMAT_POLICY = """
只输出最终回复内容。
不要输出解释、前后缀、角色名、引号、括号、项目符号、at、@、表情包或 emoji。
回复保持自然、口语化，不要写成分析说明。
""".strip()


def _join_non_empty(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(normalized_parts)


def build_group_identity_block(identity: str, bot_name: str) -> str:
    return _join_non_empty(
        identity,
        "你正在群聊中聊天。",
        f"其中标注 {bot_name}(你) 的发言是你自己的发言，请注意区分。",
    )


def build_private_identity_block(identity: str, sender_name: str) -> str:
    return _join_non_empty(
        identity,
        f"你正在和{sender_name}私聊。",
    )


def build_reply_style_block(reply_style: str) -> str:
    return _join_non_empty(reply_style, BASE_REPLY_FORMAT_POLICY)


def build_reply_memory_block(
    *,
    knowledge_prompt: str,
    memory_retrieval: str,
    person_memory_block: str,
    relation_info_block: str,
    extra_info_block: str,
    expression_habits_block: str,
    actions_info: str,
) -> str:
    return _join_non_empty(
        knowledge_prompt,
        memory_retrieval,
        person_memory_block,
        relation_info_block,
        extra_info_block,
        expression_habits_block,
        actions_info,
    )


def build_chat_history_block(*, time_block: str, dialogue_prompt: str) -> str:
    return _join_non_empty(
        "聊天记录:",
        time_block,
        dialogue_prompt,
    )


def build_planner_hint_block(planner_reasoning: str) -> str:
    if not planner_reasoning.strip():
        return ""
    cleaned_reasoning = " ".join(planner_reasoning.replace("你的想法是：", "").split())
    return "\n".join(
        [
            "规划阶段结论:",
            "- 已决定在当前轮次回复",
            f"- 回复重点: {cleaned_reasoning}",
            "- 只吸收这个方向，不要复述或解释这段结论",
        ]
    )


def build_chat_customization_block(*, chat_prompt_block: str, keywords_reaction_prompt: str) -> str:
    return _join_non_empty(chat_prompt_block, keywords_reaction_prompt)


def build_group_reply_task() -> str:
    return "\n".join(
        [
            "任务:",
            "- 阅读聊天记录并结合上面的参考信息",
            "- 直接给出一条适合当前群聊语境的自然回复",
            "- 回复可以有个性，但不要写成过于有条理的说明文",
            "现在请直接给出回复。",
        ]
    )


def build_private_reply_task(sender_name: str) -> str:
    return "\n".join(
        [
            "任务:",
            f"- 你正在回复 {sender_name}",
            "- 阅读聊天记录并结合上面的参考信息",
            "- 直接给出一条自然、简短、连贯的私聊回复",
            "现在请直接给出回复。",
        ]
    )


def build_private_self_reply_target(*, target: str, reason: str) -> str:
    return "\n".join(
        [
            f"你刚刚自己的发言是: {target}",
            f"你现在想补充说明这条发言，原因是: {reason}",
            "请从你自己的视角继续往下说，保持上下文连贯。",
        ]
    )


def build_private_self_reply_task() -> str:
    return "\n".join(
        [
            "任务:",
            "- 在刚刚自己的发言基础上继续补充，不要重复背景说明",
            "- 回复保持自然、简短、连续，像同一人紧接着继续说话",
            "现在请直接给出回复。",
        ]
    )


def init_replyer_prompt() -> None:
    Prompt("正在群里聊天", "chat_target_group2")
    Prompt("和{sender_name}聊天", "chat_target_private2")
