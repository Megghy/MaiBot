from __future__ import annotations

import re

from src.chat.utils.prompt_builder import Prompt


BASE_REPLY_SAFETY_POLICY = "请不要输出违法违规内容，不要输出色情、暴力、政治相关内容，如有敏感内容请规避。"

BASE_REPLY_FORMAT_POLICY = """
只输出最终回复内容。
不要输出解释、前后缀、角色名、引号、括号、项目符号、at、@、表情包或 emoji。
回复保持自然、口语化，不要写成分析说明。
如果你确实想连续发送多条消息，请直接使用换行分隔，不要用空格来暗示多条消息。
""".strip()

GREETING_PATTERNS = (
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "早安",
    "午安",
    "晚安",
    "在吗",
    "在不在",
    "哈喽",
    "hello",
    "hi",
    "嗨",
)
QUESTION_MARKERS = ("?", "？", "吗", "嘛", "么", "啥", "什么", "怎么", "咋", "为什么", "哪", "谁", "几", "推荐")


def _join_non_empty(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(normalized_parts)


def _normalize_target_text(target: str) -> str:
    normalized = re.sub(r"\s+", "", target or "")
    return normalized.strip().lower()


def detect_reply_scene(*, target: str, reply_reason: str = "", is_self_continuation: bool = False) -> str:
    if is_self_continuation:
        return "self_continuation"

    normalized_target = _normalize_target_text(target)
    if not normalized_target:
        return "default"

    if any(pattern in normalized_target for pattern in GREETING_PATTERNS) and len(normalized_target) <= 18:
        return "greeting"

    if any(marker in target for marker in ("?", "？")) or any(marker in normalized_target for marker in QUESTION_MARKERS):
        return "question"

    if len(normalized_target) <= 10 and not reply_reason.strip():
        return "light_touch"

    return "default"


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
            "should_reply=yes",
            "scope=current_message_first",
            f"focus={cleaned_reasoning}",
            "repeat_planner_reasoning=no",
        ]
    )


def build_chat_customization_block(*, chat_prompt_block: str, keywords_reaction_prompt: str) -> str:
    return _join_non_empty(chat_prompt_block, keywords_reaction_prompt)


def _build_reply_scene_budget(reply_scene: str) -> str:
    common_lines = [
        "信息优先级:",
        "1. 先看最新消息本身",
        "2. 再看最近几轮聊天",
        "3. 再参考规划结论",
        "4. 最后才参考更早的记忆和背景",
        "如果后面的信息和最新消息的直觉不一致，以最新消息为准。",
        "",
        "风格预算:",
        "- 最多一个调侃点",
        "- 最多一个轻微玩梗点",
        "- 不要同时打招呼、翻旧账、再开新话题",
    ]
    if reply_scene == "greeting":
        common_lines.extend(
            [
                "",
                "复杂度预算:",
                "- 这是轻量寒暄场景，默认短答",
                "- 默认不要主动引用更早的话题",
                "- 只表达一个意思",
            ]
        )
    elif reply_scene == "question":
        common_lines.extend(
            [
                "",
                "复杂度预算:",
                "- 这是问答场景，优先回答问题本身",
                "- 先答核心点，再决定要不要补一句延伸",
                "- 不要因为想显得有趣而偏题",
            ]
        )
    else:
        common_lines.extend(
            [
                "",
                "复杂度预算:",
                "- 回复保持低复杂度",
                "- 先回应当前点，不要无故扩写",
            ]
        )
    return "\n".join(common_lines)


def _build_reply_few_shots(reply_scene: str) -> str:
    if reply_scene == "greeting":
        return "\n".join(
            [
                "示例:",
                "用户: 晚上好麦麦",
                "好输出: 晚上好 又来打卡了啊",
                "好输出: 晚上好 这会儿才冒头啊",
                "坏输出: 晚上好 你这一天打卡挺勤快啊 中午刚吃完咖喱 晚上又来报到了",
            ]
        )
    if reply_scene == "question":
        return "\n".join(
            [
                "示例:",
                "用户: 晚饭吃什么好",
                "好输出: 去吃咖喱或者黄焖呗",
                "好输出: 你要省事就点个盖饭",
                "坏输出: 晚饭吃什么好 让我先回忆一下今天白天我们聊过什么再决定",
            ]
        )
    if reply_scene == "light_touch":
        return "\n".join(
            [
                "示例:",
                "用户: 打卡",
                "好输出: 又来刷存在感了啊",
                "好输出: 行 行 行 看到你了",
                "坏输出: 你今天从中午到晚上都在认真打卡 这个习惯真稳定",
            ]
        )
    return "\n".join(
        [
            "示例:",
            "用户: 我刚到家",
            "好输出: 行 先歇会儿吧",
            "好输出: 到家就行 赶紧歇着",
            "坏输出: 根据你今天之前的全部聊天记录来看 你现在应该已经很累了",
        ]
    )


def _build_task_intro(*, reply_scene: str, sender_name: str | None = None) -> list[str]:
    if reply_scene == "greeting":
        return [
            "任务:",
            "- 这是问候或轻寒暄场景",
            "- 优先就近回应，不主动翻旧话题",
            "- 回复尽量短、自然、像顺手接一句",
            "- 如果确实要连续发两条，请用换行分隔，不要靠空格或停顿暗示多条消息",
        ]
    if reply_scene == "question":
        return [
            "任务:",
            f"- 你正在回复 {sender_name}" if sender_name else "- 你正在回复对方",
            "- 这是明确提问场景",
            "- 先回答问题，再考虑是否补一句自然延伸",
            "- 如果确实要连续发两条，请用换行分隔，不要靠空格或停顿暗示多条消息",
        ]
    if reply_scene == "light_touch":
        return [
            "任务:",
            f"- 你正在回复 {sender_name}" if sender_name else "- 你正在回复对方",
            "- 这是短消息或轻互动场景",
            "- 给出轻量、顺手、不过度展开的回应",
            "- 如果确实要连续发两条，请用换行分隔，不要靠空格或停顿暗示多条消息",
        ]
    return [
        "任务:",
        f"- 你正在回复 {sender_name}" if sender_name else "- 你正在回复对方",
        "- 阅读聊天记录并结合上面的参考信息",
        "- 直接给出一条自然、连贯、低复杂度的回复",
        "- 如果确实要连续发两条，请用换行分隔，不要靠空格或停顿暗示多条消息",
    ]


def build_group_reply_task(*, reply_scene: str) -> str:
    return "\n".join(
        _build_task_intro(reply_scene=reply_scene)
        + [
            _build_reply_scene_budget(reply_scene),
            _build_reply_few_shots(reply_scene),
            "现在请直接给出回复。",
        ]
    )


def build_private_reply_task(sender_name: str, *, reply_scene: str) -> str:
    return "\n".join(
        _build_task_intro(reply_scene=reply_scene, sender_name=sender_name)
        + [
            _build_reply_scene_budget(reply_scene),
            _build_reply_few_shots(reply_scene),
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
            "- 不要突然切到更早的话题，不要把补充说明写成总结",
            "现在请直接给出回复。",
        ]
    )


def init_replyer_prompt() -> None:
    Prompt("正在群里聊天", "chat_target_group2")
    Prompt("和{sender_name}聊天", "chat_target_private2")
