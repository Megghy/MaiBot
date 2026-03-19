from __future__ import annotations

import re

from src.chat.utils.prompt_builder import Prompt


BASE_REPLY_SAFETY_POLICY = "请不要输出违法违规内容，不要输出色情、暴力、政治相关内容，如有敏感内容请规避。"

BASE_REPLY_FORMAT_POLICY = """
只输出最终回复内容。
不要输出解释、前后缀、角色名、引号、括号、项目符号、at、@、表情包或 emoji。
回复保持自然、口语化，不要写成分析说明。

【时间约束】
- 简单问候（早/晚上好）→ 别提超过30分钟前的具体事
- 超过1小时的话题 → 别突然翻出来说
- 这不是要你必须接话，而是避免时间错位

【随意性】
- 想说啥说啥，不用非得接话
- 可以突然提起别的，可以答非所问
- 真人聊天本来就是随机的

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

# 公共提示词片段
COMMON_TIME_PRIORITY = """看消息的顺序:
1. 当前这条说了啥
2. 5分钟内的对话
3. 更早的记忆（想提就提，不想提就算了）
不用非得按顺序来，想到啥说啥。"""

COMMON_STYLE_GUIDE = """风格:
- 想调侃就调侃，想玩梗就玩梗
- 但别连续用同一个套路
- 可以突然说些不相关的，这很正常"""

MULTI_MESSAGE_HINT = "想发两条就换行，别用空格装多条"


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
    scene_specific = {
        "greeting": "这看起来是打招呼:\n- 可以简单回，也可以顺便说点别的\n- 别提太久之前的具体事就行",
        "question": "这看起来是提问:\n- 想答就答，不想答也可以岔开话题\n- 真人也不是每次都正经回答",
    }
    specific = scene_specific.get(reply_scene, "普通聊天:\n- 想说啥说啥\n- 可以接话，也可以说点别的")
    return f"{COMMON_TIME_PRIORITY}\n\n{COMMON_STYLE_GUIDE}\n\n{specific}"


def _format_example(good: list[str], bad: list[str], context: list[str] = None) -> str:
    """格式化示例，减少重复代码"""
    lines = ["参考:" if not context else "参考（注意时间）:"]
    if context:
        lines.extend(context)
    lines.extend(f"✓ {ex}" for ex in good)
    lines.extend(f"✗ {ex}" for ex in bad)
    return "\n".join(lines)


def _build_reply_few_shots(reply_scene: str) -> str:
    if reply_scene == "greeting":
        # greeting场景比较特殊，有时间上下文，手动构建
        return "\n".join([
            "参考（注意时间）:",
            "[中午12:30] 用户: 吃啥好",
            "[中午12:35] 你: 咖喱或者黄焖",
            "[晚上20:00] 用户: 晚上好麦麦",
            "✓ 晚上好 又来了",
            "✓ 嗯 这会儿才冒头",
            "✗ 晚上好 中午刚吃完咖喱 晚上又来了（别翻中午的事）",
            "",
            "[早上08:00] 用户: 早",
            "[晚上21:00] 用户: 在吗",
            "✓ 在 咋了",
            "✗ 在 早上不是打过招呼了吗（时间跨度太大）",
        ])
    if reply_scene == "question":
        return _format_example(
            good=["咖喱或者黄焖呗", "懒得想就点盖饭"],
            bad=["让我先回忆一下今天白天聊过什么再决定（别整这些）"],
            context=["用户: 晚饭吃什么好"],
        )
    if reply_scene == "light_touch":
        return _format_example(
            good=["又来刷存在感了", "行 看到你了"],
            bad=["你今天从中午到晚上都在认真打卡（别总结一天）"],
            context=["用户: 打卡"],
        )
    return _format_example(
        good=["行 先歇会儿吧", "到家就行"],
        bad=["根据你今天之前的全部聊天记录来看 你现在应该很累了（别分析）"],
        context=["用户: 我刚到家"],
    )


def _build_task_intro(*, reply_scene: str, sender_name: str | None = None) -> list[str]:
    target = "现在回一句:" if reply_scene == "greeting" else f"现在回 {sender_name or '对方'}:"

    scene_hints = {
        "greeting": ["看起来是打招呼", "想简单回就简单回，想多说两句也行"],
        "question": ["看起来是提问", "想答就答，想岔开话题也可以"],
        "light_touch": ["短消息", "随便回点啥就行"],
    }

    hints = scene_hints.get(reply_scene, ["看看聊天记录", "想说啥说啥"])
    return [target] + [f"- {h}" for h in hints] + [f"- {MULTI_MESSAGE_HINT}"]


def build_group_reply_task(*, reply_scene: str) -> str:
    return "\n".join(
        _build_task_intro(reply_scene=reply_scene)
        + [
            _build_reply_scene_budget(reply_scene),
            _build_reply_few_shots(reply_scene),
            "你说:",
        ]
    )


def build_private_reply_task(sender_name: str, *, reply_scene: str) -> str:
    return "\n".join(
        _build_task_intro(reply_scene=reply_scene, sender_name=sender_name)
        + [
            _build_reply_scene_budget(reply_scene),
            _build_reply_few_shots(reply_scene),
            "你说:",
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
            "现在, 你说:",
        ]
    )


def init_replyer_prompt() -> None:
    Prompt("正在群里聊天", "chat_target_group2")
    Prompt("和{sender_name}聊天", "chat_target_private2")
