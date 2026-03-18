import time
import traceback
from datetime import datetime
from typing import List, Tuple, Union

from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("planner")


def init_prompt() -> None:
    Prompt(
        """{name_block}
你的兴趣是: {interest}

**Action Selection Requirements**
{plan_style}
{moderation_prompt}""",
        "planner_system_prompt",
    )

    shared_user_prompt = """
{time_block}
{reply_stats_block}
{chat_context_description}, 以下是具体的聊天内容

**Chat Content**
{chat_content_block}

请选择一个合适的 action.
"""

    Prompt(
        shared_user_prompt
        + """
**Critial Instruction**: 
在调用工具前, 你必须先在心中回答: **"最新这条消息是发给谁的?"** (参考 Context 中的对话流向). 
不要默认消息是发给你的, 只有当你确信需要回应时才行动.

**Available Actions**:
请参考 Tool Definitions 中的工具列表, 使用 Tool Calls 来执行 actions.
""",
        "planner_user_prompt",
    )

    Prompt(
        shared_user_prompt
        + """
首先, 思考你的选择理由, 然后使用 tool calls 来执行 action.
**Action Selection Requirements**
请根据聊天内容, 用户的最新消息和以下标准选择合适的 action:
1. 思考**所有**的可用的 action 中的**每个动作**是否符合当下条件, 如果动作使用条件符合聊天内容就使用
2. 如果相同的内容已经被执行, 请不要重复执行

请使用 Tool Calls 来执行 actions. 你可以同时调用多个 tools.
""",
        "planner_mentioned_user_prompt",
    )


def format_time_delta(seconds: float) -> str:
    seconds_int = int(max(0, seconds))
    if seconds_int < 60:
        return f"{seconds_int}秒"

    minutes, sec = divmod(seconds_int, 60)
    if minutes < 60:
        return f"{minutes}分钟" if sec == 0 else f"{minutes}分钟{sec}秒"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}小时" if minutes == 0 else f"{hours}小时{minutes}分钟"

    days, hours = divmod(hours, 24)
    return f"{days}天" if hours == 0 else f"{days}天{hours}小时"


def build_reply_stats_block(
    plan_log: List[Tuple[str, float, Union[List[ActionPlannerInfo], str]]],
    *,
    window_seconds: float,
) -> str:
    now = time.time()
    reply_timestamps: List[float] = []

    for _reasoning, timestamp, content in plan_log:
        if now - timestamp > window_seconds:
            continue
        if isinstance(content, list) and all(isinstance(action, ActionPlannerInfo) for action in content):
            if any(action.action_type == "reply" for action in content):
                reply_timestamps.append(timestamp)

    minutes = max(1, int(window_seconds // 60) or 1)
    if not reply_timestamps:
        return f"在最近{minutes}分钟内，你还没有回复过消息，可以根据对话内容谨慎选择是否发言。"

    count = len(reply_timestamps)
    last_delta = now - max(reply_timestamps)
    delta_text = format_time_delta(last_delta)
    return f"在最近{minutes}分钟内，你已经回复了{count}条消息，距离你上一次回复已经过去了{delta_text}。"


async def build_planner_prompts(
    *,
    interest: str,
    chat_content_block: str,
    is_mentioned: bool,
    has_consecutive_no_reply: bool,
    plan_log: List[Tuple[str, float, Union[List[ActionPlannerInfo], str]]],
    reply_window_seconds: float,
) -> tuple[str, str, bool, bool]:
    try:
        moderation_prompt_block = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。"
        time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        bot_name = global_config.bot.nickname
        bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""
        name_block = f"你的名字是{bot_name}{bot_nickname}，请注意哪些是你自己的发言。"
        reply_stats_block = build_reply_stats_block(plan_log, window_seconds=reply_window_seconds)

        system_prompt = await global_prompt_manager.format_prompt(
            "planner_system_prompt",
            name_block=name_block,
            interest=interest,
            plan_style=global_config.personality.plan_style,
            moderation_prompt=moderation_prompt_block,
        )

        if is_mentioned:
            allow_no_reply_until_call = False
            allow_reply_action = False
            prompt_name = "planner_mentioned_user_prompt"
        else:
            allow_no_reply_until_call = has_consecutive_no_reply
            allow_reply_action = True
            prompt_name = "planner_user_prompt"

        prompt = await global_prompt_manager.format_prompt(
            prompt_name,
            time_block=time_block,
            reply_stats_block=reply_stats_block,
            chat_context_description="你现在正在一个群聊中",
            chat_content_block=chat_content_block,
        )

        return prompt, system_prompt, allow_no_reply_until_call, allow_reply_action
    except Exception as exc:
        logger.error(f"构建 Planner 提示词时出错: {exc}")
        logger.error(traceback.format_exc())
        return "构建 Planner Prompt 时出错", "", False, True
