import time
import traceback
from datetime import datetime
from typing import List, Tuple, Union

from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.logger import get_logger
from src.config.config import global_config
from src.runtime.agent import ContextAssembler, PromptComposer, PromptBlock, build_agent_profile

logger = get_logger("planner")


def init_prompt() -> None:
    return None


def _join_non_empty(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(normalized_parts)


def _build_planner_policy_block(plan_style: str) -> str:
    return _join_non_empty(
        "请不要输出违法违规内容，不要输出色情、暴力、政治相关内容，如有敏感内容请规避。",
        plan_style,
    )


def _build_planner_identity_block(interest: str) -> str:
    bot_name = global_config.bot.nickname
    bot_nickname = f"，也有人叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""
    return "\n".join(
        [
            f"你的名字是{bot_name}{bot_nickname}。",
            "你正在负责为当前聊天选择下一步动作。",
            "请注意区分哪些发言是你自己说的。",
            f"你的兴趣是: {interest}",
        ]
    )


def _build_planner_state_block(
    *,
    time_block: str,
    reply_stats_block: str,
    is_mentioned: bool,
    has_consecutive_no_reply: bool,
) -> str:
    mention_state = "最新消息明确提到了你，需要优先判断该不该立即行动。" if is_mentioned else "最新消息没有明确提到你。"
    no_reply_state = (
        "最近连续多次未回复，本轮允许继续保持 no_reply_until_call，除非你确信需要行动。"
        if has_consecutive_no_reply
        else "如果你确信需要回应，可以直接选择动作，不必为了保守而强行沉默。"
    )
    return "\n".join([time_block, reply_stats_block, mention_state, no_reply_state])


def _build_planner_chat_history_block(chat_content_block: str) -> str:
    return _join_non_empty(
        "聊天内容:",
        chat_content_block,
    )


def _build_planner_task_block(is_mentioned: bool) -> str:
    instructions = [
        "任务清单:",
        "1. 先判断最新一条消息究竟是发给谁的，不要默认是发给你。",
        "2. 逐个检查当前所有可用 action 是否真的满足触发条件，不要只凭直觉挑一个。",
        "3. 如果同类内容刚执行过，不要重复执行。",
        "4. 如果没有足够依据说明你应该行动，就不要勉强选择动作。",
        "5. 一旦确认要行动，直接使用 tool calls 选择动作，不要输出额外解释。",
    ]
    if is_mentioned:
        instructions.insert(4, "4. 由于消息提到了你，需要优先判断哪个动作最匹配当前上下文。")
    return "\n".join(instructions)


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
) -> tuple[list[PromptBlock], str, str, bool, bool]:
    try:
        time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        reply_stats_block = build_reply_stats_block(plan_log, window_seconds=reply_window_seconds)

        allow_no_reply_until_call = not is_mentioned and has_consecutive_no_reply
        allow_reply_action = not is_mentioned

        assembled = ContextAssembler().from_parts(
            system_parts={
                "policy": _build_planner_policy_block(global_config.personality.plan_style),
                "identity": _build_planner_identity_block(interest),
            },
            user_parts={
                "state": _build_planner_state_block(
                    time_block=time_block,
                    reply_stats_block=reply_stats_block,
                    is_mentioned=is_mentioned,
                    has_consecutive_no_reply=has_consecutive_no_reply,
                ),
                "chat_history": _build_planner_chat_history_block(chat_content_block),
                "task": _build_planner_task_block(is_mentioned),
            },
        )
        planner_profile = build_agent_profile("planner_default")
        prepared_prompt = PromptComposer().prepare(assembled.blocks, planner_profile)
        composed = PromptComposer().compose(prepared_prompt.blocks)
        return (
            prepared_prompt.blocks,
            composed.user_prompt,
            composed.system_prompt,
            allow_no_reply_until_call,
            allow_reply_action,
        )
    except Exception as exc:
        logger.error(f"构建 Planner 提示词时出错: {exc}")
        logger.error(traceback.format_exc())
        fallback_prompt = "任务清单:\n1. 检查最新消息是否明确发给你。\n2. 只使用 tool calls 选择动作。"
        return (
            ContextAssembler().from_parts(user_parts={"task": fallback_prompt}).blocks,
            fallback_prompt,
            "",
            False,
            True,
        )
