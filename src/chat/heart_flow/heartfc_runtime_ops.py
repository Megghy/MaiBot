import random
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.chat.utils.timer_calculator import Timer
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.data_models.message_data_model import ReplyContentType
from src.common.logger import get_logger
from src.config.config import global_config
from src.person_info.person_info import Person
from src.plugin_system.apis import database_api, generator_api, message_api, send_api
from src.plugin_system.base.component_types import ActionInfo

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages
    from src.common.data_models.message_data_model import ReplySetModel

logger = get_logger("hfc")


async def send_response(
    chatting,
    reply_set: "ReplySetModel",
    message_data: "DatabaseMessages",
    selected_expressions: Optional[List[int]] = None,
) -> str:
    new_message_count = message_api.count_new_messages(
        chat_id=chatting.chat_stream.stream_id,
        start_time=chatting.last_read_time,
        end_time=time.time(),
    )
    need_reply = new_message_count >= random.randint(2, 3)
    if need_reply:
        logger.info(f"{chatting.log_prefix} 从思考到回复，共有{new_message_count}条新消息，使用引用回复")

    reply_text = ""
    first_replied = False
    for reply_content in reply_set.reply_data:
        if reply_content.content_type != ReplyContentType.TEXT:
            continue
        data: str = reply_content.content  # type: ignore
        await send_api.text_to_stream(
            text=data,
            stream_id=chatting.chat_stream.stream_id,
            reply_message=message_data,
            set_reply=need_reply and not first_replied,
            typing=first_replied,
            selected_expressions=selected_expressions,
        )
        first_replied = True
        reply_text += data

    return reply_text


async def send_and_store_reply(
    chatting,
    response_set: "ReplySetModel",
    action_message: "DatabaseMessages",
    cycle_timers: Dict[str, float],
    thinking_id,
    actions,
    selected_expressions: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], str, Dict[str, float]]:
    with Timer("回复发送", cycle_timers):
        reply_text = await send_response(
            chatting,
            reply_set=response_set,
            message_data=action_message,
            selected_expressions=selected_expressions,
        )

    platform = action_message.chat_info.platform or getattr(chatting.chat_stream, "platform", "unknown")
    person = Person(platform=platform, user_id=action_message.user_info.user_id)
    person_name = person.person_name
    action_prompt_display = f"你对{person_name}进行了回复：{reply_text}"

    await database_api.store_action_info(
        chat_stream=chatting.chat_stream,
        action_build_into_prompt=False,
        action_prompt_display=action_prompt_display,
        action_done=True,
        thinking_id=thinking_id,
        action_data={"reply_text": reply_text},
        action_name="reply",
    )

    loop_info: Dict[str, Any] = {
        "loop_plan_info": {"action_result": actions},
        "loop_action_info": {
            "action_taken": True,
            "reply_text": reply_text,
            "command": "",
            "taken_time": time.time(),
        },
    }
    return loop_info, reply_text, cycle_timers


async def run_planner_without_reply(
    chatting,
    available_actions: Dict[str, ActionInfo],
    cycle_timers: Dict[str, float],
) -> List[ActionPlannerInfo]:
    try:
        with Timer("规划器", cycle_timers):
            action_to_use_info = await chatting.action_planner.plan(
                loop_start_time=chatting.last_read_time,
                available_actions=available_actions,
                is_mentioned=True,
            )
        return [action for action in action_to_use_info if action.action_type != "reply"]
    except Exception as exc:
        logger.error(f"{chatting.log_prefix} Planner执行失败: {exc}")
        traceback.print_exc()
        return []


async def do_reply(
    chatting,
    reply_message: "DatabaseMessages",
    thinking_id: str,
    cycle_timers: Dict[str, float],
    available_actions: Dict[str, ActionInfo],
    chosen_actions: Optional[List[ActionPlannerInfo]] = None,
    reply_reason: str = "",
    reply_time_point: Optional[float] = None,
    timer_name: str = "回复生成",
) -> Dict[str, Any]:
    chosen_actions = chosen_actions or []
    reply_time_point = chatting.last_read_time if reply_time_point is None else reply_time_point

    try:
        chatting.questioned = False
        chatting.consecutive_no_reply_count = 0

        await database_api.store_action_info(
            chat_stream=chatting.chat_stream,
            action_build_into_prompt=bool(reply_reason),
            action_prompt_display=reply_reason,
            action_done=True,
            thinking_id=thinking_id,
            action_data={},
            action_name="reply",
            action_reasoning=reply_reason,
        )

        with Timer(timer_name, cycle_timers):
            success, llm_response = await generator_api.generate_reply(
                chat_stream=chatting.chat_stream,
                reply_message=reply_message,
                available_actions=available_actions,
                chosen_actions=chosen_actions,
                reply_reason=reply_reason,
                enable_tool=global_config.tool.enable_tool,
                request_type="replyer",
                from_plugin=False,
                reply_time_point=reply_time_point,
            )

        if not success or not llm_response or not llm_response.reply_set:
            logger.warning(f"{chatting.log_prefix} {timer_name}失败")
            return {"action_type": "reply", "success": False, "result": f"{timer_name}失败", "loop_info": None}

        loop_info, reply_text, _ = await send_and_store_reply(
            chatting,
            response_set=llm_response.reply_set,
            action_message=reply_message,
            cycle_timers=cycle_timers,
            thinking_id=thinking_id,
            actions=chosen_actions,
            selected_expressions=llm_response.selected_expressions,
        )
        chatting.last_active_time = time.time()
        return {
            "action_type": "reply",
            "success": True,
            "result": f"你回复内容{reply_text}",
            "loop_info": loop_info,
        }
    except Exception as exc:
        logger.error(f"{chatting.log_prefix} {timer_name}异常: {exc}")
        traceback.print_exc()
        return {"action_type": "reply", "success": False, "result": f"{timer_name}异常: {exc}", "loop_info": None}


async def handle_action(
    chatting,
    action: str,
    action_reasoning: str,
    action_data: dict,
    cycle_timers: Dict[str, float],
    thinking_id: str,
    action_message: Optional["DatabaseMessages"] = None,
) -> tuple[bool, str]:
    try:
        try:
            action_handler = chatting.action_manager.create_action(
                action_name=action,
                action_data=action_data,
                cycle_timers=cycle_timers,
                thinking_id=thinking_id,
                chat_stream=chatting.chat_stream,
                log_prefix=chatting.log_prefix,
                action_reasoning=action_reasoning,
                action_message=action_message,
            )
        except Exception as exc:
            logger.error(f"{chatting.log_prefix} 创建动作处理器时出错: {exc}")
            traceback.print_exc()
            return False, ""

        success, action_text = await action_handler.execute()
        return success, action_text
    except Exception as exc:
        logger.error(f"{chatting.log_prefix} 处理{action}时出错: {exc}")
        traceback.print_exc()
        return False, ""


async def execute_action(
    chatting,
    action_planner_info: ActionPlannerInfo,
    chosen_action_plan_infos: List[ActionPlannerInfo],
    thinking_id: str,
    available_actions: Dict[str, ActionInfo],
    cycle_timers: Dict[str, float],
):
    try:
        with Timer(f"动作{action_planner_info.action_type}", cycle_timers):
            if action_planner_info.action_type == "no_reply":
                reason = action_planner_info.reasoning or "选择不回复"
                chatting.consecutive_no_reply_count += 1
                await database_api.store_action_info(
                    chat_stream=chatting.chat_stream,
                    action_build_into_prompt=True,
                    action_prompt_display=reason,
                    action_done=True,
                    thinking_id=thinking_id,
                    action_data={},
                    action_name="no_reply",
                    action_reasoning=reason,
                )
                return {"action_type": "no_reply", "success": True, "result": "选择不回复", "loop_info": None}

            if action_planner_info.action_type == "no_reply_until_call":
                logger.info(f"{chatting.log_prefix} 保持沉默，直到有人直接叫的名字")
                reason = action_planner_info.reasoning or "选择不回复"
                chatting.consecutive_no_reply_count += 1
                chatting.no_reply_until_call = True
                await database_api.store_action_info(
                    chat_stream=chatting.chat_stream,
                    action_build_into_prompt=True,
                    action_prompt_display=reason,
                    action_done=True,
                    thinking_id=thinking_id,
                    action_data={},
                    action_name="no_reply_until_call",
                    action_reasoning=reason,
                )
                return {
                    "action_type": "no_reply_until_call",
                    "success": True,
                    "result": "保持沉默，直到有人直接叫的名字",
                    "loop_info": None,
                }

            if action_planner_info.action_type == "reply":
                reason = (action_planner_info.reasoning or "").strip()
                reply_reason = reason.split(" | ", 1)[0].strip() if " | " in reason else reason
                return await do_reply(
                    chatting,
                    reply_message=action_planner_info.action_message,  # type: ignore[arg-type]
                    thinking_id=thinking_id,
                    cycle_timers=cycle_timers,
                    available_actions=available_actions,
                    chosen_actions=chosen_action_plan_infos,
                    reply_reason=reply_reason,
                    reply_time_point=action_planner_info.action_data.get("loop_start_time", time.time())
                    if action_planner_info.action_data
                    else time.time(),
                )

            with Timer("动作执行", cycle_timers):
                success, result = await handle_action(
                    chatting,
                    action=action_planner_info.action_type,
                    action_reasoning=action_planner_info.action_reasoning or "",
                    action_data=action_planner_info.action_data or {},
                    cycle_timers=cycle_timers,
                    thinking_id=thinking_id,
                    action_message=action_planner_info.action_message,
                )

            chatting.last_active_time = time.time()
            return {"action_type": action_planner_info.action_type, "success": success, "result": result}
    except Exception as exc:
        logger.error(f"{chatting.log_prefix} 执行动作时出错: {exc}")
        logger.error(f"{chatting.log_prefix} 错误信息: {traceback.format_exc()}")
        return {
            "action_type": action_planner_info.action_type,
            "success": False,
            "result": "",
            "loop_info": None,
            "error": str(exc),
        }
