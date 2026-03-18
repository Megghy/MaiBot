import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.common.data_models.info_data_model import ActionPlannerInfo
from src.common.logger import get_logger
from src.llm_models.payload_content.tool_option import ToolCall, ToolParamType
from src.plugin_system.base.component_types import ActionInfo

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("planner")

INTERNAL_ACTION_NAMES = ["no_reply", "reply", "wait_time", "no_reply_until_call"]


def find_message_by_id(
    message_id: str,
    message_id_list: List[Tuple[str, "DatabaseMessages"]],
) -> Optional["DatabaseMessages"]:
    for item in message_id_list:
        if item[0] == message_id:
            return item[1]
    return None


def replace_message_ids_with_text(
    text: Optional[str],
    message_id_list: List[Tuple[str, "DatabaseMessages"]],
) -> Optional[str]:
    if not text:
        return text

    def _replace_match(match: re.Match) -> str:
        message_id = match.group(1)
        message = find_message_by_id(message_id, message_id_list)
        if not message:
            return match.group(0)
        replacement_text = message.processed_plain_text or message.display_message or ""
        replacement_text = replacement_text.strip()
        if not replacement_text:
            return match.group(0)
        return f"[{replacement_text}]"

    return re.sub(r"\[(m\d+)\]", _replace_match, text)


def convert_actions_to_tools(
    actions: Dict[str, ActionInfo],
    *,
    include_no_reply_until_call: bool = False,
    include_reply: bool = True,
) -> List[Dict[str, Any]]:
    tools = []

    if include_reply:
        tools.append(
            {
                "name": "reply",
                "description": "回复一条消息. 当你想回应用户或聊天上下文时使用此项.",
                "parameters": [
                    ("target_message_id", ToolParamType.STRING, "你回复的消息的ID (m+数字)", True, None),
                    (
                        "reason",
                        ToolParamType.STRING,
                        "回复的原因和预期方式/角度, 用尽可能简短明确的语句说明",
                        True,
                        None,
                    ),
                ],
            }
        )

    tools.append(
        {
            "name": "no_reply",
            "description": "不回复. 当你想保持沉默时使用此项.",
            "parameters": [
                ("reason", ToolParamType.STRING, "不回复的原因, 用尽可能简短的语句说明", True, None),
                ("target_message_id", ToolParamType.STRING, "你决定不回复的那条消息的ID (m+数字)", False, None),
            ],
        }
    )
    tools.append(
        {
            "name": "wait_time",
            "description": "在下一个动作之前等待一段时间.",
            "parameters": [
                ("duration", ToolParamType.INTEGER, "等待的持续时间 (秒)", True, None),
                ("reason", ToolParamType.STRING, "等待的原因, 用简短的语句说明", True, None),
            ],
        }
    )

    if include_no_reply_until_call:
        tools.append(
            {
                "name": "no_reply_until_call",
                "description": "保持沉默直到被指名道姓地呼叫.",
                "parameters": [
                    ("reason", ToolParamType.STRING, "进入此模式的原因, 用简短的语句说明", True, None),
                ],
            }
        )

    for action_name, action_info in actions.items():
        if action_name in INTERNAL_ACTION_NAMES:
            continue

        params = []
        metadata_params = (
            (action_info.metadata or {}).get("tool_parameters") if hasattr(action_info, "metadata") else None
        )
        if metadata_params:
            for param in metadata_params:
                name = param.get("name")
                if not name:
                    continue
                raw_type = param.get("type", "STRING")
                param_type = ToolParamType[raw_type.upper()] if isinstance(raw_type, str) else ToolParamType.STRING
                desc = param.get("description") or action_info.action_parameters.get(name, "请输入参数")
                required = bool(param.get("required", True))
                choices = param.get("choices")
                params.append((name, param_type, desc, required, choices))
        elif action_info.action_parameters:
            for param_name, param_desc in action_info.action_parameters.items():
                params.append((param_name, ToolParamType.STRING, param_desc, True, None))

        params.append(("target_message_id", ToolParamType.STRING, "触发此动作的消息ID", False, None))
        params.append(("reason", ToolParamType.STRING, "使用此动作的原因", True, None))

        description = action_info.description or "No description"
        requirements = list(action_info.action_require or [])
        if not action_info.parallel_action:
            requirements.append("当选择这个动作时，请不要选择其他动作 (Exclusive Action)")
        if requirements:
            description += "\n使用条件/要求:\n" + "\n".join([f"- {req}" for req in requirements])

        tools.append(
            {
                "name": action_name,
                "description": description,
                "parameters": params,
            }
        )

    return tools


def format_action_reasoning(
    action_type: str,
    reasoning: str,
    target_message: Optional["DatabaseMessages"],
    action_data: Dict[str, Any],
    target_message_id: Optional[str] = None,
) -> str:
    parts: List[str] = []
    if reasoning:
        parts.append(reasoning.strip())

    msg_content: Optional[str] = None
    if target_message and (target_message.processed_plain_text or target_message.display_message):
        msg_content = (target_message.processed_plain_text or target_message.display_message or "").strip()

    if msg_content:
        excerpt = msg_content[:80] + ("..." if len(msg_content) > 80 else "")
        parts.append(f"目标消息: {excerpt}")
    elif target_message_id:
        parts.append(f"目标消息ID: [{target_message_id}] (内容未找到)")

    if action_data:
        kv_pairs = [f"{k}={v}" for k, v in action_data.items() if v not in (None, "")]
        if kv_pairs:
            parts.append("参数: " + ", ".join(kv_pairs))

    if not parts:
        return f"{action_type}:（无reason）"
    return " | ".join(parts)


def create_no_reply(reasoning: str, available_actions: Dict[str, ActionInfo]) -> List[ActionPlannerInfo]:
    return [
        ActionPlannerInfo(
            action_type="no_reply",
            reasoning=reasoning,
            action_data={},
            action_message=None,
            available_actions=available_actions,
        )
    ]


def parse_single_tool_call(
    tool_call: ToolCall,
    message_id_list: List[Tuple[str, "DatabaseMessages"]],
    current_available_actions: List[Tuple[str, ActionInfo]],
    *,
    log_prefix: str = "",
    extracted_reasoning: str = "",
) -> List[ActionPlannerInfo]:
    action_planner_infos = []
    action_type = tool_call.func_name
    action_args = tool_call.args or {}

    try:
        original_reasoning = action_args.get("reason", "No reason provided")
        reasoning = replace_message_ids_with_text(original_reasoning, message_id_list) or original_reasoning
        target_message_id = action_args.get("target_message_id")
        action_data = {k: v for k, v in action_args.items() if k not in ["target_message_id", "reason"]}

        target_message = None
        if target_message_id:
            target_message = find_message_by_id(target_message_id, message_id_list)
            if target_message is None and message_id_list:
                logger.warning(f"{log_prefix}无法找到target_message_id '{target_message_id}' 对应的消息")
                target_message = message_id_list[-1][1]
        elif message_id_list:
            target_message = message_id_list[-1][1]

        if not target_message_id and target_message is not None:
            for msg_id, msg in reversed(message_id_list):
                if msg is target_message:
                    target_message_id = msg_id
                    break

        available_action_names = [action_name for action_name, _ in current_available_actions]
        if action_type not in INTERNAL_ACTION_NAMES and action_type not in available_action_names:
            logger.warning(
                f"{log_prefix}LLM 返回了当前不可用或无效的动作: '{action_type}' (可用: {available_action_names})，将强制使用 'no_reply'"
            )
            reasoning = (
                f"LLM 返回了当前不可用的动作 '{action_type}' (可用: {available_action_names})。原始理由: {reasoning}"
            )
            action_type = "no_reply"

        structured_reasoning = format_action_reasoning(
            action_type,
            reasoning,
            target_message,
            action_data,
            target_message_id,
        )

        action_planner_infos.append(
            ActionPlannerInfo(
                action_type=action_type,
                reasoning=structured_reasoning,
                action_data=action_data,
                action_message=target_message,
                available_actions=dict(current_available_actions),
                action_reasoning=extracted_reasoning if extracted_reasoning else None,
            )
        )
    except Exception as exc:
        logger.error(f"{log_prefix}解析单个ToolCall时出错: {exc}")
        action_planner_infos.append(
            ActionPlannerInfo(
                action_type="no_reply",
                reasoning=f"解析单个ToolCall时出错: {exc}",
                action_data={},
                action_message=None,
                available_actions=dict(current_available_actions),
                action_reasoning=extracted_reasoning if extracted_reasoning else None,
            )
        )

    return action_planner_infos


def try_parse_tagged_tool_fallback(
    llm_content: str,
    filtered_actions: Dict[str, ActionInfo],
    message_id_list: List[Tuple[str, "DatabaseMessages"]],
    reasoning_text: str,
    *,
    log_prefix: str = "",
) -> Optional[List[ActionPlannerInfo]]:
    if not llm_content:
        return None

    text = llm_content.strip()
    if not text or "<arg_key>" not in text or "</arg_key>" not in text:
        return None

    candidate_actions = INTERNAL_ACTION_NAMES + list(filtered_actions.keys())
    action_type: Optional[str] = None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0] in candidate_actions:
        action_type = lines[0]

    if action_type is None:
        name_match = re.search(r"<(name|action|action_type)>\s*([A-Za-z0-9_]+)\s*</\1>", text)
        if name_match:
            maybe_name = name_match.group(2).strip()
            if maybe_name in candidate_actions:
                action_type = maybe_name

    if action_type is None:
        earliest_pos: Optional[int] = None
        for candidate in candidate_actions:
            match = re.search(rf"\b{re.escape(candidate)}\b", text)
            if match and (earliest_pos is None or match.start() < earliest_pos):
                earliest_pos = match.start()
                action_type = candidate

    if action_type is None:
        return None

    matches = re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", text, re.DOTALL)
    if not matches:
        return None

    args: Dict[str, Any] = {}
    for raw_key, raw_value in matches:
        key = raw_key.strip()
        if key:
            args[key] = raw_value.strip()

    if not args:
        return None

    fake_tool_call = ToolCall(call_id="tagged_fallback_1", func_name=action_type, args=args)
    logger.info(
        f"{log_prefix}检测到 <arg_key>/<arg_value> 格式的工具调用回退: {action_type}，参数: {list(args.keys())}"
    )
    return parse_single_tool_call(
        fake_tool_call,
        message_id_list,
        list(filtered_actions.items()),
        log_prefix=log_prefix,
        extracted_reasoning=reasoning_text,
    )


def try_parse_json_fallback(
    llm_content: str,
    filtered_actions: Dict[str, ActionInfo],
    message_id_list: List[Tuple[str, "DatabaseMessages"]],
    reasoning_text: str,
    *,
    log_prefix: str = "",
) -> Optional[List[ActionPlannerInfo]]:
    content_to_parse = llm_content.strip()
    for prefix in ("```json", "```"):
        if content_to_parse.startswith(prefix):
            content_to_parse = content_to_parse[len(prefix) :]
            break
    if content_to_parse.endswith("```"):
        content_to_parse = content_to_parse[:-3]
    content_to_parse = content_to_parse.strip()

    if not (content_to_parse.startswith("{") and content_to_parse.endswith("}")):
        return None

    logger.info(f"{log_prefix}检测到 JSON 格式的内容，尝试作为工具调用解析")
    action_data = json.loads(content_to_parse)
    action_type = action_data.get("name") or action_data.get("action") or action_data.get("action_type")

    if not action_type:
        for key in filtered_actions:
            if key in action_data:
                action_type = key
                break
        if not action_type:
            if "target_message_id" in action_data:
                action_type = "reply"
            elif "reason" in action_data and len(action_data) == 1:
                action_type = "no_reply"

    if not action_type:
        raise ValueError("无法从 JSON 中推断 action type")

    parsed_args = action_data
    if isinstance(action_data.get("arguments"), dict):
        parsed_args = dict(action_data["arguments"])
    else:
        inner = action_data.get(action_type)
        if isinstance(inner, dict):
            parsed_args = dict(inner)

    fake_tool_call = ToolCall(call_id="fake_json_call", func_name=action_type, args=parsed_args)
    return parse_single_tool_call(
        fake_tool_call,
        message_id_list,
        list(filtered_actions.items()),
        log_prefix=log_prefix,
        extracted_reasoning=reasoning_text,
    )
