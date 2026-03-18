import random
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from src.chat.planner_actions.action_manager import ActionManager
from src.chat.utils.utils import get_chat_type_and_target_info
from src.common.logger import get_logger
from src.plugin_system.base.component_types import ActionActivationType, ActionInfo, ComponentType
from src.plugin_system.core.component_registry import component_registry

if TYPE_CHECKING:
    from src.common.data_models.info_data_model import TargetPersonInfo

logger = get_logger("planner")


def get_necessary_info(
    chat_id: str,
    action_manager: ActionManager,
    *,
    log_prefix: str = "",
) -> Tuple[bool, Optional["TargetPersonInfo"], Dict[str, ActionInfo]]:
    is_group_chat, chat_target_info = get_chat_type_and_target_info(chat_id)
    logger.debug(f"{log_prefix}获取到聊天信息 - 群聊: {is_group_chat}, 目标信息: {chat_target_info}")

    current_available_actions_dict = action_manager.get_using_actions()
    all_registered_actions: Dict[str, ActionInfo] = component_registry.get_components_by_type(ComponentType.ACTION)  # type: ignore
    current_available_actions = {}
    for action_name in current_available_actions_dict:
        if action_name in all_registered_actions:
            current_available_actions[action_name] = all_registered_actions[action_name]
        else:
            logger.warning(f"{log_prefix}使用中的动作 {action_name} 未在已注册动作中找到")

    return is_group_chat, chat_target_info, current_available_actions


def filter_actions_by_activation_type(
    available_actions: Dict[str, ActionInfo],
    chat_content_block: str,
    *,
    log_prefix: str = "",
) -> Dict[str, ActionInfo]:
    filtered_actions = {}

    for action_name, action_info in available_actions.items():
        if action_info.activation_type == ActionActivationType.NEVER:
            logger.debug(f"{log_prefix}动作 {action_name} 设置为 NEVER 激活类型，跳过")
            continue
        if action_info.activation_type in [ActionActivationType.LLM_JUDGE, ActionActivationType.ALWAYS]:
            filtered_actions[action_name] = action_info
            continue
        if action_info.activation_type == ActionActivationType.RANDOM:
            if random.random() < action_info.random_activation_probability:
                filtered_actions[action_name] = action_info
            continue
        if action_info.activation_type == ActionActivationType.KEYWORD:
            if any(keyword in chat_content_block for keyword in (action_info.activation_keywords or [])):
                filtered_actions[action_name] = action_info
            continue

        logger.warning(f"{log_prefix}未知的激活类型: {action_info.activation_type}，跳过处理")

    return filtered_actions


def score_action(action_info: ActionInfo, chat_content_block: str) -> float:
    score = 0.0
    activation = action_info.activation_type
    if activation == ActionActivationType.ALWAYS:
        score += 3
    elif activation == ActionActivationType.LLM_JUDGE:
        score += 2
    elif activation == ActionActivationType.RANDOM:
        score += 1

    if action_info.parallel_action:
        score -= 0.3

    if any(keyword and keyword in chat_content_block for keyword in (action_info.activation_keywords or [])):
        score += 1.5

    metadata_priority = 1.0
    if hasattr(action_info, "metadata") and action_info.metadata:
        metadata_priority = float(action_info.metadata.get("planner_priority", metadata_priority))
    return score * metadata_priority


def select_actions_for_prompt(
    available_actions: Dict[str, ActionInfo],
    chat_content_block: str,
    *,
    max_dynamic_action_tools: int,
) -> Dict[str, ActionInfo]:
    if len(available_actions) <= max_dynamic_action_tools:
        return available_actions

    scored_actions = []
    for name, info in available_actions.items():
        scored_actions.append((name, info, score_action(info, chat_content_block)))

    scored_actions.sort(key=lambda item: item[2], reverse=True)
    top_items = scored_actions[:max_dynamic_action_tools]
    return {name: info for name, info, _ in top_items}
