import time
import traceback
import random
import re
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING, Any
from rich.traceback import install
from datetime import datetime

from src.llm_models.utils_model import LLMRequest
from src.llm_models.payload_content.tool_option import ToolParamType, ToolCall
from src.config.config import global_config, model_config
from src.common.logger import get_logger
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.utils.chat_message_builder import (
    build_readable_actions,
    get_actions_by_timestamp_with_chat,
    build_readable_messages_with_id,
    get_raw_msg_before_timestamp_with_chat,
)
from src.chat.utils.utils import get_chat_type_and_target_info
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.message_receive.chat_stream import get_chat_manager
from src.plugin_system.base.component_types import ActionInfo, ComponentType, ActionActivationType
from src.plugin_system.core.component_registry import component_registry

if TYPE_CHECKING:
    from src.common.data_models.info_data_model import TargetPersonInfo
    from src.common.data_models.database_data_model import DatabaseMessages

logger = get_logger("planner")

install(extra_lines=3)


def init_prompt():
    Prompt(
        """
{time_block}
{name_block}
你的兴趣是: {interest}
{chat_context_description}, 以下是具体的聊天内容
**聊天内容**
{chat_content_block}

**Action Records**
{actions_before_now_block}

请选择一个合适的 action.
首先, 思考你的选择理由, 然后使用 tool calls 来执行 action.
**Action Selection Requirements**
请根据聊天内容, 用户的最新消息和以下标准选择合适的 action:
{plan_style}
{moderation_prompt}

⚠️ 只允许通过 Tool Calls 返回你的决定, 请勿输出 JSON、自然语言总结或其他自由文本。
如果不需要执行任何动作, 请调用 `no_reply` 工具并说明原因。
""",
        "brain_planner_prompt",
    )


class BrainPlanner:
    def __init__(self, chat_id: str, action_manager: ActionManager):
        self.chat_id = chat_id
        self.log_prefix = f"[{get_chat_manager().get_stream_name(chat_id) or chat_id}]"
        self.action_manager = action_manager
        # LLM规划器配置
        self.planner_llm = LLMRequest(
            model_set=model_config.model_task_config.planner, request_type="planner"
        )  # 用于动作规划

        self.last_obs_time_mark = 0.0

    def find_message_by_id(
        self, message_id: str, message_id_list: List[Tuple[str, "DatabaseMessages"]]
    ) -> Optional["DatabaseMessages"]:
        # sourcery skip: use-next
        """
        根据message_id从message_id_list中查找对应的原始消息

        Args:
            message_id: 要查找的消息ID
            message_id_list: 消息ID列表，格式为[{'id': str, 'message': dict}, ...]

        Returns:
            找到的原始消息字典，如果未找到则返回None
        """
        for item in message_id_list:
            if item[0] == message_id:
                return item[1]
        return None

    def _convert_actions_to_tools(self, actions: Dict[str, ActionInfo]) -> List[Dict[str, Any]]:
        """Convert actions to tool definitions"""
        tools = []

        # Add standard actions
        # reply
        tools.append(
            {
                "name": "reply",
                "description": "回复一条消息. 当你想回应用户或聊天上下文时使用此项.",
                "parameters": [
                    (
                        "target_message_id",
                        ToolParamType.STRING,
                        "你回复的消息的ID (m+数字)",
                        True,
                        None,
                    ),
                    ("reason", ToolParamType.STRING, "回复的原因", True, None),
                ],
            }
        )

        # no_reply
        tools.append(
            {
                "name": "no_reply",
                "description": "不回复. 当你想保持沉默时使用此项.",
                "parameters": [
                    ("reason", ToolParamType.STRING, "不回复的原因", True, None),
                ],
            }
        )

        # wait_time
        tools.append(
            {
                "name": "wait_time",
                "description": "在下一个动作之前等待一段时间.",
                "parameters": [
                    ("duration", ToolParamType.INTEGER, "等待的持续时间 (秒)", True, None),
                    ("reason", ToolParamType.STRING, "等待的原因", True, None),
                ],
            }
        )

        for action_name, action_info in actions.items():
            if action_name in ["reply", "no_reply", "wait_time"]:
                continue

            params = []
            if action_info.action_parameters:
                for param_name, param_desc in action_info.action_parameters.items():
                    # Default to STRING for plugin parameters as type is not specified in ActionInfo
                    params.append((param_name, ToolParamType.STRING, param_desc, True, None))

            # Add common parameters for all actions
            params.append(
                (
                    "target_message_id",
                    ToolParamType.STRING,
                    "触发此动作的消息ID",
                    False,
                    None,
                )
            )
            params.append(("reason", ToolParamType.STRING, "使用此动作的原因", True, None))

            # Construct rich description
            description = action_info.description or "No description"

            requirements = []
            if action_info.action_require:
                requirements.extend(action_info.action_require)

            if not action_info.parallel_action:
                requirements.append("当选择这个动作时，请不要选择其他动作 (Exclusive Action)")

            if requirements:
                description += "\n使用条件/要求:\n" + "\n".join([f"- {req}" for req in requirements])

            tools.append(
                {"name": action_name, "description": description, "parameters": params}
            )
            
        return tools

    def _parse_single_tool_call(
        self,
        tool_call: ToolCall,
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        current_available_actions: List[Tuple[str, ActionInfo]],
    ) -> List[ActionPlannerInfo]:
        """解析单个ToolCall并返回ActionPlannerInfo列表"""
        action_planner_infos = []
        action_type = tool_call.func_name
        action_args = tool_call.args or {}

        try:
            reasoning = action_args.get("reason", "No reason provided")
            target_message_id = action_args.get("target_message_id")
            
            # 清理args中的通用参数
            action_data = {k: v for k, v in action_args.items() if k not in ["target_message_id", "reason"]}

            target_message = None
            if target_message_id:
                target_message = self.find_message_by_id(target_message_id, message_id_list)
                if target_message is None:
                    logger.warning(f"{self.log_prefix}无法找到target_message_id '{target_message_id}' 对应的消息")
                    target_message = message_id_list[-1][1]
            else:
                target_message = message_id_list[-1][1]
                # logger.debug(f"{self.log_prefix}动作'{action_type}'缺少target_message_id，使用最新消息作为target_message")

            # 验证action是否可用
            available_action_names = [action_name for action_name, _ in current_available_actions]
            internal_action_names = ["no_reply", "reply", "wait_time"]

            if action_type not in internal_action_names and action_type not in available_action_names:
                logger.warning(
                    f"{self.log_prefix}LLM 返回了当前不可用或无效的动作: '{action_type}' (可用: {available_action_names})，将强制使用 'no_reply'"
                )
                reasoning = (
                    f"LLM 返回了当前不可用的动作 '{action_type}' (可用: {available_action_names})。原始理由: {reasoning}"
                )
                action_type = "no_reply"

            # 创建ActionPlannerInfo对象
            available_actions_dict = dict(current_available_actions)
            action_planner_infos.append(
                ActionPlannerInfo(
                    action_type=action_type,
                    reasoning=reasoning,
                    action_data=action_data,
                    action_message=target_message,
                    available_actions=available_actions_dict,
                )
            )

        except Exception as e:
            logger.error(f"{self.log_prefix}解析单个ToolCall时出错: {e}")
            available_actions_dict = dict(current_available_actions)
            action_planner_infos.append(
                ActionPlannerInfo(
                    action_type="no_reply",
                    reasoning=f"解析单个ToolCall时出错: {e}",
                    action_data={},
                    action_message=None,
                    available_actions=available_actions_dict,
                )
            )

        return action_planner_infos

    async def plan(
        self,
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float = 0.0,
    ) -> List[ActionPlannerInfo]:
        # sourcery skip: use-named-expression
        """
        规划器 (Planner): 使用LLM根据上下文决定做出什么动作。
        """

        # 获取聊天上下文
        message_list_before_now = get_raw_msg_before_timestamp_with_chat(
            chat_id=self.chat_id,
            timestamp=time.time(),
            limit=int(global_config.chat.max_context_size * 0.6),
        )
        message_id_list: list[Tuple[str, "DatabaseMessages"]] = []
        chat_content_block, message_id_list = build_readable_messages_with_id(
            messages=message_list_before_now,
            timestamp_mode="normal_no_YMD",
            read_mark=self.last_obs_time_mark,
            truncate=True,
            show_actions=True,
        )

        message_list_before_now_short = message_list_before_now[-int(global_config.chat.max_context_size * 0.3) :]
        chat_content_block_short, message_id_list_short = build_readable_messages_with_id(
            messages=message_list_before_now_short,
            timestamp_mode="normal_no_YMD",
            truncate=False,
            show_actions=False,
        )

        self.last_obs_time_mark = time.time()

        # 获取必要信息
        is_group_chat, chat_target_info, current_available_actions = self.get_necessary_info()

        # 提及/被@ 的处理由心流或统一判定模块驱动；Planner 不再做硬编码强制回复

        # 应用激活类型过滤
        filtered_actions = self._filter_actions_by_activation_type(available_actions, chat_content_block_short)

        logger.debug(f"{self.log_prefix}过滤后有{len(filtered_actions)}个可用动作")

        # 构建包含所有动作的提示词
        prompt, message_id_list = await self.build_planner_prompt(
            is_group_chat=is_group_chat,
            chat_target_info=chat_target_info,
            current_available_actions=filtered_actions,
            chat_content_block=chat_content_block,
            message_id_list=message_id_list,
            interest=global_config.personality.interest,
        )

        # 调用LLM获取决策
        actions = await self._execute_main_planner(
            prompt=prompt,
            message_id_list=message_id_list,
            filtered_actions=filtered_actions,
            available_actions=available_actions,
            loop_start_time=loop_start_time,
        )

        return actions

    async def build_planner_prompt(
        self,
        is_group_chat: bool,
        chat_target_info: Optional["TargetPersonInfo"],
        current_available_actions: Dict[str, ActionInfo],
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        chat_content_block: str = "",
        interest: str = "",
    ) -> tuple[str, List[Tuple[str, "DatabaseMessages"]]]:
        """构建 Planner LLM 的提示词 (获取模板并填充数据)"""
        try:
            # 获取最近执行过的动作
            actions_before_now = get_actions_by_timestamp_with_chat(
                chat_id=self.chat_id,
                timestamp_start=time.time() - 600,
                timestamp_end=time.time(),
                limit=6,
            )
            actions_before_now_block = build_readable_actions(actions=actions_before_now)
            if actions_before_now_block:
                actions_before_now_block = f"你刚刚选择并执行过的action是：\n{actions_before_now_block}"
            else:
                actions_before_now_block = ""

            if chat_target_info:
                # 构建聊天上下文描述
                chat_context_description = (
                    f"你正在和 {chat_target_info.person_name or chat_target_info.user_nickname or '对方'} 聊天中"
                )

            # 其他信息
            moderation_prompt_block = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。"
            time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            bot_name = global_config.bot.nickname
            bot_nickname = (
                f",也可以叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""
            )
            name_block = f"你的名字是{bot_name}{bot_nickname}，请注意哪些是你自己的发言。"

            # 获取主规划器模板并填充
            planner_prompt_template = await global_prompt_manager.get_prompt_async("brain_planner_prompt")
            prompt = planner_prompt_template.format(
                time_block=time_block,
                chat_context_description=chat_context_description,
                chat_content_block=chat_content_block,
                actions_before_now_block=actions_before_now_block,
                moderation_prompt=moderation_prompt_block,
                name_block=name_block,
                interest=interest,
                plan_style=global_config.personality.private_plan_style,
            )

            return prompt, message_id_list
        except Exception as e:
            logger.error(f"构建 Planner 提示词时出错: {e}")
            logger.error(traceback.format_exc())
            return "构建 Planner Prompt 时出错", []

    def get_necessary_info(self) -> Tuple[bool, Optional["TargetPersonInfo"], Dict[str, ActionInfo]]:
        """
        获取 Planner 需要的必要信息
        """
        is_group_chat = True
        is_group_chat, chat_target_info = get_chat_type_and_target_info(self.chat_id)
        logger.debug(f"{self.log_prefix}获取到聊天信息 - 群聊: {is_group_chat}, 目标信息: {chat_target_info}")

        current_available_actions_dict = self.action_manager.get_using_actions()

        # 获取完整的动作信息
        all_registered_actions: Dict[str, ActionInfo] = component_registry.get_components_by_type(  # type: ignore
            ComponentType.ACTION
        )
        current_available_actions = {}
        for action_name in current_available_actions_dict:
            if action_name in all_registered_actions:
                current_available_actions[action_name] = all_registered_actions[action_name]
            else:
                logger.warning(f"{self.log_prefix}使用中的动作 {action_name} 未在已注册动作中找到")

        return is_group_chat, chat_target_info, current_available_actions

    def _filter_actions_by_activation_type(
        self, available_actions: Dict[str, ActionInfo], chat_content_block: str
    ) -> Dict[str, ActionInfo]:
        """根据激活类型过滤动作"""
        filtered_actions = {}

        for action_name, action_info in available_actions.items():
            if action_info.activation_type == ActionActivationType.NEVER:
                logger.debug(f"{self.log_prefix}动作 {action_name} 设置为 NEVER 激活类型，跳过")
                continue
            elif action_info.activation_type in [ActionActivationType.LLM_JUDGE, ActionActivationType.ALWAYS]:
                filtered_actions[action_name] = action_info
            elif action_info.activation_type == ActionActivationType.RANDOM:
                if random.random() < action_info.random_activation_probability:
                    filtered_actions[action_name] = action_info
            elif action_info.activation_type == ActionActivationType.KEYWORD:
                if action_info.activation_keywords:
                    for keyword in action_info.activation_keywords:
                        if keyword in chat_content_block:
                            filtered_actions[action_name] = action_info
                            break
            else:
                logger.warning(f"{self.log_prefix}未知的激活类型: {action_info.activation_type}，跳过处理")

        return filtered_actions


    async def _execute_main_planner(
        self,
        prompt: str,
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        filtered_actions: Dict[str, ActionInfo],
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float,
    ) -> List[ActionPlannerInfo]:
        """执行主规划器"""
        llm_content = None
        actions: List[ActionPlannerInfo] = []

        try:
            # Build tools from filtered actions
            tools = self._convert_actions_to_tools(filtered_actions)

            # 调用LLM
            llm_content, (reasoning_content, _, tool_calls) = await self.planner_llm.generate_response_async(
                prompt=prompt,
                tools=tools
            )

            if global_config.debug.show_planner_prompt:
                logger.info(f"{self.log_prefix}规划器原始提示词: {prompt}")
                logger.info(f"{self.log_prefix}规划器原始响应: {llm_content}")
                if reasoning_content:
                    logger.info(f"{self.log_prefix}规划器推理: {reasoning_content}")
                if tool_calls:
                     logger.info(f"{self.log_prefix}规划器工具调用: {[t.func_name for t in tool_calls]}")
            else:
                logger.debug(f"{self.log_prefix}规划器原始提示词: {prompt}")
                logger.debug(f"{self.log_prefix}规划器原始响应: {llm_content}")
                if reasoning_content:
                    logger.debug(f"{self.log_prefix}规划器推理: {reasoning_content}")

        except Exception as req_e:
            logger.error(f"{self.log_prefix}LLM 请求执行失败: {req_e}")
            return [
                ActionPlannerInfo(
                    action_type="no_reply",
                    reasoning=f"LLM 请求失败，模型出现问题: {req_e}",
                    action_data={},
                    action_message=None,
                    available_actions=available_actions,
                )
            ]

        # Parse Tool Calls
        if tool_calls:
            try:
                logger.debug(f"{self.log_prefix}从响应中提取到{len(tool_calls)}个工具调用")
                filtered_actions_list = list(filtered_actions.items())
                for tool_call in tool_calls:
                    actions.extend(self._parse_single_tool_call(tool_call, message_id_list, filtered_actions_list))
            except Exception as json_e:
                 logger.warning(f"{self.log_prefix}解析LLM工具调用失败 {json_e}")
                 traceback.print_exc()
                 actions = self._create_no_reply(f"解析LLM工具调用失败: {json_e}", available_actions)
        else:
            logger.warning(f"{self.log_prefix}LLM没有返回工具调用: {llm_content}")
            reason = "规划器没有返回有效的动作选择"
            if llm_content:
                reason = f"规划器未选择动作，回复内容: {llm_content[:50]}..."
            actions = self._create_no_reply(reason, available_actions)

        # 添加循环开始时间到所有非no_reply动作
        for action in actions:
            action.action_data = action.action_data or {}
            action.action_data["loop_start_time"] = loop_start_time

        logger.debug(
            f"{self.log_prefix}规划器决定执行{len(actions)}个动作: {' '.join([a.action_type for a in actions])}"
        )

        return actions

    def _create_no_reply(self, reasoning: str, available_actions: Dict[str, ActionInfo]) -> List[ActionPlannerInfo]:
        """创建no_reply"""
        return [
            ActionPlannerInfo(
                action_type="no_reply",
                reasoning=reasoning,
                action_data={},
                action_message=None,
                available_actions=available_actions,
            )
        ]


init_prompt()
