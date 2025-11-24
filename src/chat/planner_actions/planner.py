import time
import traceback
import random
import re
from typing import Dict, Optional, Tuple, List, TYPE_CHECKING, Union, Any
from rich.traceback import install
from datetime import datetime
from src.llm_models.utils_model import LLMRequest
from src.llm_models.payload_content.tool_option import ToolParamType, ToolCall
from src.config.config import global_config, model_config
from src.common.logger import get_logger
from src.common.data_models.info_data_model import ActionPlannerInfo
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.utils.chat_message_builder import (
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
MAX_DYNAMIC_ACTION_TOOLS = 6

install(extra_lines=3)


def init_prompt():
    Prompt(
        """
{time_block}
{chat_context_description}, 以下是具体的聊天内容
**Chat Content**
{chat_content_block}

**Action Records**
{actions_before_now_block}

请选择一个合适的 action.
""",
        "planner_prompt",
    )

    Prompt(
        """{time_block}
{chat_context_description}, 以下是具体的聊天内容
**聊天内容**
{chat_content_block}

**Action Records**
{actions_before_now_block}

请选择一个合适的 action.
首先, 思考你的选择理由, 然后使用 tool calls 来执行 action.
**Action Selection Requirements**
请根据聊天内容, 用户的最新消息和以下标准选择合适的 action:
1. 思考**所有**的可用的 action 中的**每个动作**是否符合当下条件, 如果动作使用条件符合聊天内容就使用
2. 如果相同的内容已经被执行, 请不要重复执行
""",
        "planner_prompt_mentioned",
    )

    Prompt(
        """
{action_name}
动作描述: {action_description}
使用条件 {parallel_text}:
{action_require}
""",
        "action_prompt",
    )


class ActionPlanner:
    def __init__(self, chat_id: str, action_manager: ActionManager):
        self.chat_id = chat_id
        self.log_prefix = f"[{get_chat_manager().get_stream_name(chat_id) or chat_id}]"
        self.action_manager = action_manager
        # LLM规划器配置
        self.planner_llm = LLMRequest(
            model_set=model_config.model_task_config.planner, request_type="planner"
        )  # 用于动作规划

        self.last_obs_time_mark = 0.0

        self.plan_log: List[Tuple[str, float, Union[List[ActionPlannerInfo], str]]] = []

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

    def _replace_message_ids_with_text(
        self, text: Optional[str], message_id_list: List[Tuple[str, "DatabaseMessages"]]
    ) -> Optional[str]:
        """将文本中的 m+数字 消息ID替换为原消息内容，并添加双引号"""
        if not text:
            return text

        id_to_message = {msg_id: msg for msg_id, msg in message_id_list}

        # 匹配m后带2-4位数字，前后不是字母数字下划线
        pattern = r"(?<![A-Za-z0-9_])m\d{2,4}(?![A-Za-z0-9_])"

        matches = re.findall(pattern, text)
        if matches:
            available_ids = set(id_to_message.keys())
            found_ids = set(matches)
            missing_ids = found_ids - available_ids
            if missing_ids:
                logger.info(f"{self.log_prefix}planner理由中引用的消息ID不在当前上下文中: {missing_ids}, 可用ID: {list(available_ids)[:10]}...")
            logger.info(f"{self.log_prefix}planner理由替换: 找到{len(matches)}个消息ID引用，其中{len(found_ids & available_ids)}个在上下文中")

        def _replace(match: re.Match[str]) -> str:
            msg_id = match.group(0)
            message = id_to_message.get(msg_id)
            if not message:
                logger.warning(f"{self.log_prefix}planner理由引用 {msg_id} 未找到对应消息，保持原样")
                return msg_id

            msg_text = (message.processed_plain_text or message.display_message or "").strip()
            if not msg_text:
                logger.warning(f"{self.log_prefix}planner理由引用 {msg_id} 的消息内容为空，保持原样")
                return msg_id

            preview = msg_text if len(msg_text) <= 100 else f"{msg_text[:97]}..."
            logger.info(f"{self.log_prefix}planner理由引用 {msg_id} -> 消息（{preview}）")
            return f"消息（{msg_text}）"

        return re.sub(pattern, _replace, text)

    def _convert_actions_to_tools(self, actions: Dict[str, ActionInfo]) -> List[Dict[str, Any]]:
        """Convert actions to tool definitions"""
        tools = []
        
        # Add standard actions
        # reply
        tools.append({
            "name": "reply",
            "description": "回复一条消息. 当你想回应用户或聊天上下文时使用此项.",
            "parameters": [
                ("target_message_id", ToolParamType.STRING, "你回复的消息的ID (m+数字)", True, None),
                ("reason", ToolParamType.STRING, "回复的原因, 用简短的语句说明", True, None)
            ]
        })
        
        # no_reply
        tools.append({
            "name": "no_reply",
            "description": "不回复. 当你想保持沉默时使用此项.",
            "parameters": [
                ("reason", ToolParamType.STRING, "不回复的原因, 用简短的语句说明", True, None)
            ]
        })

        # wait_time
        tools.append({
            "name": "wait_time",
            "description": "在下一个动作之前等待一段时间.",
            "parameters": [
                ("duration", ToolParamType.INTEGER, "等待的持续时间 (秒)", True, None),
                ("reason", ToolParamType.STRING, "等待的原因, 用简短的语句说明", True, None)
            ]
        })

        # no_reply_until_call
        tools.append({
            "name": "no_reply_until_call",
            "description": "保持沉默直到被指名道姓地呼叫.",
            "parameters": [
                ("reason", ToolParamType.STRING, "进入此模式的原因, 用简短的语句说明", True, None)
            ]
        })

        for action_name, action_info in actions.items():
            if action_name in ["reply", "no_reply", "wait_time", "no_reply_until_call"]:
                continue
                
            params = []
            metadata_params = (action_info.metadata or {}).get("tool_parameters") if hasattr(action_info, "metadata") else None
            if metadata_params:
                for param in metadata_params:
                    name = param.get("name")
                    if not name:
                        continue
                    param_type = ToolParamType[param.get("type", "STRING").upper()] if isinstance(param.get("type"), str) else ToolParamType.STRING
                    desc = param.get("description") or action_info.action_parameters.get(name, "请输入参数")
                    required = bool(param.get("required", True))
                    choices = param.get("choices")
                    params.append((name, param_type, desc, required, choices))
            elif action_info.action_parameters:
                for param_name, param_desc in action_info.action_parameters.items():
                    params.append((param_name, ToolParamType.STRING, param_desc, True, None))
            
            # Add common parameters for all actions
            params.append(("target_message_id", ToolParamType.STRING, "触发此动作的消息ID", False, None))
            params.append(("reason", ToolParamType.STRING, "使用此动作的原因", True, None))

            tools.append({
                "name": action_name,
                "description": action_info.description,
                "parameters": params
            })
            
        return tools

    def _parse_single_tool_call(
        self,
        tool_call: ToolCall,
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        current_available_actions: List[Tuple[str, ActionInfo]],
        extracted_reasoning: str = "",
    ) -> List[ActionPlannerInfo]:
        """解析单个ToolCall并返回ActionPlannerInfo列表"""
        action_planner_infos = []
        action_type = tool_call.func_name
        action_args = tool_call.args or {}

        try:
            original_reasoning = action_args.get("reason", "No reason provided")
            reasoning = self._replace_message_ids_with_text(original_reasoning, message_id_list)
            if reasoning is None:
                reasoning = original_reasoning

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

            # 验证action是否可用
            available_action_names = [action_name for action_name, _ in current_available_actions]
            internal_action_names = ["no_reply", "reply", "wait_time", "no_reply_until_call"]

            if action_type not in internal_action_names and action_type not in available_action_names:
                logger.warning(
                    f"{self.log_prefix}LLM 返回了当前不可用或无效的动作: '{action_type}' (可用: {available_action_names})，将强制使用 'no_reply'"
                )
                reasoning = (
                    f"LLM 返回了当前不可用的动作 '{action_type}' (可用: {available_action_names})。原始理由: {reasoning}"
                )
                action_type = "no_reply"

            structured_reasoning = self._format_action_reasoning(action_type, reasoning, target_message, action_data)

            # 创建ActionPlannerInfo对象
            available_actions_dict = dict(current_available_actions)
            action_planner_infos.append(
                ActionPlannerInfo(
                    action_type=action_type,
                    reasoning=structured_reasoning,
                    action_data=action_data,
                    action_message=target_message,
                    available_actions=available_actions_dict,
                    action_reasoning=extracted_reasoning if extracted_reasoning else None,
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
                    action_reasoning=extracted_reasoning if extracted_reasoning else None,
                )
            )

        return action_planner_infos

    async def plan(
        self,
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float = 0.0,
        is_mentioned: bool = False,
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

        # 应用激活类型过滤
        filtered_actions = self._filter_actions_by_activation_type(available_actions, chat_content_block_short)
        selected_actions = self._select_actions_for_prompt(filtered_actions, chat_content_block_short)
        if not selected_actions and filtered_actions:
            selected_actions = filtered_actions

        logger.debug(f"{self.log_prefix}过滤后有{len(filtered_actions)}个可用动作")

        # 如果是提及时且没有可用动作，直接返回空列表，不调用LLM以节省token
        if is_mentioned and not filtered_actions:
            logger.info(f"{self.log_prefix}提及时没有可用动作，跳过plan调用")
            return []

        # 构建包含所有动作的提示词
        prompt, message_id_list = await self.build_planner_prompt(
            is_group_chat=is_group_chat,
            chat_target_info=chat_target_info,
            current_available_actions=selected_actions,
            chat_content_block=chat_content_block,
            message_id_list=message_id_list,
            interest=global_config.personality.interest,
            is_mentioned=is_mentioned,
        )

        # 调用LLM获取决策
        reasoning, actions = await self._execute_main_planner(
            prompt=prompt,
            message_id_list=message_id_list,
            filtered_actions=selected_actions,
            available_actions=available_actions,
            loop_start_time=loop_start_time,
        )

        if actions:
            action_reasons = []
            for action in actions:
                action_reason = action.reasoning or ""
                if action_reason:
                    action_reasons.append(f"{action.action_type}:{action_reason}")
                else:
                    action_reasons.append(f"{action.action_type}:（无reason）")
            log_reasoning = " | ".join(action_reasons)
        else:
            log_reasoning = reasoning or "（无reason）"

        logger.info(
            f"{self.log_prefix}Planner:{log_reasoning}。选择了{len(actions)}个动作: {' '.join([a.action_type for a in actions])}"
        )

        self.add_plan_log(reasoning, actions)

        return actions

    def add_plan_log(self, reasoning: str, actions: List[ActionPlannerInfo]):
        self.plan_log.append((reasoning, time.time(), actions))
        if len(self.plan_log) > 20:
            self.plan_log.pop(0)

    def add_plan_excute_log(self, result: str):
        self.plan_log.append(("", time.time(), result))
        if len(self.plan_log) > 20:
            self.plan_log.pop(0)

    def get_plan_log_str(self, max_action_records: int = 2, max_execution_records: int = 5) -> str:
        """
        获取计划日志字符串

        Args:
            max_action_records: 显示多少条最新的action记录，默认2
            max_execution_records: 显示多少条最新执行结果记录，默认8

        Returns:
            格式化的日志字符串
        """
        action_records = []
        execution_records = []

        # 从后往前遍历，收集最新的记录
        for reasoning, timestamp, content in reversed(self.plan_log):
            if isinstance(content, list) and all(isinstance(action, ActionPlannerInfo) for action in content):
                # 这是action记录
                if len(action_records) < max_action_records:
                    action_records.append((reasoning, timestamp, content, "action"))
            else:
                # 这是执行结果记录
                if len(execution_records) < max_execution_records:
                    execution_records.append((reasoning, timestamp, content, "execution"))

        # 合并所有记录并按时间戳排序
        all_records = action_records + execution_records
        all_records.sort(key=lambda x: x[1])  # 按时间戳排序

        plan_log_str = ""

        # 按时间顺序添加所有记录
        for reasoning, timestamp, content, record_type in all_records:
            time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
            if record_type == "action":
                # plan_log_str += f"{time_str}:{reasoning}|你使用了{','.join([action.action_type for action in content])}\n"
                plan_log_str += f"{time_str}:{reasoning}\n"
            else:
                plan_log_str += f"{time_str}:你执行了action:{content}\n"

        return plan_log_str

    def _has_consecutive_no_reply(self, min_count: int = 3) -> bool:
        """
        检查是否有连续min_count次以上的no_reply

        Args:
            min_count: 需要连续的最少次数，默认3

        Returns:
            如果有连续min_count次以上no_reply返回True，否则返回False
        """
        consecutive_count = 0

        # 从后往前遍历plan_log，检查最新的连续记录
        for _reasoning, _timestamp, content in reversed(self.plan_log):
            if isinstance(content, list) and all(isinstance(action, ActionPlannerInfo) for action in content):
                # 检查所有action是否都是no_reply
                if all(action.action_type == "no_reply" for action in content):
                    consecutive_count += 1
                    if consecutive_count >= min_count:
                        return True
                else:
                    # 如果遇到非no_reply的action，重置计数
                    break

        return False

    async def build_planner_prompt(
        self,
        is_group_chat: bool,
        chat_target_info: Optional["TargetPersonInfo"],
        current_available_actions: Dict[str, ActionInfo],
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        chat_content_block: str = "",
        interest: str = "",
        is_mentioned: bool = False,
    ) -> tuple[str, List[Tuple[str, "DatabaseMessages"]]]:
        """构建 Planner LLM 的提示词 (获取模板并填充数据)"""
        try:
            actions_before_now_block = self.get_plan_log_str()

            # 构建聊天上下文描述
            chat_context_description = "你现在正在一个群聊中"

            # 构建动作选项块
            action_options_block = await self._build_action_options_block(current_available_actions)

            # 其他信息
            moderation_prompt_block = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。"
            time_block = f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            bot_name = global_config.bot.nickname
            bot_nickname = (
                f",也有人叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""
            )
            name_block = f"你的名字是{bot_name}{bot_nickname}，请注意哪些是你自己的发言。"

            # 根据是否是提及时选择不同的模板
            if is_mentioned:
                # 提及时使用简化版提示词，不需要reply、no_reply、no_reply_until_call
                planner_prompt_template = await global_prompt_manager.get_prompt_async("planner_prompt_mentioned")
                prompt = planner_prompt_template.format(
                    time_block=time_block,
                    chat_context_description=chat_context_description,
                    chat_content_block=chat_content_block,
                    actions_before_now_block=actions_before_now_block,
                    action_options_text=action_options_block,
                    moderation_prompt=moderation_prompt_block,
                    name_block=name_block,
                    interest=interest,
                    plan_style=global_config.personality.plan_style,
                )
            else:
                # 正常流程使用完整版提示词
                # 检查是否有连续3次以上no_reply，如果有则添加no_reply_until_call选项
                no_reply_until_call_block = ""
                if self._has_consecutive_no_reply(min_count=3):
                    no_reply_until_call_block = """no_reply_until_call
动作描述：
保持沉默，直到有人直接叫你的名字
当前话题不感兴趣时使用，或有人不喜欢你的发言时使用
当你频繁选择no_reply时使用，表示话题暂时与你无关
{{"action":"no_reply_until_call"}}
"""

                planner_prompt_template = await global_prompt_manager.get_prompt_async("planner_prompt")
                prompt = planner_prompt_template.format(
                    time_block=time_block,
                    chat_context_description=chat_context_description,
                    chat_content_block=chat_content_block,
                    actions_before_now_block=actions_before_now_block,
                    action_options_text=action_options_block,
                    no_reply_until_call_block=no_reply_until_call_block,
                    moderation_prompt=moderation_prompt_block,
                    name_block=name_block,
                    interest=interest,
                    plan_style=global_config.personality.plan_style,
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

    async def _build_action_options_block(self, current_available_actions: Dict[str, ActionInfo]) -> str:
        """构建动作选项块"""
        if not current_available_actions:
            return ""

        action_options_block = ""
        for action_name, action_info in current_available_actions.items():
            # 构建参数文本
            param_text = ""
            if action_info.action_parameters:
                param_text = "\n"
                for param_name, param_description in action_info.action_parameters.items():
                    param_text += f'    "{param_name}":"{param_description}"\n'
                param_text = param_text.rstrip("\n")

            # 构建要求文本
            require_text = ""
            for require_item in action_info.action_require:
                require_text += f"- {require_item}\n"
            require_text = require_text.rstrip("\n")

            if not action_info.parallel_action:
                parallel_text = "(当选择这个动作时，请不要选择其他动作)"
            else:
                parallel_text = ""

            # 获取动作提示模板并填充
            using_action_prompt = await global_prompt_manager.get_prompt_async("action_prompt")
            using_action_prompt = using_action_prompt.format(
                action_name=action_name,
                action_description=action_info.description,
                action_parameters=param_text,
                action_require=require_text,
                parallel_text=parallel_text,
            )

            action_options_block += using_action_prompt

        return action_options_block

    def _select_actions_for_prompt(
        self, available_actions: Dict[str, ActionInfo], chat_content_block: str
    ) -> Dict[str, ActionInfo]:
        if len(available_actions) <= MAX_DYNAMIC_ACTION_TOOLS:
            return available_actions

        scored_actions: List[Tuple[str, ActionInfo, float]] = []
        for name, info in available_actions.items():
            score = self._score_action(info, chat_content_block)
            scored_actions.append((name, info, score))

        scored_actions.sort(key=lambda item: item[2], reverse=True)
        top_items = scored_actions[:MAX_DYNAMIC_ACTION_TOOLS]
        return {name: info for name, info, _ in top_items}

    def _score_action(self, action_info: ActionInfo, chat_content_block: str) -> float:
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

        keywords = action_info.activation_keywords or []
        if keywords:
            for keyword in keywords:
                if keyword and keyword in chat_content_block:
                    score += 1.5
                    break

        metadata_priority = 1.0
        if hasattr(action_info, "metadata") and action_info.metadata:
            metadata_priority = float(action_info.metadata.get("planner_priority", metadata_priority))
        score *= metadata_priority
        return score

    def _format_action_reasoning(
        self,
        action_type: str,
        reasoning: str,
        target_message: Optional["DatabaseMessages"],
        action_data: Dict[str, Any],
    ) -> str:
        parts: List[str] = []
        if reasoning:
            parts.append(reasoning.strip())

        if target_message and (target_message.processed_plain_text or target_message.display_message):
            content = target_message.processed_plain_text or target_message.display_message or ""
            content = content.strip()
            if content:
                excerpt = content[:80] + ("..." if len(content) > 80 else "")
                parts.append(f"目标消息: {excerpt}")

        if action_data:
            kv_pairs = [f"{k}={v}" for k, v in action_data.items() if v not in (None, "")]
            if kv_pairs:
                parts.append("参数: " + ", ".join(kv_pairs))

        if not parts:
            return f"{action_type}:（无reason）"
        return " | ".join(parts)

    async def _execute_main_planner(
        self,
        prompt: str,
        message_id_list: List[Tuple[str, "DatabaseMessages"]],
        filtered_actions: Dict[str, ActionInfo],
        available_actions: Dict[str, ActionInfo],
        loop_start_time: float,
    ) -> Tuple[str, List[ActionPlannerInfo]]:
        """执行主规划器"""
        llm_content = None
        actions: List[ActionPlannerInfo] = []
        reasoning_text = ""

        try:
            # Build tools from filtered actions
            tools = self._convert_actions_to_tools(filtered_actions)

            # 构建相对稳定的 system 提示词（身份与行为规范），用于提升缓存命中率
            moderation_prompt_block = "请不要输出违法违规内容，不要输出色情，暴力，政治相关内容，如有敏感内容，请规避。"
            bot_name = global_config.bot.nickname
            bot_nickname = (
                f",也有人叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""
            )
            name_line = f"你的名字是{bot_name}{bot_nickname}，请注意哪些是你自己的发言。"

            system_prompt = (
                f"{name_line}\n"
                f"你的兴趣是: {global_config.personality.interest}\n"
                "你是 MaiBot 的动作规划器，负责根据聊天上下文选择并执行合适的动作（通过 Tool Calls）。\n\n"
                "在调用工具前, 你必须先在心中回答: \"最新这条消息是发给谁的?\" (参考 Context 中的对话流向)。\n"
                "不要默认消息是发给你的, 只有当你确信需要回应时才行动。\n\n"
                "**Action Selection Requirements**\n"
                f"{global_config.personality.plan_style}\n"
                f"{moderation_prompt_block}\n\n"
                "请使用 Tool Calls 来执行 actions。\n"
            )

            # 调用LLM，使用 system + user 分离的消息结构
            llm_content, (reasoning_content, _, tool_calls) = (
                await self.planner_llm.generate_response_with_system_user_async(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    tools=tools,
                )
            )
            
            reasoning_text = reasoning_content or llm_content or ""

            if global_config.debug.show_planner_prompt:
                logger.info(f"{self.log_prefix}规划器 system 提示词: {system_prompt}")
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
            return f"LLM 请求失败，模型出现问题: {req_e}", [
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
                    actions.extend(
                        self._parse_single_tool_call(
                            tool_call, message_id_list, filtered_actions_list, reasoning_text
                        )
                    )
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

        logger.debug(f"{self.log_prefix}规划器选择了{len(actions)}个动作: {' '.join([a.action_type for a in actions])}")

        return reasoning_text, actions

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
