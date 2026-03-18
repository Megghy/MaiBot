import asyncio
import time
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from rich.traceback import install

from src.config.config import global_config
from src.common.logger import get_logger
from src.chat.utils.prompt_builder import global_prompt_manager
from src.chat.utils.timer_calculator import Timer
from src.chat.planner_actions.planner import ActionPlanner
from src.chat.planner_actions.action_modifier import ActionModifier
from src.chat.planner_actions.action_manager import ActionManager
from src.chat.heart_flow.heartfc_runtime_ops import (
    do_reply,
    execute_action,
    run_planner_without_reply,
)
from src.express.expression_learner import expression_learner_manager
from src.chat.frequency_control.frequency_control import frequency_control_manager
from src.jargon import extract_and_store_jargon
from src.plugin_system.base.component_types import ActionInfo
from src.chat.utils.chat_history_summarizer import ChatHistorySummarizer
from src.runtime.chat_controller import BaseChatController

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages


install(extra_lines=3)

# 注释：原来的动作修改超时常量已移除，因为改为顺序执行

logger = get_logger("hfc")  # Logger Name Changed


class HeartFChatting(BaseChatController):
    """
    管理一个连续的Focus Chat循环
    用于在特定聊天流中生成回复。
    其生命周期现在由其关联的 SubHeartflow 的 FOCUSED 状态控制。
    """

    def __init__(self, chat_id: str):
        """
        HeartFChatting 初始化函数

        参数:
            chat_id: 聊天流唯一标识符(如stream_id)
            on_stop_focus_chat: 当收到stop_focus_chat命令时调用的回调函数
            performance_version: 性能记录版本号，用于区分不同启动版本
        """
        super().__init__(chat_id)
        self.expression_learner = expression_learner_manager.get_expression_learner(self.stream_id)
        self.action_manager = ActionManager()
        self.action_planner = ActionPlanner(chat_id=self.stream_id, action_manager=self.action_manager)
        self.action_modifier = ActionModifier(action_manager=self.action_manager, chat_id=self.stream_id)
        self.no_reply_until_call = False
        self.is_mute = False
        self.last_active_time = time.time()  # 记录上一次非noreply时间
        self.question_probability_multiplier = 1
        self.questioned = False
        # 跟踪连续 no_reply 次数，用于动态调整阈值
        self.consecutive_no_reply_count = 0
        # 聊天内容概括器
        self.chat_history_summarizer = ChatHistorySummarizer(chat_id=self.stream_id)

    async def _initialize_runtime(self) -> None:
        await self.chat_history_summarizer.start()

    def print_cycle_info(self, cycle_timers):
        # 记录循环信息和计时器结果
        timer_strings = []
        for name, elapsed in cycle_timers.items():
            if elapsed < 0.1:
                # 不显示小于0.1秒的计时器
                continue
            formatted_time = f"{elapsed:.2f}秒"
            timer_strings.append(f"{name}: {formatted_time}")

        logger.info(
            f"{self.log_prefix} 第{self._current_cycle_detail.cycle_id}次思考,"
            f"耗时: {self._current_cycle_detail.end_time - self._current_cycle_detail.start_time:.1f}秒;"  # type: ignore
            + (f"详情: {'; '.join(timer_strings)}" if timer_strings else "")
        )

    async def _process_recent_messages_locked(self, recent_messages_list: List["DatabaseMessages"]) -> bool:
        """直接处理一批新消息，供新运行时在内存中触发决策。"""
        # 根据连续 no_reply 次数动态调整阈值
        if self.consecutive_no_reply_count >= 5:
            threshold = 2
        elif self.consecutive_no_reply_count >= 3:
            threshold = 2 if random.random() < 0.5 else 1
        else:
            threshold = 1

        if len(recent_messages_list) < threshold:
            return False

        if self.no_reply_until_call:
            for message in recent_messages_list:
                if (
                    message.is_mentioned
                    or message.is_at
                    or len(recent_messages_list) >= 8
                    or time.time() - self.last_read_time > 600
                ):
                    self.no_reply_until_call = False
                    self.last_read_time = time.time()
                    break
            if self.no_reply_until_call:
                return False

        self.last_read_time = time.time()

        mentioned_message = None
        for message in recent_messages_list:
            if (message.is_mentioned or message.is_at) and global_config.chat.mentioned_bot_reply:
                mentioned_message = message

        if mentioned_message:
            await self._observe(recent_messages_list=recent_messages_list, force_reply_message=mentioned_message)
            return True

        if (
            random.random()
            < global_config.chat.get_talk_value(self.stream_id)
            * frequency_control_manager.get_or_create_frequency_control(self.stream_id).get_talk_frequency_adjust()
        ):
            await self._observe(recent_messages_list=recent_messages_list)
            return True

        return False

    async def _generate_mentioned_reply(
        self,
        force_reply_message: "DatabaseMessages",
        thinking_id: str,
        cycle_timers: Dict[str, float],
        available_actions: Dict[str, ActionInfo],
    ) -> Dict[str, Any]:
        """当被提及时，独立生成回复的任务"""
        return await do_reply(
            self,
            reply_message=force_reply_message,
            thinking_id=thinking_id,
            cycle_timers=cycle_timers,
            available_actions=available_actions,
            chosen_actions=[],
            reply_reason="",
            timer_name="提及回复生成",
        )

    async def _observe(
        self,  # interest_value: float = 0.0,
        recent_messages_list: Optional[List["DatabaseMessages"]] = None,
        force_reply_message: Optional["DatabaseMessages"] = None,
    ) -> bool:  # sourcery skip: merge-else-if-into-elif, remove-redundant-if
        if recent_messages_list is None:
            recent_messages_list = []
        _reply_text = ""  # 初始化reply_text变量，避免UnboundLocalError

        start_time = time.time()

        async with global_prompt_manager.async_message_scope(self.chat_stream.context.get_template_name()):
            asyncio.create_task(self.expression_learner.trigger_learning_for_chat())
            asyncio.create_task(
                frequency_control_manager.get_or_create_frequency_control(self.stream_id).trigger_frequency_adjust()
            )

            # 添加curious检测任务 - 检测聊天记录中的矛盾、冲突或需要提问的内容
            # asyncio.create_task(check_and_make_question(self.stream_id))
            # 添加jargon提取任务 - 提取聊天中的黑话/俚语并入库（内部自行取消息并带冷却）
            asyncio.create_task(extract_and_store_jargon(self.stream_id))
            # 添加聊天内容概括任务 - 累积、打包和压缩聊天记录
            # 注意：后台循环已在start()中启动，这里作为额外触发点，在有思考时立即处理
            # asyncio.create_task(self.chat_history_summarizer.process())

            cycle_timers, thinking_id = self.start_cycle()
            logger.info(
                f"{self.log_prefix} 开始第{self._cycle_counter}次思考(频率: {global_config.chat.get_talk_value(self.stream_id)})"
            )

            # 第一步：动作检查
            available_actions: Dict[str, ActionInfo] = {}
            try:
                await self.action_modifier.modify_actions()
                available_actions = self.action_manager.get_using_actions()
            except Exception as e:
                logger.error(f"{self.log_prefix} 动作修改失败: {e}")

            # 如果被提及，让回复生成和planner并行执行
            if force_reply_message:
                logger.info(f"{self.log_prefix} 检测到提及，回复生成与planner并行执行")

                # 并行执行planner和回复生成
                planner_task = asyncio.create_task(
                    run_planner_without_reply(
                        self,
                        available_actions=available_actions,
                        cycle_timers=cycle_timers,
                    )
                )
                reply_task = asyncio.create_task(
                    self._generate_mentioned_reply(
                        force_reply_message=force_reply_message,
                        thinking_id=thinking_id,
                        cycle_timers=cycle_timers,
                        available_actions=available_actions,
                    )
                )

                # 等待两个任务完成
                planner_result, reply_result = await asyncio.gather(planner_task, reply_task, return_exceptions=True)

                # 处理planner结果
                if isinstance(planner_result, BaseException):
                    logger.error(f"{self.log_prefix} Planner执行异常: {planner_result}")
                    action_to_use_info = []
                else:
                    action_to_use_info = planner_result

                # 处理回复结果
                if isinstance(reply_result, BaseException):
                    logger.error(f"{self.log_prefix} 回复生成异常: {reply_result}")
                    reply_result = {
                        "action_type": "reply",
                        "success": False,
                        "result": "回复生成异常",
                        "loop_info": None,
                    }
            else:
                # 正常流程：只执行planner（消息获取已在 plan() 内部完成）
                with Timer("规划器", cycle_timers):
                    action_to_use_info = await self.action_planner.plan(
                        loop_start_time=self.last_read_time,
                        available_actions=available_actions,
                    )
                reply_result = None

            # 只在提及情况下过滤掉planner返回的reply动作（提及时已有独立回复生成）
            if force_reply_message:
                action_to_use_info = [action for action in action_to_use_info if action.action_type != "reply"]

            logger.info(
                f"{self.log_prefix} 决定执行{len(action_to_use_info)}个动作: {' '.join([a.action_type for a in action_to_use_info])}"
            )

            # 3. 并行执行所有动作（不包括reply，reply已经独立执行）
            action_tasks = [
                asyncio.create_task(
                    execute_action(self, action, action_to_use_info, thinking_id, available_actions, cycle_timers)
                )
                for action in action_to_use_info
            ]

            # 并行执行所有任务
            results = await asyncio.gather(*action_tasks, return_exceptions=True)

            # 如果有独立的回复结果，添加到结果列表中
            if reply_result:
                results = list(results) + [reply_result]

            # 处理执行结果
            reply_loop_info = None
            reply_text_from_reply = ""
            action_success = False
            action_reply_text = ""

            excute_result_str = ""
            for result in results:
                excute_result_str += f"{result['action_type']} 执行结果:{result['result']}\n"

                if isinstance(result, BaseException):
                    logger.error(f"{self.log_prefix} 动作执行异常: {result}")
                    continue

                if result["action_type"] != "reply":
                    action_success = result["success"]
                    action_reply_text = result["result"]
                elif result["action_type"] == "reply":
                    if result["success"]:
                        reply_loop_info = result["loop_info"]
                        reply_text_from_reply = result["result"]
                    else:
                        logger.warning(f"{self.log_prefix} 回复动作执行失败")

            self.action_planner.add_plan_excute_log(result=excute_result_str)

            # 构建最终的循环信息
            if reply_loop_info:
                # 如果有回复信息，使用回复的loop_info作为基础
                loop_info = reply_loop_info
                # 更新动作执行信息
                loop_info["loop_action_info"].update(
                    {
                        "action_taken": action_success,
                        "taken_time": time.time(),
                    }
                )
                _reply_text = reply_text_from_reply
            else:
                # 没有回复信息，构建纯动作的loop_info
                loop_info = {
                    "loop_plan_info": {
                        "action_result": action_to_use_info,
                    },
                    "loop_action_info": {
                        "action_taken": action_success,
                        "reply_text": action_reply_text,
                        "taken_time": time.time(),
                    },
                }
                _reply_text = action_reply_text

            self.end_cycle(loop_info, cycle_timers)
            self.print_cycle_info(cycle_timers)

            end_time = time.time()
            if end_time - start_time < global_config.chat.planner_smooth:
                wait_time = global_config.chat.planner_smooth - (end_time - start_time)
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.1)
            return True
