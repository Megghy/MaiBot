import traceback
import os
import re

from typing import Dict, Any, Optional
from maim_message import UserInfo, Seg, GroupInfo

from src.common.logger import get_logger
from src.config.config import global_config
from src.mood.mood_manager import mood_manager  # 导入情绪管理器
from src.chat.message_receive.message import MessageRecv
from src.chat.message_receive.storage import MessageStorage
from src.chat.heart_flow.heartflow_message_processor import HeartFCMessageReceiver
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.plugin_system.base import BaseCommand, EventType
from src.plugin_system.base.component_types import ComponentType
from src.runtime import events_manager
from src.runtime.local_component_state import local_component_state
from src.runtime.manager import RuntimeManager, set_runtime_manager
from src.runtime.message_pipeline import MessagePipeline
from src.runtime.models import CommandExecutionResult
from src.runtime.registries import runtime_registries

# 定义日志配置

# 获取项目根目录（假设本文件在src/chat/message_receive/下，根目录为上上上级目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# 配置主程序日志格式
logger = get_logger("chat")


def _check_ban_words(text: str, userinfo: UserInfo, group_info: Optional[GroupInfo] = None) -> bool:
    """检查消息是否包含过滤词

    Args:
        text: 待检查的文本
        chat: 聊天对象
        userinfo: 用户信息

    Returns:
        bool: 是否包含过滤词
    """
    for word in global_config.message_receive.ban_words:
        if word in text:
            chat_name = group_info.group_name if group_info else "私聊"
            logger.info(f"[{chat_name}]{userinfo.user_nickname}:{text}")
            logger.info(f"[过滤词识别]消息中含有{word}，filtered")
            return True
    return False


def _check_ban_regex(text: str, userinfo: UserInfo, group_info: Optional[GroupInfo] = None) -> bool:
    """检查消息是否匹配过滤正则表达式

    Args:
        text: 待检查的文本
        chat: 聊天对象
        userinfo: 用户信息

    Returns:
        bool: 是否匹配过滤正则
    """
    # 检查text是否为None或空字符串
    if text is None or not text:
        return False

    for pattern in global_config.message_receive.ban_msgs_regex:
        if re.search(pattern, text):
            chat_name = group_info.group_name if group_info else "私聊"
            logger.info(f"[{chat_name}]{userinfo.user_nickname}:{text}")
            logger.info(f"[正则表达式过滤]消息匹配到{pattern}，filtered")
            return True
    return False


class ChatBot:
    def __init__(self):
        self.bot = None  # bot 实例引用
        self._started = False
        self.mood_manager = mood_manager  # 获取情绪管理器单例
        self.heartflow_message_receiver = HeartFCMessageReceiver()  # 新增
        self.runtime = RuntimeManager(
            decision_runner=self._run_runtime_decision,
            message_storage=MessageStorage.store_message,
            stream_resolver=None,
        )
        self.message_pipeline = MessagePipeline(
            self.runtime,
            notice_handler=self.handle_notice_message,
            ban_word_checker=_check_ban_words,
            ban_regex_checker=_check_ban_regex,
        )
        self.runtime.stream_resolver = self.message_pipeline.resolve_chat_stream
        self.runtime.event_bus.register(
            "message.received",
            "legacy_pre_process",
            self._run_legacy_pre_process,
            weight=100,
            intercept=True,
        )
        self.runtime.command_registry.register(
            "legacy_plugin_command",
            self._match_legacy_command,
            self._execute_legacy_command,
            priority=100,
        )
        set_runtime_manager(self.runtime)

    async def _ensure_started(self):
        """确保所有任务已启动"""
        if not self._started:
            logger.debug("确保ChatBot所有任务已启动")

            self._started = True

    async def _resolve_chat_stream(self, message: MessageRecv):
        return await self.message_pipeline.resolve_chat_stream(message)

    async def _run_legacy_pre_process(self, message: MessageRecv):
        continue_flag, modified_message = await events_manager.handle_mai_events(
            EventType.ON_MESSAGE_PRE_PROCESS, message
        )
        if not continue_flag:
            return False, modified_message or message
        if modified_message and modified_message._modify_flags.modify_message_segments:
            message.message_segment = Seg(type="seglist", data=modified_message.message_segments)
            await message.process()
        return True, message

    def _match_legacy_command(self, text: str, *, message: MessageRecv | None = None, session=None):
        command_result = runtime_registries.find_command(text)
        if not command_result:
            return None

        _, _, command_info = command_result
        if (
            session
            and session.stream_id
            and command_info.name
            in local_component_state.get_disabled_components(session.stream_id, ComponentType.COMMAND)
        ):
            logger.info("用户禁用的命令，跳过处理")
            return None

        return {"command_result": command_result}

    async def _execute_legacy_command(self, message: MessageRecv, *, session=None, command_result=None):
        if command_result is None:
            raise ValueError("command_result is required")

        command_class, matched_groups, command_info = command_result
        plugin_config = runtime_registries.get_plugin_config(command_info.plugin_name)
        command_instance: BaseCommand = command_class(message, plugin_config)
        command_instance.set_matched_groups(matched_groups)
        message.is_command = True

        try:
            success, response, intercept_message = await command_instance.execute()
            if success:
                logger.info(f"命令执行成功: {command_class.__name__} (拦截: {intercept_message})")
            else:
                logger.warning(f"命令执行失败: {command_class.__name__} - {response}")
            return CommandExecutionResult(
                name=command_info.name,
                handled=True,
                intercepted=bool(intercept_message),
                response=response,
            )
        except Exception as e:
            logger.error(f"执行命令时出错: {command_class.__name__} - {e}")
            logger.error(traceback.format_exc())
            try:
                await command_instance.send_text(f"命令执行出错: {str(e)}")
            except Exception as send_error:
                logger.error(f"发送错误消息失败: {send_error}")
            return CommandExecutionResult(
                name=command_info.name,
                handled=True,
                intercepted=True,
                response=str(e),
            )

    async def _run_runtime_decision(self, session, message: MessageRecv):
        continue_flag, modified_message = await events_manager.handle_mai_events(EventType.ON_MESSAGE, message)
        if not continue_flag:
            return None
        if modified_message and modified_message._modify_flags.modify_plain_text:
            message.processed_plain_text = modified_message.plain_text

        if message.message_info.template_info and not message.message_info.template_info.template_default:
            template_group_name: Optional[str] = message.message_info.template_info.template_name  # type: ignore
            template_items = message.message_info.template_info.template_items
            async with global_prompt_manager.async_message_scope(template_group_name):
                if isinstance(template_items, dict):
                    for k in template_items.keys():
                        await Prompt.create_async(template_items[k], k)
                        logger.debug(f"注册{template_items[k]},{k}")
        else:
            template_group_name = None

        async def preprocess():
            await self.heartflow_message_receiver.process_message(message, persist_message=False)

        if template_group_name:
            async with global_prompt_manager.async_message_scope(template_group_name):
                await preprocess()
        else:
            await preprocess()
        return None

    async def _process_commands_with_new_system(self, message: MessageRecv):
        # sourcery skip: use-named-expression
        """使用新插件系统处理命令"""
        try:
            text = message.processed_plain_text

            # 使用新的组件注册中心查找命令
            command_result = runtime_registries.find_command(text)
            if command_result:
                command_class, matched_groups, command_info = command_result
                plugin_name = command_info.plugin_name
                command_name = command_info.name
                if (
                    message.chat_stream
                    and message.chat_stream.stream_id
                    and command_name
                    in local_component_state.get_disabled_components(
                        message.chat_stream.stream_id, ComponentType.COMMAND
                    )
                ):
                    logger.info("用户禁用的命令，跳过处理")
                    return False, None, True

                message.is_command = True

                # 获取插件配置
                plugin_config = runtime_registries.get_plugin_config(plugin_name)

                # 创建命令实例
                command_instance: BaseCommand = command_class(message, plugin_config)
                command_instance.set_matched_groups(matched_groups)

                try:
                    # 执行命令
                    success, response, intercept_message = await command_instance.execute()

                    # 记录命令执行结果
                    if success:
                        logger.info(f"命令执行成功: {command_class.__name__} (拦截: {intercept_message})")
                    else:
                        logger.warning(f"命令执行失败: {command_class.__name__} - {response}")

                    # 根据命令的拦截设置决定是否继续处理消息
                    return True, response, not intercept_message  # 找到命令，根据intercept_message决定是否继续

                except Exception as e:
                    logger.error(f"执行命令时出错: {command_class.__name__} - {e}")
                    logger.error(traceback.format_exc())

                    try:
                        await command_instance.send_text(f"命令执行出错: {str(e)}")
                    except Exception as send_error:
                        logger.error(f"发送错误消息失败: {send_error}")

                    # 命令出错时，根据命令的拦截设置决定是否继续处理消息
                    return True, str(e), False  # 出错时继续处理消息

            # 没有找到命令，继续处理消息
            return False, None, True

        except Exception as e:
            logger.error(f"处理命令时出错: {e}")
            return False, None, True  # 出错时继续处理消息

    async def handle_notice_message(self, message: MessageRecv):
        if message.message_info.message_id == "notice":
            message.is_notify = True
            logger.debug("notice消息")
            try:
                seg = message.message_segment
                mi = message.message_info
                sub_type = None
                scene = None
                msg_id = None
                recalled_id = None

                if getattr(seg, "type", None) == "notify" and isinstance(getattr(seg, "data", None), dict):
                    sub_type = seg.data.get("sub_type")
                    scene = seg.data.get("scene")
                    msg_id = seg.data.get("message_id")
                    recalled = seg.data.get("recalled_user_info") or {}
                    if isinstance(recalled, dict):
                        recalled_id = recalled.get("user_id")

                op = mi.user_info
                gid = mi.group_info.group_id if mi.group_info else None

                # 撤回事件打印；无法获取被撤回者则省略
                if sub_type == "recall":
                    op_name = (
                        getattr(op, "user_cardname", None)
                        or getattr(op, "user_nickname", None)
                        or str(getattr(op, "user_id", None))
                    )
                    recalled_name = None
                    try:
                        if isinstance(recalled, dict):
                            recalled_name = (
                                recalled.get("user_cardname")
                                or recalled.get("user_nickname")
                                or str(recalled.get("user_id"))
                            )
                    except Exception:
                        pass

                    if recalled_name and str(recalled_id) != str(getattr(op, "user_id", None)):
                        logger.info(f"{op_name} 撤回了 {recalled_name} 的消息")
                    else:
                        logger.info(f"{op_name} 撤回了消息")
                else:
                    logger.debug(
                        f"[notice] sub_type={sub_type} scene={scene} op={getattr(op, 'user_nickname', None)}({getattr(op, 'user_id', None)}) "
                        f"gid={gid} msg_id={msg_id} recalled={recalled_id}"
                    )
            except Exception:
                logger.info("[notice] (简略) 收到一条通知事件")

            return True

        return

    async def echo_message_process(self, raw_data: Dict[str, Any]) -> None:
        """
        用于专门处理回送消息ID的函数
        """
        message_data: Dict[str, Any] = raw_data.get("content", {})
        if not message_data:
            return
        message_type = message_data.get("type")
        if message_type != "echo":
            return
        mmc_message_id = message_data.get("echo")
        actual_message_id = message_data.get("actual_id")
        if MessageStorage.update_message(mmc_message_id, actual_message_id):
            logger.debug(f"更新消息ID成功: {mmc_message_id} -> {actual_message_id}")
        else:
            logger.warning(f"更新消息ID失败: {mmc_message_id} -> {actual_message_id}")

    async def message_process(self, message_data: Dict[str, Any]) -> None:
        """处理转化后的统一格式消息
        这个函数本质是预处理一些数据，根据配置信息和消息内容，预处理消息，并分发到合适的消息处理器中
        heart_flow模式：使用思维流系统进行回复
        - 包含思维流状态管理
        - 在回复前进行观察和状态更新
        - 回复后更新思维流状态
        - 消息过滤
        - 记忆激活
        - 意愿计算
        - 消息生成和发送
        - 表情包处理
        - 性能计时
        """
        try:
            # 确保所有任务已启动
            await self._ensure_started()
            await self.message_pipeline.process(message_data)

        except Exception as e:
            logger.error(f"预处理消息失败: {e}")
            traceback.print_exc()


# 创建全局ChatBot实例
chat_bot = ChatBot()
