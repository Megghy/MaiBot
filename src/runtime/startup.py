from __future__ import annotations

import asyncio
import os

from maim_message import MessageServer

from src.chat.emoji_system.emoji_manager import get_emoji_manager
from src.chat.knowledge import lpmm_start_up
from src.chat.message_receive.bot import chat_bot
from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.utils.statistic import OnlineTimeRecordTask, StatisticOutputTask
from src.common.logger import get_logger
from src.common.remote import TelemetryHeartBeatTask
from src.common.server import Server
from src.config.config import global_config
from src.manager.async_task_manager import async_task_manager
from src.mood.mood_manager import mood_manager
from src.plugin_system.base.component_types import EventType
from src.runtime import events_manager
from src.runtime.bootstrap import bootstrap_runtime

logger = get_logger("runtime.startup")


def setup_webui_server():
    webui_enabled = os.getenv("WEBUI_ENABLED", "false").lower() == "true"
    if not webui_enabled:
        logger.info("WebUI 已禁用")
        return None

    webui_mode = os.getenv("WEBUI_MODE", "production").lower()
    try:
        from src.webui.webui_server import get_webui_server

        webui_server = get_webui_server()
        if webui_mode == "development":
            logger.info("📝 WebUI 开发模式已启用")
            logger.info("🌐 后端 API 将运行在 http://0.0.0.0:8001")
            logger.info("💡 请手动启动前端开发服务器: cd MaiBot-Dashboard && bun dev")
            logger.info("💡 前端将运行在 http://localhost:7999")
        else:
            logger.info("✅ WebUI 生产模式已启用")
            logger.info("🌐 WebUI 将运行在 http://0.0.0.0:8001")
            logger.info("💡 请确保已构建前端: cd MaiBot-Dashboard && bun run build")
        return webui_server
    except Exception as exc:
        logger.error(f"❌ 初始化 WebUI 服务器失败: {exc}")
        return None


async def register_background_tasks() -> None:
    await async_task_manager.add_task(OnlineTimeRecordTask())

    if getattr(global_config.statistic, "enable_output", False):
        await async_task_manager.add_task(StatisticOutputTask())

    await async_task_manager.add_task(TelemetryHeartBeatTask())

    from src.chat.utils.memory_forget_task import MemoryForgetTask

    await async_task_manager.add_task(MemoryForgetTask())


async def initialize_runtime_services(app: MessageServer) -> None:
    await register_background_tasks()

    lpmm_start_up()
    bootstrap_runtime()

    get_emoji_manager().initialize()
    logger.info("表情包管理器初始化成功")

    if global_config.mood.enable_mood:
        await mood_manager.start()
        logger.info("情绪管理器初始化成功")

    chat_manager = get_chat_manager()
    await chat_manager._initialize()
    asyncio.create_task(chat_manager._auto_save_task())
    logger.info("聊天管理器初始化成功")

    app.register_message_handler(chat_bot.message_process)
    app.register_custom_message_handler("message_id_echo", chat_bot.echo_message_process)

    await events_manager.handle_mai_events(event_type=EventType.ON_START)


def build_service_tasks(app: MessageServer, server: Server, webui_server) -> list:
    tasks = [
        get_emoji_manager().start_periodic_check_register(),
        app.run(),
        server.run(),
    ]
    if webui_server:
        tasks.append(webui_server.start())
    return tasks
