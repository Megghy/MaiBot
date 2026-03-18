from __future__ import annotations

import asyncio
import time
import traceback
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from src.chat.heart_flow.hfc_utils import CycleDetail
from src.chat.message_receive.chat_stream import ChatStream, get_chat_manager
from src.common.logger import get_logger
from src.plugin_system.apis import message_api

if TYPE_CHECKING:
    from src.common.data_models.database_data_model import DatabaseMessages


class BaseChatController(ABC):
    controller_name = "BaseChatController"
    logger = get_logger("runtime.chat_controller")

    def __init__(self, chat_id: str):
        self.stream_id: str = chat_id
        self.chat_stream: ChatStream = get_chat_manager().get_stream(self.stream_id)  # type: ignore[assignment]
        if not self.chat_stream:
            raise ValueError(f"无法找到聊天流: {self.stream_id}")
        self.log_prefix = f"[{get_chat_manager().get_stream_name(self.stream_id) or self.stream_id}]"

        self.running: bool = False
        self._loop_task: Optional[asyncio.Task] = None
        self._message_lock = asyncio.Lock()

        self.history_loop: list[CycleDetail] = []
        self._cycle_counter = 0
        self._current_cycle_detail: CycleDetail = None  # type: ignore[assignment]

        self.last_read_time = time.time() - 2
        self._runtime_initialized = False

    async def initialize_runtime(self) -> None:
        if self._runtime_initialized:
            return
        await self._initialize_runtime()
        self._runtime_initialized = True

    async def _initialize_runtime(self) -> None:
        return None

    async def start(self) -> None:
        if self.running:
            self.logger.debug(f"{self.log_prefix} {self.controller_name} 已激活，无需重复启动")
            return

        try:
            await self.initialize_runtime()
            self.running = True
            self._loop_task = asyncio.create_task(self._main_chat_loop())
            self._loop_task.add_done_callback(self._handle_loop_completion)
            self.logger.info(f"{self.log_prefix} {self.controller_name} 启动完成")
        except Exception as exc:
            self.running = False
            self._loop_task = None
            self.logger.error(f"{self.log_prefix} {self.controller_name} 启动失败: {exc}")
            raise

    def _handle_loop_completion(self, task: asyncio.Task) -> None:
        try:
            if exception := task.exception():
                self.logger.error(f"{self.log_prefix} {self.controller_name}: 脱离了聊天(异常): {exception}")
                self.logger.error(traceback.format_exc())
            else:
                self.logger.info(f"{self.log_prefix} {self.controller_name}: 脱离了聊天 (外部停止)")
        except asyncio.CancelledError:
            self.logger.info(f"{self.log_prefix} {self.controller_name}: 结束了聊天")

    def start_cycle(self) -> tuple[dict[str, float], str]:
        self._cycle_counter += 1
        self._current_cycle_detail = CycleDetail(self._cycle_counter)
        self._current_cycle_detail.thinking_id = f"tid{str(round(time.time(), 2))}"
        cycle_timers: dict[str, float] = {}
        return cycle_timers, self._current_cycle_detail.thinking_id

    def end_cycle(self, loop_info: dict[str, Any], cycle_timers: dict[str, float]) -> None:
        self._current_cycle_detail.set_loop_info(loop_info)
        self.history_loop.append(self._current_cycle_detail)
        self._current_cycle_detail.timers = cycle_timers
        self._current_cycle_detail.end_time = time.time()

    async def _loopbody(self) -> bool:
        recent_messages_list = message_api.get_messages_by_time_in_chat(
            chat_id=self.stream_id,
            start_time=self.last_read_time,
            end_time=time.time(),
            limit=20,
            limit_mode="latest",
            filter_mai=True,
            filter_command=True,
        )
        processed = await self.process_recent_messages(recent_messages_list)
        if not processed:
            await asyncio.sleep(0.2)
        return True

    async def process_recent_messages(self, recent_messages_list: Optional[list["DatabaseMessages"]] = None) -> bool:
        async with self._message_lock:
            return await self._process_recent_messages_locked(list(recent_messages_list or []))

    @abstractmethod
    async def _process_recent_messages_locked(self, recent_messages_list: list["DatabaseMessages"]) -> bool:
        raise NotImplementedError

    async def _main_chat_loop(self) -> None:
        try:
            while self.running:
                success = await self._loopbody()
                await asyncio.sleep(0.1)
                if not success:
                    break
        except asyncio.CancelledError:
            self.logger.info(f"{self.log_prefix} 麦麦已关闭聊天")
        except Exception:
            self.logger.error(f"{self.log_prefix} 麦麦聊天意外错误，将于3s后尝试重新启动")
            print(traceback.format_exc())
            await asyncio.sleep(3)
            self._loop_task = asyncio.create_task(self._main_chat_loop())
        self.logger.error(f"{self.log_prefix} 结束了当前聊天循环")
