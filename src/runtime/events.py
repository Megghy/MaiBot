from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.message_receive.message import MessageRecv, MessageSending
from src.common.logger import get_logger
from src.plugin_system.base.base_events_handler import BaseEventHandler
from src.plugin_system.base.component_types import ComponentType, CustomEventHandlerResult, EventType, MaiMessages
from src.runtime.local_component_state import local_component_state
from src.runtime.registries import runtime_registries

if TYPE_CHECKING:
    from src.common.data_models.llm_data_model import LLMGenerationDataModel

logger = get_logger("runtime.events")


class EventsManager:
    def __init__(self):
        self._handler_tasks: Dict[str, List[asyncio.Task]] = {}
        self._events_result_history: Dict[EventType | str, List[CustomEventHandlerResult]] = {}
        self._history_enable_map: Dict[EventType | str, bool] = {}

        for event in EventType:
            self.register_event(event, enable_history_result=False)

    def register_event(self, event_type: EventType | str, enable_history_result: bool = False):
        if event_type in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 已存在")
        self._history_enable_map[event_type] = enable_history_result
        if enable_history_result:
            self._events_result_history[event_type] = []

    def register_event_subscriber(self, handler_info, handler_class) -> bool:
        return runtime_registries.get_registered_event_handler_info(handler_info.name) is not None

    async def unregister_event_subscriber(self, handler_name: str) -> bool:
        await self.cancel_handler_tasks(handler_name)
        return runtime_registries.get_registered_event_handler_info(handler_name) is None

    async def handle_mai_events(
        self,
        event_type: EventType,
        message: Optional[MessageRecv | MessageSending] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional["LLMGenerationDataModel"] = None,
        stream_id: Optional[str] = None,
        action_usage: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[MaiMessages]]:
        continue_flag = True
        transformed_message = self._prepare_message(
            event_type,
            message,
            llm_prompt,
            llm_response,
            stream_id,
            action_usage,
        )
        if transformed_message:
            transformed_message = transformed_message.deepcopy()

        registrations = runtime_registries.get_enabled_event_handlers_for_event(event_type)
        if not registrations:
            return True, None

        current_stream_id = transformed_message.stream_id if transformed_message else None
        modified_message: Optional[MaiMessages] = None
        for registration in registrations:
            handler_info = registration.info
            handler_class = registration.handler_class

            if current_stream_id and handler_info.name in local_component_state.get_disabled_components(
                current_stream_id,
                ComponentType.EVENT_HANDLER,
            ):
                continue

            from src.plugin_system.core.component_registry import component_registry

            plugin_config = component_registry.get_plugin_config(handler_info.plugin_name) or {}
            handler = handler_class()
            handler.set_plugin_name(handler_info.plugin_name or "unknown")
            handler.set_plugin_config(plugin_config)

            if handler.intercept_message or event_type == EventType.ON_STOP:
                should_continue, modified_message = await self._dispatch_intercepting_handler_task(
                    handler,
                    event_type,
                    modified_message or transformed_message,
                )
                continue_flag = continue_flag and should_continue
            else:
                self._dispatch_handler_task(handler, event_type, transformed_message)

        return continue_flag, modified_message

    async def cancel_handler_tasks(self, handler_name: str) -> None:
        tasks_to_be_cancelled = self._handler_tasks.get(handler_name, [])
        if remaining_tasks := [task for task in tasks_to_be_cancelled if not task.done()]:
            for task in remaining_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(f"取消事件处理器 {handler_name} 的任务超时")
            except Exception as e:
                logger.error(f"取消事件处理器 {handler_name} 的任务时发生异常: {e}")
        self._handler_tasks.pop(handler_name, None)

    async def get_event_result_history(self, event_type: EventType | str) -> List[CustomEventHandlerResult]:
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        if not self._history_enable_map[event_type]:
            raise ValueError(f"事件类型 {event_type} 的历史记录未启用")
        return self._events_result_history[event_type]

    async def clear_event_result_history(self, event_type: EventType | str) -> None:
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        if not self._history_enable_map[event_type]:
            raise ValueError(f"事件类型 {event_type} 的历史记录未启用")
        self._events_result_history[event_type] = []

    def _transform_event_message(
        self,
        message: MessageRecv | MessageSending,
        llm_prompt: Optional[str] = None,
        llm_response: Optional["LLMGenerationDataModel"] = None,
    ) -> MaiMessages:
        transformed_message = MaiMessages(
            llm_prompt=llm_prompt,
            llm_response_content=llm_response.content if llm_response else None,
            llm_response_reasoning=llm_response.reasoning if llm_response else None,
            llm_response_model=llm_response.model if llm_response else None,
            llm_response_tool_call=llm_response.tool_calls if llm_response else None,
            raw_message=message.raw_message,
            additional_data=message.message_info.additional_config or {},
        )

        if message.message_segment.type == "seglist":
            transformed_message.message_segments = list(message.message_segment.data)  # type: ignore
        else:
            transformed_message.message_segments = [message.message_segment]

        if hasattr(message, "chat_stream") and message.chat_stream:
            transformed_message.stream_id = message.chat_stream.stream_id

        transformed_message.plain_text = message.processed_plain_text

        if hasattr(message, "message_info") and message.message_info:
            if message.message_info.platform:
                transformed_message.message_base_info["platform"] = message.message_info.platform
            if message.message_info.group_info:
                transformed_message.is_group_message = True
                transformed_message.message_base_info.update(
                    {
                        "group_id": message.message_info.group_info.group_id,
                        "group_name": message.message_info.group_info.group_name,
                    }
                )
            if message.message_info.user_info:
                if not transformed_message.is_group_message:
                    transformed_message.is_private_message = True
                transformed_message.message_base_info.update(
                    {
                        "user_id": message.message_info.user_info.user_id,
                        "user_cardname": message.message_info.user_info.user_cardname,
                        "user_nickname": message.message_info.user_info.user_nickname,
                    }
                )

        return transformed_message

    def _build_message_from_stream(
        self,
        stream_id: str,
        llm_prompt: Optional[str] = None,
        llm_response: Optional["LLMGenerationDataModel"] = None,
    ) -> MaiMessages:
        chat_stream = get_chat_manager().get_stream(stream_id)
        assert chat_stream, f"未找到流ID为 {stream_id} 的聊天流"
        message = chat_stream.context.get_last_message()
        return self._transform_event_message(message, llm_prompt, llm_response)

    def _transform_event_without_message(
        self,
        stream_id: str,
        llm_prompt: Optional[str] = None,
        llm_response: Optional["LLMGenerationDataModel"] = None,
        action_usage: Optional[List[str]] = None,
    ) -> MaiMessages:
        chat_stream = get_chat_manager().get_stream(stream_id)
        assert chat_stream, f"未找到流ID为 {stream_id} 的聊天流"
        return MaiMessages(
            stream_id=stream_id,
            llm_prompt=llm_prompt,
            llm_response_content=(llm_response.content if llm_response else None),
            llm_response_reasoning=(llm_response.reasoning if llm_response else None),
            llm_response_model=(llm_response.model if llm_response else None),
            llm_response_tool_call=(llm_response.tool_calls if llm_response else None),
            is_group_message=(not (not chat_stream.group_info)),
            is_private_message=(not chat_stream.group_info),
            action_usage=action_usage,
            additional_data={"response_is_processed": True},
        )

    def _prepare_message(
        self,
        event_type: EventType,
        message: Optional[MessageRecv | MessageSending] = None,
        llm_prompt: Optional[str] = None,
        llm_response: Optional["LLMGenerationDataModel"] = None,
        stream_id: Optional[str] = None,
        action_usage: Optional[List[str]] = None,
    ) -> Optional[MaiMessages]:
        if message:
            return self._transform_event_message(message, llm_prompt, llm_response)

        if event_type not in [EventType.ON_START, EventType.ON_STOP]:
            assert stream_id, "如果没有消息，必须为非启动/关闭事件提供流ID"
            if event_type in [EventType.ON_MESSAGE, EventType.ON_PLAN, EventType.POST_LLM, EventType.AFTER_LLM]:
                return self._build_message_from_stream(stream_id, llm_prompt, llm_response)
            return self._transform_event_without_message(stream_id, llm_prompt, llm_response, action_usage)

        return None

    def _dispatch_handler_task(
        self,
        handler: BaseEventHandler,
        event_type: EventType | str,
        message: Optional[MaiMessages] = None,
    ):
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        try:
            task = asyncio.create_task(handler.execute(message))
            task.set_name(handler.handler_name)
            task.add_done_callback(lambda t: self._task_done_callback(t, handler.handler_name, event_type))
            self._handler_tasks.setdefault(handler.handler_name, []).append(task)
        except Exception as e:
            logger.error(f"创建事件处理器任务 {handler.handler_name} 时发生异常: {e}", exc_info=True)

    async def _dispatch_intercepting_handler_task(
        self,
        handler: BaseEventHandler,
        event_type: EventType | str,
        message: Optional[MaiMessages] = None,
    ) -> Tuple[bool, Optional[MaiMessages]]:
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        if event_type not in self._history_enable_map:
            raise ValueError(f"事件类型 {event_type} 未注册")
        try:
            result = await handler.execute(message)
            expected_fields = ["success", "continue_processing", "return_message", "custom_result", "modified_message"]
            if not isinstance(result, tuple) or len(result) != 5:
                logger.error(
                    f"[{self.__class__.__name__}] EventHandler {handler.handler_name} 返回值不符合预期: 期望 {expected_fields}"
                )
                return True, None

            success, continue_processing, return_message, custom_result, modified_message = result
            if not success:
                logger.error(f"EventHandler {handler.handler_name} 执行失败: {return_message}")
            if self._history_enable_map[event_type] and custom_result:
                self._events_result_history[event_type].append(custom_result)
            return continue_processing, modified_message
        except Exception as e:
            logger.error(f"EventHandler {handler.handler_name} 发生异常: {e}", exc_info=True)
            return True, None

    def _task_done_callback(
        self,
        task: asyncio.Task[Tuple[bool, bool, str | None, CustomEventHandlerResult | None, MaiMessages | None]],
        handler_name: str,
        event_type: EventType | str,
    ):
        if event_type == EventType.UNKNOWN:
            raise ValueError("未知事件类型")
        try:
            success, _, result, custom_result, _ = task.result()
            if success:
                logger.debug(f"事件处理任务 {handler_name} 已成功完成: {result}")
            else:
                logger.error(f"事件处理任务 {handler_name} 执行失败: {result}")
            if self._history_enable_map[event_type] and custom_result:
                self._events_result_history[event_type].append(custom_result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"事件处理任务 {handler_name} 发生异常: {e}")
        finally:
            with contextlib.suppress(ValueError):
                self._handler_tasks.get(handler_name, []).remove(task)


events_manager = EventsManager()
