"""Plugin-owned turn decisions, executed by the session's single reasoning worker."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from src.common.logger import get_logger
from src.config.config import global_config
from src.maisaka.context.messages import SessionBackedMessage
from src.maisaka.monitor.events import emit_turn_decision
from src.plugin_runtime.host.hook_spec_registry import HookSpec
from src.plugin_runtime.host.message_utils import PluginMessageUtils
from src.plugin_runtime.integration import get_plugin_runtime_manager

if TYPE_CHECKING:
    from src.chat.message_receive.message import SessionMessage
    from src.maisaka.runtime import MaisakaHeartFlowChatting
    from src.maisaka.turn_trigger.scheduler import MessageTurnScheduler

HOOK_NAME = "maisaka.turn.decide"
logger = get_logger("maisaka_turn_scheduler")
PRIVATE_DIRECT_REPLY_REFERENCE = (
    "按语境自然简短回应: 日常提问或聊正事就直接接话解答; "
    "若对方在挑衅找茬或对喷, 看对方火力直接回敬一句, 戳穿痛处就行, 别太过分; "
    "若对方在抱怨别人, 分享遭遇, 搬运言论或冒抽象/种草念头, 用冷淡刁钻的角度简短阴阳一句(先作废第一反应的平庸槽点): "
    "对方吐槽外人时别当捧哏陪骂, 掉转准星点破说话人自己在里面的尴尬处境, 地位盲点或隐瞒的事实; "
    "对方自己种草智商税或冒抽象念头时, 顺着该设定冷静推导出一个荒谬后果; "
    "点到即止, 不做低级人身攻击或诅咒, 不用'……是吧/挽尊/笑死'等套话."
)


class TurnDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["pass", "reply", "planner", "wait", "consume"] = "pass"
    message_id: str = ""
    reply_reference: str = ""
    context_message_ids: list[str] = Field(default_factory=list)
    detail: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reply(self) -> TurnDecision:
        if self.action == "reply" and not (self.message_id.strip() and self.reply_reference.strip()):
            raise ValueError("reply requires message_id and reply_reference")
        return self


class TurnDecisionArguments(BaseModel):
    request: dict[str, JsonValue]
    result: TurnDecision


class TurnCompletionArguments(BaseModel):
    session_id: str
    result: TurnDecision
    sent: bool
    timestamp: float


def build_turn_decision_hook_specs() -> list[HookSpec]:
    return [
        HookSpec(
            name=HOOK_NAME,
            description="消息防抖后决定回复、等待、静默已读或进入 Planner；由核心串行执行。",
            parameters_schema=TurnDecisionArguments.model_json_schema(),
            default_timeout_ms=12000,
            allow_observe=False,
        ),
        HookSpec(
            name="maisaka.turn.completed",
            description="决策已经应用；sent 表示原生回复流程至少成功发送一条消息。",
            parameters_schema=TurnCompletionArguments.model_json_schema(),
            allow_observe=False,
            allow_kwargs_mutation=False,
        ),
    ]


def has_turn_decision_owner(runtime: MaisakaHeartFlowChatting) -> bool:
    count = get_plugin_runtime_manager().hook_dispatcher.count_blocking_subscribers(HOOK_NAME)
    if count > 1:
        raise RuntimeError("Only one plugin may own maisaka.turn.decide")
    return (
        count == 1
        and runtime._agent_state != runtime._STATE_WAIT
        and not runtime._is_reply_frequency_silent()
        and not runtime._is_planner_continuation_active()
    )


def build_turn_request(runtime: MaisakaHeartFlowChatting, messages: list[SessionMessage]) -> dict[str, object]:
    stream = runtime.chat_stream
    return {
        "version": 1,
        "session_id": runtime.session_id,
        "platform": stream.platform,
        "group_id": stream.group_id or "",
        "is_group_session": stream.is_group_session,
        "bot_name": global_config.bot.nickname,
        "personality": global_config.personality.personality,
        "forced": runtime._has_forced_turn_trigger(),
        "timestamp": time.time(),
        "last_reply_at": next((item.timestamp.timestamp() for item in reversed(runtime._chat_history)
                               if item.source == "guided_reply"), 0.0),
        "pending_messages": [
            PluginMessageUtils._session_message_to_dict(message, include_binary_data=False) for message in messages
        ],
        "recent_messages": [
            {
                "message_id": item.message_id,
                "sender_id": item.original_message.message_info.user_info.user_id,
                "sender": (item.original_message.message_info.user_info.user_cardname
                           or item.original_message.message_info.user_info.user_nickname),
                "reply_to": item.original_message.reply_to,
                "text": item.processed_plain_text,
                "is_self": item.source == "guided_reply",
                "is_at": bool(item.original_message.is_at),
                "is_mentioned": bool(item.original_message.is_mentioned),
                "timestamp": item.timestamp.timestamp(),
            }
            for item in runtime._chat_history[-40:]
            if isinstance(item, SessionBackedMessage) and item.source in {"user", "guided_reply"}
        ],
    }


class PluginTurnDecision:
    def __init__(self, scheduler: MessageTurnScheduler) -> None:
        self._scheduler = scheduler
        self._runtime = scheduler._runtime
        self._waiting_message: SessionMessage | None = None

    def is_waiting_for_new_message(self) -> bool:
        return self._waiting_message is self._runtime.message_cache[-1]

    async def run(self) -> bool:
        """Return false only when the existing Planner turn should continue."""
        runtime = self._runtime
        if not has_turn_decision_owner(runtime):
            return False
        messages = list(runtime.message_cache[runtime._last_processed_index:])
        if not messages:
            runtime._mark_message_turn_unscheduled()
            return True
        stream = getattr(runtime, "chat_stream", None)
        if stream is not None and not stream.is_group_session:
            decision = TurnDecision(
                action="reply",
                message_id=messages[-1].message_id,
                reply_reference=PRIVATE_DIRECT_REPLY_REFERENCE,
                detail="私聊直接回复",
                metadata={"provider": "direct", "conversation": "active"},
            )
            logger.info(f"{runtime.log_prefix} 私聊直接回复 待处理={len(messages)} target={decision.message_id}")
            await _record_turn_decision_event(runtime, decision, len(messages), 0.0)
            self._waiting_message = None
            await apply_decision(runtime, decision, messages, notify_plugin=False)
            return True
        index = runtime._last_processed_index
        started_at = time.perf_counter()
        runtime._update_stage_status("回复判定", f"正在评估 {len(messages)} 条消息")
        try:
            result = await get_plugin_runtime_manager().invoke_hook(
                HOOK_NAME, request=build_turn_request(runtime, messages), result=TurnDecision().model_dump()
            )
            if result.errors or result.aborted:
                raise RuntimeError(f"Turn decision failed: {result.errors}")
            decision = TurnDecision.model_validate(result.kwargs["result"])
        except BaseException:
            runtime._update_stage_status("空闲", "回复判定异常中断")
            raise
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        # New arrivals invalidate the whole batch, including uncommitted plugin state.
        if index != runtime._last_processed_index or runtime.message_cache[-1] is not messages[-1]:
            runtime._mark_message_turn_unscheduled()
            runtime._schedule_message_turn()
            return True
        logger.info(f"{runtime.log_prefix} 插件判定={decision.action} 待处理={len(messages)} {decision.detail}")
        if decision.action == "pass":
            should_stop = self._run_native_gate(len(messages))
            if should_stop:
                runtime._update_stage_status("空闲", "等待后续消息")
            return should_stop
        await _record_turn_decision_event(runtime, decision, len(messages), elapsed_ms)
        if decision.action == "planner":
            await complete_decision(runtime, decision, sent=False)
            return False
        if decision.action == "wait":
            self._waiting_message = messages[-1]
            runtime._mark_message_turn_unscheduled()
            runtime._update_stage_status("空闲", "回复判定：等待后续消息")
            await complete_decision(runtime, decision, sent=False)
            return True
        self._waiting_message = None
        await apply_decision(runtime, decision, messages)
        return True

    def _run_native_gate(self, pending_count: int) -> bool:
        runtime = self._runtime
        result = self._scheduler._evaluate(pending_count)
        if result.should_trigger:
            return False
        runtime._mark_message_turn_unscheduled()
        if result.decision == "delay":
            runtime._defer_message_turn_check(result.delay_seconds)
        else:
            self._waiting_message = runtime.message_cache[-1]
        return True


async def apply_decision(
    runtime: MaisakaHeartFlowChatting,
    decision: TurnDecision,
    messages: list[SessionMessage],
    *,
    notify_plugin: bool = True,
) -> None:
    target = None
    if decision.action == "reply":
        target = next((message for message in messages if message.message_id == decision.message_id), None)
        if target is None:
            raise ValueError("Turn decision reply target is outside the evaluated message batch")
    collected = runtime._collect_pending_messages()
    runtime._consume_forced_turn_reason()
    try:
        await runtime._reasoning_engine._ingest_messages(collected)
        await runtime._reasoning_engine._emit_flow_step(
            "messages.merged", "接入新消息", f"插件判定接入 {len(messages)} 条消息。",
            cycle_id=runtime._cycle_counter,
        )
        sent = await reply_from_decision(runtime, decision, messages, target) if target is not None else False
        if target is None:
            from src.chat.heart_flow.heartFC_utils import CycleDetail

            await runtime._reasoning_engine._post_process_chat_history_after_cycle(
                CycleDetail(cycle_id=runtime._cycle_counter), enable_compaction=False,
            )
            runtime._update_stage_status("空闲", "回复判定：静默已读")
        if not sent and messages:
            from src.services.memory_flow_service import memory_automation_service

            await memory_automation_service.chat_summary_writeback.enqueue(messages[-1])
        if notify_plugin:
            await complete_decision(runtime, decision, sent=sent)
    finally:
        runtime._mark_message_turn_unscheduled()
        runtime._schedule_message_turn()


async def reply_from_decision(
    runtime: MaisakaHeartFlowChatting, decision: TurnDecision, messages: list[SessionMessage], target: SessionMessage
) -> bool:
    # Import here to keep the reasoning engine / scheduler dependency acyclic.
    from src.maisaka.reasoning_engine import CycleEnd, CycleRuntimeState

    engine = runtime._reasoning_engine
    runtime._enter_running_state()
    cycle, round_text = await engine._start_cycle_round(0)
    state = CycleRuntimeState(reply_target_message=target, planner_extra_lines=[decision.detail])
    try:
        await engine._refresh_visual_placeholders_for_cycle(cycle)
        await engine._post_process_chat_history_after_cycle(cycle, enable_compaction=False, consume_references=False)
        allowed_ids = set(decision.context_message_ids)
        memory_sources = [m for m in messages if m.message_id in allowed_ids] if allowed_ids else messages
        memory = await engine._build_planner_injected_user_messages(
            profile_message=target,
            source_messages=memory_sources or [target],
            deferred_tools_reminder="",
            round_text=round_text,
        )
        await engine._reply_without_tool_call(
            decision.reply_reference,
            cycle,
            state,
            use_supplement=False,
            extra_info="\n\n".join(item for item in memory if item.strip()),
            context_message_ids=decision.context_message_ids,
        )
        return state.direct_reply_sent
    except BaseException:
        state.cycle_end = CycleEnd("turn_decision_error", "插件直接回复失败或取消。")
        raise
    finally:
        try:
            await engine._finalize_cycle(cycle_detail=cycle, round_index=0, state=state)
        finally:
            runtime._enter_stop_state()
            runtime._update_stage_status("空闲", state.cycle_end.detail)


async def _record_turn_decision_event(
    runtime: MaisakaHeartFlowChatting,
    decision: TurnDecision,
    pending_count: int,
    fallback_duration_ms: float,
) -> None:
    meta = decision.metadata
    raw_elapsed = meta.get("elapsed_ms")
    duration_ms = float(raw_elapsed) if isinstance(raw_elapsed, (int, float)) and raw_elapsed > 0 else fallback_duration_ms
    probabilities = meta.get("probabilities")
    await emit_turn_decision(
        session_id=runtime.session_id,
        action=decision.action,
        target_message_id=decision.message_id,
        detail=decision.detail,
        pending_count=pending_count,
        duration_ms=duration_ms,
        provider=str(meta.get("provider") or ""),
        model_name=str(meta.get("model") or ""),
        tone=str(meta.get("tone") or ""),
        conversation=str(meta.get("conversation") or ""),
        probabilities=probabilities if isinstance(probabilities, dict) else {},
    )


async def complete_decision(runtime: MaisakaHeartFlowChatting, decision: TurnDecision, *, sent: bool) -> None:
    result = await get_plugin_runtime_manager().invoke_hook(
        "maisaka.turn.completed", session_id=runtime.session_id,
        result=decision.model_dump(), sent=sent, timestamp=time.time(),
    )
    if result.errors or result.aborted:
        raise RuntimeError(f"Turn completion notification failed: {result.errors}")
