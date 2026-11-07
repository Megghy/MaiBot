import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from src.maisaka.turn_trigger import plugin_decision as decisions
from src.maisaka.turn_trigger import scheduler as scheduling
from src.maisaka.turn_trigger.gates import TurnGateResult
from src.maisaka.turn_trigger.scheduler import MessageTurnScheduler


@pytest.fixture
def turn(monkeypatch):
    message = SimpleNamespace(message_id="one")
    runtime = SimpleNamespace(
        session_id="group", message_cache=[message], _last_processed_index=0,
        _agent_state="stop", _STATE_WAIT="wait", _cycle_counter=1, log_prefix="[test]",
        _is_reply_frequency_silent=lambda: False, _is_planner_continuation_active=lambda: False,
        _mark_message_turn_unscheduled=Mock(), _schedule_message_turn=Mock(),
        _defer_message_turn_check=Mock(), _update_stage_status=Mock(),
        _consume_forced_turn_reason=Mock(), _collect_pending_messages=Mock(return_value=[message]),
        _reasoning_engine=SimpleNamespace(
            _ingest_messages=AsyncMock(), _emit_flow_step=AsyncMock(),
            _post_process_chat_history_after_cycle=AsyncMock(),
        ),
    )
    manager = SimpleNamespace(
        hook_dispatcher=SimpleNamespace(count_blocking_subscribers=Mock(return_value=1)),
        invoke_hook=AsyncMock(return_value=SimpleNamespace(
            errors=[], aborted=False, kwargs={"result": {"action": "wait"}},
        )),
    )
    monkeypatch.setattr(decisions, "get_plugin_runtime_manager", lambda: manager)
    monkeypatch.setattr(decisions, "build_turn_request", Mock(return_value={"version": 1}))
    monkeypatch.setattr(decisions, "emit_turn_decision", AsyncMock())
    scheduler = SimpleNamespace(_runtime=runtime, _evaluate=Mock())
    return runtime, manager, decisions.PluginTurnDecision(scheduler), scheduler


def test_single_owner_and_lifecycle_contract(turn):
    runtime, manager, _, _ = turn
    assert decisions.has_turn_decision_owner(runtime)
    runtime._agent_state = "wait"
    assert not decisions.has_turn_decision_owner(runtime)
    runtime._agent_state = "stop"
    runtime._is_planner_continuation_active = lambda: True
    assert not decisions.has_turn_decision_owner(runtime)
    runtime._is_planner_continuation_active = lambda: False
    runtime._is_reply_frequency_silent = lambda: True
    assert not decisions.has_turn_decision_owner(runtime)
    manager.hook_dispatcher.count_blocking_subscribers.return_value = 2
    with pytest.raises(RuntimeError, match="Only one"):
        decisions.has_turn_decision_owner(runtime)


@pytest.mark.asyncio
async def test_stale_snapshot_requeues_without_applying_plugin_state(turn):
    runtime, manager, policy, _ = turn

    async def decide(*args, **kwargs):
        runtime.message_cache.append(SimpleNamespace(message_id="new"))
        return SimpleNamespace(errors=[], aborted=False, kwargs={"result": {"action": "consume"}})

    manager.invoke_hook.side_effect = decide
    assert await policy.run()
    assert manager.invoke_hook.await_count == 1
    runtime._collect_pending_messages.assert_not_called()
    runtime._schedule_message_turn.assert_called_once()


@pytest.mark.asyncio
async def test_wait_retains_input_until_a_new_message(turn):
    runtime, manager, policy, _ = turn
    manager.invoke_hook.return_value.kwargs["result"] = {
        "action": "wait",
        "detail": "Jev wait p=0.960",
        "metadata": {
            "provider": "jev",
            "model": "jev-1.14.0",
            "tone": "normal",
            "conversation": "active",
            "elapsed_ms": 420.5,
            "probabilities": {"action": {"reply": 0.03, "wait": 0.96, "consume": 0.01}},
        },
    }
    assert await policy.run()
    assert policy.is_waiting_for_new_message()
    runtime._collect_pending_messages.assert_not_called()
    decisions.emit_turn_decision.assert_awaited_once()
    emitted = decisions.emit_turn_decision.await_args.kwargs
    assert emitted["action"] == "wait"
    assert emitted["model_name"] == "jev-1.14.0"
    assert emitted["tone"] == "normal"
    assert emitted["duration_ms"] == 420.5
    assert emitted["probabilities"]["action"]["wait"] == 0.96
    assert manager.invoke_hook.await_args.args == ("maisaka.turn.completed",)
    runtime.message_cache.append(SimpleNamespace(message_id="new"))
    assert not policy.is_waiting_for_new_message()


@pytest.mark.asyncio
async def test_planner_handoff_preserves_pending_messages(turn):
    runtime, manager, policy, _ = turn
    manager.invoke_hook.return_value.kwargs["result"] = {"action": "planner"}
    assert not await policy.run()
    runtime._collect_pending_messages.assert_not_called()
    assert manager.invoke_hook.await_count == 2


@pytest.mark.asyncio
async def test_pass_restores_native_delay_without_another_model_request(turn):
    runtime, manager, policy, scheduler = turn
    manager.invoke_hook.return_value.kwargs["result"] = {"action": "pass"}
    scheduler._evaluate.return_value = TurnGateResult(decision="delay", delay_seconds=3, detail="backoff")
    assert await policy.run()
    runtime._defer_message_turn_check.assert_called_once_with(3)
    assert manager.invoke_hook.await_count == 1


@pytest.mark.asyncio
async def test_consume_uses_original_ingestion_and_compaction(turn):
    runtime, manager, _, _ = turn
    await decisions.apply_decision(runtime, decisions.TurnDecision(action="consume"), runtime.message_cache)
    runtime._reasoning_engine._ingest_messages.assert_awaited_once_with(runtime.message_cache)
    runtime._reasoning_engine._post_process_chat_history_after_cycle.assert_awaited_once()
    assert manager.invoke_hook.await_args.kwargs["sent"] is False
    runtime._schedule_message_turn.assert_called_once()


@pytest.mark.asyncio
async def test_invalid_reply_target_does_not_consume_messages(turn):
    runtime, _, _, _ = turn
    result = decisions.TurnDecision(action="reply", message_id="wrong", reply_reference="respond")
    with pytest.raises(ValueError, match="outside"):
        await decisions.apply_decision(runtime, result, runtime.message_cache)
    runtime._collect_pending_messages.assert_not_called()


@pytest.mark.asyncio
async def test_failed_reply_does_not_commit_decision_state(turn, monkeypatch):
    runtime, manager, _, _ = turn
    monkeypatch.setattr(decisions, "reply_from_decision", AsyncMock(side_effect=RuntimeError("failed")))
    result = decisions.TurnDecision(action="reply", message_id="one", reply_reference="respond")
    with pytest.raises(RuntimeError, match="failed"):
        await decisions.apply_decision(runtime, result, runtime.message_cache)
    manager.invoke_hook.assert_not_awaited()
    runtime._mark_message_turn_unscheduled.assert_called_once()
    runtime._schedule_message_turn.assert_called_once()


@pytest.mark.asyncio
async def test_model_or_completion_errors_propagate(turn):
    runtime, manager, policy, _ = turn
    manager.invoke_hook.return_value.errors = ["timeout"]
    with pytest.raises(RuntimeError, match="decision failed"):
        await policy.run()
    runtime._collect_pending_messages.assert_not_called()
    with pytest.raises(RuntimeError, match="completion notification failed"):
        await decisions.complete_decision(runtime, decisions.TurnDecision(), sent=False)


def test_hook_schema_validates_result_and_completion_contract():
    specs = decisions.build_turn_decision_hook_specs()
    assert specs[0].parameters_schema["properties"]["result"]["$ref"] == "#/$defs/TurnDecision"
    assert not specs[1].allow_kwargs_mutation
    assert specs[1].parameters_schema["required"] == ["session_id", "result", "sent", "timestamp"]
    with pytest.raises(ValidationError):
        decisions.TurnDecision(action="reply", message_id="one")


def test_debounce_precedes_plugin_and_wait_blocks_only_unchanged_batch(turn, monkeypatch):
    runtime, _, _, _ = turn
    monkeypatch.setattr(scheduling, "has_turn_decision_owner", lambda _: True)
    runtime._message_turn_scheduled = False
    runtime._get_pending_message_count = lambda: 1
    runtime._get_message_quiet_period_remaining_seconds = lambda: 1.5
    runtime._enqueue_message_turn = Mock()
    scheduler = MessageTurnScheduler(runtime)
    scheduler.schedule_message_turn()
    runtime._enqueue_message_turn.assert_not_called()
    runtime._defer_message_turn_check.assert_called_once_with(1.5)
    runtime._get_message_quiet_period_remaining_seconds = lambda: 0
    scheduler.schedule_message_turn()
    runtime._enqueue_message_turn.assert_called_once_with("decision")
    scheduler._plugin_decision._waiting_message = runtime.message_cache[-1]
    scheduler.schedule_message_turn()
    assert runtime._enqueue_message_turn.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("Jev unavailable"), asyncio.CancelledError()])
async def test_decision_failure_releases_queue_reservation_before_propagating(turn, failure):
    from src.maisaka.reasoning_engine import MaisakaReasoningEngine

    owner, _, _, _ = turn
    owner._running = True
    owner._internal_turn_queue = asyncio.Queue()
    owner._internal_turn_queue.put_nowait("decision")
    owner._message_turn_scheduler = SimpleNamespace(_plugin_decision=SimpleNamespace(run=AsyncMock(side_effect=failure)))
    owner._log_internal_loop_cancelled = Mock()
    engine = MaisakaReasoningEngine.__new__(MaisakaReasoningEngine)
    engine._runtime = owner
    with pytest.raises(type(failure)):
        await engine.run_loop()
    owner._mark_message_turn_unscheduled.assert_called_once()
    owner._collect_pending_messages.assert_not_called()


@pytest.mark.asyncio
async def test_private_session_bypasses_plugin_and_replies_directly(turn, monkeypatch):
    runtime, manager, policy, _ = turn
    runtime.chat_stream = SimpleNamespace(is_group_session=False)
    reply_mock = AsyncMock(return_value=True)
    monkeypatch.setattr(decisions, "reply_from_decision", reply_mock)
    assert await policy.run()
    manager.invoke_hook.assert_not_awaited()
    reply_mock.assert_awaited_once()
    called_decision = reply_mock.await_args.args[1]
    assert called_decision.action == "reply"
    assert called_decision.message_id == "one"
    assert called_decision.reply_reference == decisions.PRIVATE_DIRECT_REPLY_REFERENCE
    assert called_decision.context_message_ids == []


@pytest.mark.asyncio
async def test_build_retro_reply_history_filters_by_allowed_message_ids_before_hydration(monkeypatch):
    from src.maisaka import runtime as runtime_module
    from src.maisaka.runtime import MaisakaHeartFlowChatting

    messages = [
        SimpleNamespace(message_id="m-1", is_notify=False),
        SimpleNamespace(message_id="m-2", is_notify=False),
        SimpleNamespace(message_id="m-3", is_notify=False),
    ]
    monkeypatch.setattr(runtime_module, "find_messages", Mock(return_value=messages))
    monkeypatch.setattr(runtime_module, "select_messages_after_latest_clear_marker", lambda items: items)

    hydrated_ids: list[str] = []

    async def fake_build_history_message(message, source_kind):
        hydrated_ids.append(message.message_id)
        return SimpleNamespace(message_id=message.message_id, source_kind=source_kind)

    chatting = MaisakaHeartFlowChatting.__new__(MaisakaHeartFlowChatting)
    chatting.session_id = "group"
    chatting.chat_stream = SimpleNamespace(is_group_session=True)
    chatting._resolve_restored_message_source_kind = lambda _: "user"
    chatting._reasoning_engine = SimpleNamespace(_build_history_message=fake_build_history_message)

    result = await chatting.build_retro_reply_history(allowed_message_ids={"m-1", "m-3"})
    assert [item.message_id for item in result] == ["m-1", "m-3"]
    assert hydrated_ids == ["m-1", "m-3"]
