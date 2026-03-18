import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.common.database.database_model import ThinkingBack
from src.common.logger import get_logger
from src.memory_system.retrieval_format import _format_answer_payload, _load_thinking_steps

logger = get_logger("memory_retrieval")

THINKING_BACK_NOT_FOUND_RETENTION_SECONDS = 36000
THINKING_BACK_CLEANUP_INTERVAL_SECONDS = 3000

_last_not_found_cleanup_ts: float = 0.0


def _cleanup_stale_not_found_thinking_back() -> None:
    """定期清理过期的未找到答案记录"""
    global _last_not_found_cleanup_ts

    now = time.time()
    if now - _last_not_found_cleanup_ts < THINKING_BACK_CLEANUP_INTERVAL_SECONDS:
        return

    threshold_time = now - THINKING_BACK_NOT_FOUND_RETENTION_SECONDS
    try:
        deleted_rows = (
            ThinkingBack.delete()
            .where((ThinkingBack.found_answer == 0) & (ThinkingBack.update_time < threshold_time))
            .execute()
        )
        if deleted_rows:
            logger.info(f"清理过期的未找到答案thinking_back记录 {deleted_rows} 条")
        _last_not_found_cleanup_ts = now
    except Exception as exc:
        logger.error(f"清理未找到答案的thinking_back记录失败: {exc}")


def _get_recent_query_history(chat_id: str, time_window_seconds: float = 300.0) -> str:
    """获取最近一段时间内的查询历史"""
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        records = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.update_time >= start_time))
            .order_by(ThinkingBack.update_time.desc())
            .limit(5)
        )

        if not records.exists():
            return ""

        history_lines = ["最近已查询的问题和结果："]
        for record in records:
            status = "✓ 已找到答案" if record.found_answer else "✗ 未找到答案"
            answer_preview = ""
            if record.found_answer and record.answer:
                answer_preview = record.answer[:100]
                if len(record.answer) > 100:
                    answer_preview += "..."

            history_lines.append(f"- 问题：{record.question}")
            history_lines.append(f"  状态：{status}")
            if answer_preview:
                history_lines.append(f"  答案：{answer_preview}")
            history_lines.append("")

        return "\n".join(history_lines)
    except Exception as exc:
        logger.error(f"获取查询历史失败: {exc}")
        return ""


def _get_cached_memories(chat_id: str, time_window_seconds: float = 300.0) -> List[str]:
    """获取最近一段时间内缓存的记忆（只返回找到答案的记录）"""
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        records = (
            ThinkingBack.select()
            .where(
                (ThinkingBack.chat_id == chat_id)
                & (ThinkingBack.update_time >= start_time)
                & (ThinkingBack.found_answer == 1)
            )
            .order_by(ThinkingBack.update_time.desc())
            .limit(5)
        )

        if not records.exists():
            return []

        cached_memories: List[str] = []
        for record in records:
            if not record.answer:
                continue

            thinking_steps = _load_thinking_steps(record.thinking_steps)
            cached_memories.append(
                _format_answer_payload(
                    record.question,
                    record.answer,
                    thinking_steps,
                    from_cache=True,
                )
            )

        return cached_memories
    except Exception as exc:
        logger.error(f"获取缓存记忆失败: {exc}")
        return []


def _query_thinking_back(chat_id: str, question: str) -> Optional[Tuple[bool, str, List[Dict[str, Any]]]]:
    """从thinking_back数据库中查询是否有现成的答案"""
    try:
        records = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.question == question))
            .order_by(ThinkingBack.update_time.desc())
            .limit(1)
        )

        if records.exists():
            record = records.get()
            found_answer = bool(record.found_answer)
            answer = record.answer or ""
            steps = _load_thinking_steps(record.thinking_steps)
            logger.info(f"在thinking_back中找到记录，问题: {question[:50]}...，found_answer: {found_answer}")
            return found_answer, answer, steps

        return None
    except Exception as exc:
        logger.error(f"查询thinking_back失败: {exc}")
        return None


def _store_thinking_back(
    chat_id: str,
    question: str,
    context: str,
    found_answer: bool,
    answer: str,
    thinking_steps: List[Dict[str, Any]],
) -> None:
    """存储或更新思考过程到数据库（如果已存在则更新，否则创建）"""
    try:
        now = time.time()

        existing = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.question == question))
            .order_by(ThinkingBack.update_time.desc())
            .limit(1)
        )

        if existing.exists():
            record = existing.get()
            record.context = context
            record.found_answer = found_answer
            record.answer = answer
            record.thinking_steps = json.dumps(thinking_steps, ensure_ascii=False)
            record.update_time = now
            record.save()
            logger.info(f"已更新思考过程到数据库，问题: {question[:50]}...")
        else:
            ThinkingBack.create(
                chat_id=chat_id,
                question=question,
                context=context,
                found_answer=found_answer,
                answer=answer,
                thinking_steps=json.dumps(thinking_steps, ensure_ascii=False),
                create_time=now,
                update_time=now,
            )
            logger.info(f"已创建思考过程到数据库，问题: {question[:50]}...")
    except Exception as exc:
        logger.error(f"存储思考过程失败: {exc}")
