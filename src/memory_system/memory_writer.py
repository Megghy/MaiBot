from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.common.logger import get_logger
from src.person_info.person_info import Person, infer_memory_category

logger = get_logger("memory_writer")


@dataclass
class MemoryWriteRequest:
    """封装一次记忆写入所需的信息"""

    person: Person
    memory_content: str
    category: str
    weight: float = 1.0
    source: Optional[str] = None
    context: Optional[str] = None
    group_id: Optional[str] = None


def _sanitize_memory_text(text: str, *, max_length: int = 200) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip("，,。；; ") + "…"


def resolve_target_person(action_message, chat_stream) -> Optional[Person]:
    """从 action_message 或 chat_stream 中还原目标 Person"""

    try:
        if action_message and getattr(action_message, "user_info", None):
            user_info = action_message.user_info
            return Person(platform=user_info.platform, user_id=user_info.user_id)

        if chat_stream and getattr(chat_stream, "user_info", None):
            user_info = chat_stream.user_info
            if user_info:
                return Person(platform=user_info.platform, user_id=user_info.user_id)
    except Exception as exc:
        logger.warning(f"解析目标Person失败: {exc}")

    return None


def extract_group_id(action_message, chat_stream) -> Optional[str]:
    group_id = None
    try:
        if action_message and getattr(action_message, "chat_info", None):
            group_info = getattr(action_message.chat_info, "group_info", None)
            if group_info and getattr(group_info, "group_id", None):
                group_id = str(group_info.group_id)
    except Exception:
        group_id = None

    if not group_id and chat_stream and getattr(chat_stream, "group_info", None):
        try:
            group_id = str(chat_stream.group_info.group_id)  # type: ignore[attr-defined]
        except Exception:
            group_id = None
    return group_id


def build_memory_write_request(
    *,
    action_message,
    chat_stream,
    candidate_texts: Sequence[Optional[str]],
    category: Optional[str] = None,
    default_category: Optional[str] = None,
    weight: Optional[float] = None,
    source: Optional[str] = None,
    context: Optional[str] = None,
    person: Optional[Person] = None,
    group_id: Optional[str] = None,
) -> Optional[MemoryWriteRequest]:
    """根据上下文生成 MemoryWriteRequest，若缺少必要信息则返回 None"""

    person = person or resolve_target_person(action_message, chat_stream)
    if not person or not person.is_known:
        logger.debug("无法确定已认识的目标用户，跳过记忆写入")
        return None

    memory_text = ""
    for text in candidate_texts:
        sanitized = _sanitize_memory_text(text)
        if sanitized:
            memory_text = sanitized
            break

    if not memory_text:
        logger.debug("没有可写入的记忆文本")
        return None

    resolved_category = category or default_category or infer_memory_category(memory_text)
    resolved_category = resolved_category or "记忆"

    try:
        weight_value = float(weight) if weight is not None else 1.0
    except (TypeError, ValueError):
        weight_value = 1.0

    request = MemoryWriteRequest(
        person=person,
        memory_content=memory_text,
        category=resolved_category,
        weight=max(0.1, min(weight_value, 5.0)),
        source=source,
        context=_sanitize_memory_text(context, max_length=120) if context else None,
        group_id=group_id or extract_group_id(action_message, chat_stream),
    )
    return request


def persist_memory_request(request: MemoryWriteRequest) -> bool:
    """根据请求写入记忆，成功返回 True"""

    try:
        if request.group_id:
            added = request.person.add_group_memory_point(
                request.group_id,
                request.memory_content,
                category=request.category,
                weight=request.weight,
            )
        else:
            added = request.person.add_memory_point(
                request.memory_content,
                category=request.category,
                weight=request.weight,
                context=request.context,
                source=request.source,
            )
        if not added:
            logger.debug("记忆写入失败，可能是重复内容")
        return bool(added)
    except Exception as exc:
        logger.error(f"写入记忆失败: {exc}")
        return False
