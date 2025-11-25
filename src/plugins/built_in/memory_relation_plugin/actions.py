from __future__ import annotations

from typing import Optional, Sequence, Tuple

from src.common.logger import get_logger
from src.memory_system.memory_writer import (
    build_memory_write_request,
    persist_memory_request,
    resolve_target_person,
)
from src.plugin_system import BaseAction, ActionActivationType

logger = get_logger("memory_relation_actions")


def _extract_message_text(message) -> str:
    if not message:
        return ""
    text = getattr(message, "processed_plain_text", None) or getattr(message, "display_message", None)
    return (text or "").strip()


def _extract_chat_context(action_message, fallback: str = "") -> str:
    message_text = _extract_message_text(action_message)
    if message_text:
        return message_text
    return fallback.strip()


async def _persist_memory_from_candidates(
    action: BaseAction,
    *,
    candidate_texts: Sequence[Optional[str]],
    category: Optional[str] = None,
    default_category: Optional[str] = None,
    weight: Optional[float] = None,
    source: Optional[str] = None,
    context: Optional[str] = None,
    person=None,
    group_id: Optional[str] = None,
) -> Tuple[bool, str]:
    request = build_memory_write_request(
        action_message=action.action_message,
        chat_stream=action.chat_stream,
        candidate_texts=candidate_texts,
        category=category,
        default_category=default_category,
        weight=weight,
        source=source,
        context=context,
        person=person,
        group_id=group_id,
    )

    if not request:
        logger.debug("%s: 没有可写入的记忆内容", action.action_name)
        return False, "没有可写入的记忆内容"

    persisted = persist_memory_request(request)
    if not persisted:
        return False, "写入记忆失败，可能是重复内容"

    person_name = request.person.person_name or request.person.nickname or request.person.user_id or "对方"
    preview = request.memory_content[:48] + ("…" if len(request.memory_content) > 48 else "")
    await action.store_action_info(
        action_build_into_prompt=True,
        action_prompt_display=f"你记录了关于 {person_name} 的记忆：{preview}",
        action_done=True,
    )
    return True, f"已记录关于 {person_name} 的记忆"


class BuildMemoryAction(BaseAction):
    activation_type = ActionActivationType.LLM_JUDGE
    parallel_action = False
    action_name = "build_memory"
    action_description = "根据当前对话内容记录一条重要记忆，避免重复信息。"
    action_parameters = {
        "memory_summary": "可选，LLM 对记忆点的概括文本",
        "category": "可选，记忆分类（如兴趣、关系、事件等）",
        "weight": "可选，记忆权重，范围建议 0.1~5",
        "source": "可选，记忆来源描述",
        "context": "可选，附加上下文或原文片段",
    }
    action_require = [
        "当聊天中出现可以长期记住的信息（兴趣、事件、偏好等）时使用",
        "不要重复记录相同内容",
        "如果不确定信息是否有价值，请不要调用",
    ]

    async def execute(self) -> Tuple[bool, str]:
        candidate_texts = [
            self.action_data.get("memory_summary"),
            self.action_data.get("memory_content"),
            self.action_reasoning,
            self.reasoning,
            _extract_message_text(self.action_message),
        ]

        return await _persist_memory_from_candidates(
            self,
            candidate_texts=candidate_texts,
            category=self.action_data.get("category"),
            weight=self.action_data.get("weight"),
            source=self.action_data.get("source"),
            context=self.action_data.get("context"),
        )


class BuildRelationAction(BaseAction):
    activation_type = ActionActivationType.LLM_JUDGE
    parallel_action = False
    action_name = "build_relation"
    action_description = "分析当前聊天中体现的关系动态，并记录到人物记忆中。"
    action_parameters = {
        "relation_summary": "可选，对关系变化的总结",
        "focus_hint": "可选，引导记录的关系维度（如亲近、冲突、感谢等）",
        "category": "可选，记忆分类，默认使用‘关系’",
        "weight": "可选，关系记忆权重",
        "context": "可选，额外上下文",
    }
    action_require = [
        "当聊天涉及双方关系变化、情绪或承诺时使用",
        "优先记录客观事实或明显态度",
        "不要重复记录同样的关系描述",
    ]

    async def execute(self) -> Tuple[bool, str]:
        person = resolve_target_person(self.action_message, self.chat_stream)
        if not person or not person.is_known:
            return False, "无法识别目标人物"

        chat_context = self.action_data.get("chat_context") or _extract_chat_context(
            self.action_message, fallback=self.action_reasoning or ""
        )
        
        try:
            # Use new method in Person
            summary = await person.record_relation(chat_context, source="Action:build_relation")
            
            if summary:
                await self.store_action_info(
                    action_build_into_prompt=True,
                    action_prompt_display=f"已更新关系记忆：{summary}",
                    action_done=True,
                )
                return True, f"已记录关系：{summary}"
            else:
                return False, "未生成有效的关系摘要或无需记录"
        except Exception as e:
            logger.error(f"BuildRelationAction failed: {e}")
            return False, f"记录关系失败: {e}"
