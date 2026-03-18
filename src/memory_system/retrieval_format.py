import json
import re
from typing import Any, Dict, List, Optional

from src.common.logger import get_logger
from src.config.config import global_config

logger = get_logger("memory_retrieval")

MAX_COLLECTED_INFO_LENGTH = 2000
COLLECTED_INFO_TRUNCATE_NOTICE = "…（已截断，仅保留最近信息）\n"


def _safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _truncate_text(text: str, limit: int = 160) -> str:
    """对文本进行截断，避免日志或prompt过长"""

    if not text:
        return ""

    text = text.strip()
    if len(text) <= limit:
        return text
    ellipsis = "…"
    keep = max(0, limit - len(ellipsis))
    return text[:keep] + ellipsis


def _format_action_repr(action: Dict[str, Any]) -> str:
    action_type = action.get("action_type", "")
    params = action.get("action_params") or {}
    if not params:
        return action_type

    display_params = []
    for key, value in params.items():
        if key == "chat_id":
            continue
        value_str = _truncate_text(str(value), 40)
        display_params.append(f"{key}={value_str}")

    param_str = ", ".join(display_params)
    return f"{action_type}({param_str})" if display_params else action_type


def _format_agent_summary(thinking_steps: List[Dict[str, Any]], max_steps: int = 3) -> str:
    if not thinking_steps:
        return ""

    summary_lines: List[str] = []
    for step in thinking_steps[:max_steps]:
        iteration = step.get("iteration")
        line_parts = [f"第{iteration}轮"] if iteration is not None else []

        reasoning = step.get("reasoning") or step.get("thought") or ""
        if reasoning:
            line_parts.append(f"思考：{_truncate_text(reasoning, 80)}")

        actions = step.get("actions") or []
        if actions:
            action_text = "；".join(_format_action_repr(action) for action in actions)
            line_parts.append(f"动作：{action_text}")

        observations = step.get("observations") or []
        if observations:
            line_parts.append(f"观察：{_truncate_text(observations[0], 80)}")

        summary_lines.append("；".join(line_parts))

    if len(thinking_steps) > max_steps:
        summary_lines.append("……")

    return "\n".join(summary_lines)


def _has_successful_observation(thinking_steps: List[Dict[str, Any]]) -> bool:
    for step in thinking_steps or []:
        observations = step.get("observations") or []
        for observation in observations:
            text = _safe_str(observation)
            if not text:
                continue
            parts = text.split("：", 1)
            payload = parts[1].strip() if len(parts) == 2 else text
            try:
                data = json.loads(payload)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("success") is True:
                return True
    return False


def _extract_answer_source(thinking_steps: List[Dict[str, Any]]) -> str:
    for step in reversed(thinking_steps):
        for action in step.get("actions", []):
            if action.get("action_type") == "found_answer":
                params = action.get("action_params") or {}
                return params.get("source", "")
    return ""


def _describe_source(raw_source: Optional[str], *, from_cache: bool = False) -> str:
    if from_cache:
        return "历史缓存"

    if not raw_source:
        return "检索记忆"

    mapping = {"memory": "检索记忆", "model": "模型常识"}
    return mapping.get(raw_source, raw_source)


def _build_answer_block(
    question: str,
    answer: str,
    *,
    source_label: str,
    agent_summary: Optional[str] = None,
    extra_note: Optional[str] = None,
) -> str:
    lines = [
        f"问题：{question}",
        f"答案：{answer}",
    ]

    if source_label:
        lines.append(f"来源：{source_label}")

    if extra_note:
        lines.append(f"备注：{extra_note}")

    if agent_summary:
        lines.append("Agent步骤：")
        lines.append(agent_summary)

    return "\n".join(lines)


def _load_thinking_steps(raw_steps: Optional[str]) -> List[Dict[str, Any]]:
    if not raw_steps:
        return []

    try:
        loaded = json.loads(raw_steps)
        if isinstance(loaded, list):
            return loaded
    except Exception as exc:  # pragma: no cover - 防御性日志
        logger.warning(f"解析thinking_steps失败: {exc}")
    return []


def _format_answer_payload(
    question: str,
    answer: str,
    thinking_steps: Optional[List[Dict[str, Any]]] = None,
    *,
    from_cache: bool = False,
    extra_note: Optional[str] = None,
) -> str:
    steps = thinking_steps or []
    raw_source = _extract_answer_source(steps)
    source_label = _describe_source(raw_source, from_cache=from_cache)
    agent_summary = _format_agent_summary(steps)
    return _build_answer_block(
        question,
        answer,
        source_label=source_label,
        agent_summary=agent_summary or None,
        extra_note=extra_note,
    )


def _append_collected_info(current: str, addition: str) -> str:
    addition = (addition or "").strip()
    if not addition:
        return current

    combined = f"{current}\n{addition}" if current else addition
    combined = combined.strip()
    if len(combined) <= MAX_COLLECTED_INFO_LENGTH:
        return combined

    truncated = combined[-MAX_COLLECTED_INFO_LENGTH:]
    return f"{COLLECTED_INFO_TRUNCATE_NOTICE}{truncated}"


def _prepare_collected_info_for_prompt(collected_info: str, limit: int = 1800) -> str:
    if not collected_info:
        return "暂无信息"

    if len(collected_info) <= limit:
        return collected_info

    trimmed = collected_info[-limit:]
    return f"{COLLECTED_INFO_TRUNCATE_NOTICE}{trimmed}"


def _calculate_iteration_budget(question: str, initial_info: str) -> int:
    base = max(1, getattr(global_config.memory, "max_agent_iterations", 3))
    if base <= 2:
        return base

    normalized_question = (question or "").strip()
    complexity_score = 0

    if len(normalized_question) > 48:
        complexity_score += 1
    if len(re.findall(r"[?？]", normalized_question)) > 1:
        complexity_score += 1
    detail_keywords = ("为什么", "如何", "怎么", "原因", "步骤", "哪些", "哪几")
    if any(keyword in normalized_question for keyword in detail_keywords):
        complexity_score += 1
    if len(normalized_question) <= 16:
        complexity_score -= 1
    if initial_info:
        complexity_score -= 1

    if complexity_score <= -1:
        return 1
    if complexity_score == 0:
        return min(base, 2)
    return base


def _extract_question_from_block(block: str) -> Optional[str]:
    for line in block.splitlines():
        if line.startswith("问题："):
            return line.replace("问题：", "", 1).strip()
    return None
