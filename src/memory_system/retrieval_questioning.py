import json
import re
import time
from typing import Any, Dict, List, Tuple

from json_repair import repair_json

from src.common.logger import get_logger
from src.person_info.person_info import Person
from src.plugin_system.apis import database_api, message_api

logger = get_logger("memory_retrieval")


async def _record_question_generation_trace(
    chat_stream,
    question_prompt: str,
    response: str,
    concepts: List[str],
    questions: List[str],
) -> None:
    try:
        action_data = {
            "prompt": question_prompt,
            "response": response,
            "concepts": concepts,
            "questions": questions,
        }
        await database_api.store_action_info(
            chat_stream=chat_stream,
            action_name="memory_question_generation",
            action_data=action_data,
            action_reasoning="question_generation",
        )
    except Exception as exc:
        logger.warning(f"记录问题生成追踪失败: {exc}")


def _build_participant_hints(chat_id: str, *, limit: int = 10) -> str:
    """聚合最近聊天参与者与人物记忆摘要，辅助question prompt理解指代"""
    try:
        end_time = time.time()
        start_time = max(0.0, end_time - 3600)
        recent_messages = message_api.get_messages_by_time_in_chat(
            chat_id=chat_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            filter_mai=False,
            filter_command=True,
        )
    except Exception as exc:
        logger.warning(f"获取参与者信息失败: {exc}")
        recent_messages = []

    participants: Dict[str, Dict[str, Any]] = {}
    for msg in recent_messages:
        user_info = msg.user_info
        if not user_info:
            continue
        key = f"{user_info.platform}:{user_info.user_id}"
        if key in participants:
            continue
        person = Person(platform=user_info.platform, user_id=user_info.user_id)
        if not person.person_name:
            continue
        memory_preview = ""
        if person.memory_points:
            sample = person.memory_points[0]
            memory_preview = sample if isinstance(sample, str) else ""

        participants[key] = {
            "name": person.person_name,
            "known": person.is_known,
            "memory": memory_preview,
        }

    if not participants:
        return "最近未识别出其他角色。"

    lines = ["最近角色线索："]
    for info in participants.values():
        note = "已登记" if info.get("known") else "未知"
        memory = info.get("memory")
        memory_str = f"；记忆片段：{memory}" if memory else ""
        lines.append(f"- {info['name']}（{note}）{memory_str}")

    return "\n".join(lines)


def _parse_questions_json(response: str) -> Tuple[List[str], List[str]]:
    """解析问题JSON，返回概念列表和问题列表"""
    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        json_str = matches[0] if matches else response.strip()
        repaired_json = repair_json(json_str)
        parsed = json.loads(repaired_json)

        if not isinstance(parsed, dict):
            logger.warning(f"解析的JSON不是对象格式: {parsed}")
            return [], []

        concepts_raw = parsed.get("concepts", [])
        questions_raw = parsed.get("questions", [])
        if not isinstance(concepts_raw, list):
            concepts_raw = []
        if not isinstance(questions_raw, list):
            questions_raw = []

        concepts = [c for c in concepts_raw if isinstance(c, str) and c.strip()]
        questions = [q for q in questions_raw if isinstance(q, str) and q.strip()]
        return concepts, questions
    except Exception as exc:
        logger.error(f"解析问题JSON失败: {exc}, 响应内容: {response[:200]}...")
        return [], []
