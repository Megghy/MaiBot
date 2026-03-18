import json
from typing import Any, Optional

from json_repair import repair_json

from src.common.logger import get_logger
from src.config.config import model_config
from src.llm_models.utils_model import LLMRequest
from src.person_info.memory_schema import extract_categories_from_response, get_memory_content_from_memory

logger = get_logger("person_info")

relation_selection_model = LLMRequest(
    model_set=model_config.model_task_config.utils_small,
    request_type="relation_selection",
)


def get_relation_status_dict(person) -> Optional[dict[str, Any]]:
    if not person.relation_status:
        return None
    try:
        loaded = json.loads(person.relation_status)
    except (json.JSONDecodeError, TypeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def get_relation_status_summary(person) -> Optional[str]:
    data = get_relation_status_dict(person)
    if not data:
        return None
    summary = data.get("summary")
    return str(summary).strip() if summary else None


async def build_relationship(person, chat_content: str = "", info_type: str = "") -> str:
    if not person.is_known:
        return ""

    nickname_str = ""
    if person.person_name != person.nickname:
        nickname_str = f"(ta在{person.platform}上的昵称是{person.nickname})"

    points_text = ""
    category_list = person.get_all_category()

    if chat_content:
        prompt = f"""当前聊天内容：
{chat_content}

分类列表：
{category_list}
**要求**：请你根据当前聊天内容，从以下分类中选择一个与聊天内容相关的分类，并用<>包裹输出，不要输出其他内容，不要输出引号或[]，严格用<>包裹：
例如:
<分类1><分类2><分类3>......
如果没有相关的分类，请输出<none>"""
        response, _ = await relation_selection_model.generate_response_async(prompt)
        category_list = extract_categories_from_response(response)
        if "none" not in category_list:
            for category in category_list:
                relevant_memory = await person._get_relevant_memories_by_category_vector(category, chat_content, 2)
                if relevant_memory:
                    random_memory_str = "\n".join(get_memory_content_from_memory(memory) for memory in relevant_memory)
                    points_text = f"有关 {category} 的内容：{random_memory_str}"
                    break
    elif info_type:
        prompt = f"""你需要获取用户{person.person_name}的 **{info_type}** 信息。

现有信息类别列表：
{category_list}
**要求**：请你根据**{info_type}**，从以下分类中选择一个与**{info_type}**相关的分类，并用<>包裹输出，不要输出其他内容，不要输出引号或[]，严格用<>包裹：
例如:
<分类1><分类2><分类3>......
如果没有相关的分类，请输出<none>"""
        response, _ = await relation_selection_model.generate_response_async(prompt)
        category_list = extract_categories_from_response(response)
        if "none" not in category_list:
            for category in category_list:
                relevant_memory = await person._get_relevant_memories_by_category_vector(category, info_type, 3)
                if relevant_memory:
                    random_memory_str = "\n".join(get_memory_content_from_memory(memory) for memory in relevant_memory)
                    points_text = f"有关 {category} 的内容：{random_memory_str}"
                    break
    else:
        for category in category_list:
            random_memories = person.get_random_memory_by_category(category, 1)
            if not random_memories:
                continue
            points_text = f"有关 {category} 的内容：{get_memory_content_from_memory(random_memories[0])}"
            break

    points_info = f"你还记得有关{person.person_name}的内容：{points_text}" if points_text else ""
    relation_summary = get_relation_status_summary(person)
    relation_status_text = (
        f"你目前对{person.person_name}的关系状态总结是：{relation_summary}" if relation_summary else ""
    )

    if not (nickname_str or points_info or relation_status_text):
        return ""

    parts: list[str] = []
    if nickname_str:
        parts.append(nickname_str)
    if relation_status_text:
        parts.append(relation_status_text)
    if points_info:
        parts.append(points_info)
    return f"{person.person_name}:" + "".join(parts)


async def record_relation(person, chat_context: str, source: str = None) -> str:
    if not person.is_known:
        return ""

    retrieval_context = await build_relationship(person, chat_content=chat_context)
    prompt = f"""根据以下聊天内容和已有记忆，总结你与用户 {person.person_name} 的关系变化或当前状态。

聊天内容:
{chat_context}

{retrieval_context}

请你输出一个 JSON 对象，包含以下字段：
- "summary": 必填，一句话关系总结，尽量简短。
- "level": 可选，关系亲近程度标签，例如："陌生人"、"普通朋友"、"好朋友"、"亲密"、"疏远"、"紧张"、"敌对"、"未知" 等。
- "tags": 可选，字符串数组，列出 0~5 个简短标签，例如 ["经常聊天","互相吐槽"]。

如果没有值得记录的关系变化，请输出：{{"summary": "无"}}

只输出 JSON，不要输出其它任何解释或前后缀。
"""
    response, _ = await relation_selection_model.generate_response_async(prompt)
    raw = (response or "").strip()
    if not raw:
        return ""

    try:
        fixed = repair_json(raw)
        data = json.loads(fixed) if isinstance(fixed, str) else fixed
    except Exception as exc:
        logger.warning(f"解析关系状态 JSON 失败，将使用纯文本结果: {exc}")
        data = {"summary": raw}

    if not isinstance(data, dict):
        logger.warning(f"关系状态 JSON 结构非字典，忽略: {data}")
        return ""

    summary = str(data.get("summary") or "").strip()
    if not summary or summary == "无":
        return ""

    normalized: dict[str, Any] = {"summary": summary}
    if isinstance(data.get("level"), str) and data.get("level").strip():
        normalized["level"] = data["level"].strip()
    if isinstance(data.get("tags"), list):
        clean_tags = [str(tag).strip() for tag in data["tags"] if str(tag).strip()]
        if clean_tags:
            normalized["tags"] = clean_tags

    person.relation_status = json.dumps(normalized, ensure_ascii=False)
    person.sync_to_database()
    return summary
