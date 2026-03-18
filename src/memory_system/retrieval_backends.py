from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.chat.utils.utils import parse_keywords_string
from src.common.database.database_model import ChatHistory, PersonInfo
from src.common.logger import get_logger
from src.common.vector_store.manager import vector_manager
from src.config.config import global_config
from src.memory_system.memory_utils import parse_datetime_to_timestamp, parse_time_range
from src.person_info.person_info import get_person_id_by_alias, get_person_id_by_person_name

logger = get_logger("memory_retrieval_backends")


def _truncate_text(text: Any, limit: int = 200) -> str:
    if text is None:
        return ""
    text_str = str(text).strip()
    if len(text_str) <= limit:
        return text_str
    return text_str[: max(0, limit - 3)] + "..."


def _success(message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"success": True, "message": message, "data": data or {}}


def _failure(message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"success": False, "message": message, "data": data or {}}


def _build_chat_history_record_payload(record: ChatHistory) -> Dict[str, Any]:
    start_str = datetime.fromtimestamp(record.start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_str = datetime.fromtimestamp(record.end_time).strftime("%Y-%m-%d %H:%M:%S")
    entry: Dict[str, Any] = {
        "id": record.id,
        "theme": record.theme or "",
        "time": {"start": start_str, "end": end_str},
        "start_timestamp": record.start_time,
        "end_timestamp": record.end_time,
        "match_count": record.count or 0,
    }

    summary = record.summary or ""
    if summary:
        entry["summary"] = summary
    elif record.original_text:
        entry["summary"] = _truncate_text(record.original_text, 200)

    if record.keywords:
        try:
            parsed_keywords = json.loads(record.keywords) if isinstance(record.keywords, str) else record.keywords
            if isinstance(parsed_keywords, list):
                entry["keywords"] = [str(k) for k in parsed_keywords]
        except (json.JSONDecodeError, TypeError, ValueError):
            entry["keywords_raw"] = _truncate_text(record.keywords, 120)

    return entry


async def fetch_chat_history(
    chat_id: str,
    *,
    keyword: Optional[str] = None,
    time_range: Optional[str] = None,
    fuzzy: bool = True,
    result_limit: int = 3,
    scan_limit: int = 50,
) -> Dict[str, Any]:
    try:
        if not keyword and not time_range:
            return _failure(
                "未指定查询参数（需要提供keyword或time_range之一）", {"hint": "传入keyword或time_range其中之一"}
            )

        query = ChatHistory.select().where(ChatHistory.chat_id == chat_id)

        if time_range:
            if " - " in time_range:
                start_timestamp, end_timestamp = parse_time_range(time_range)
                time_filter = (ChatHistory.start_time < end_timestamp) & (ChatHistory.end_time > start_timestamp)
            else:
                target_timestamp = parse_datetime_to_timestamp(time_range)
                time_filter = (ChatHistory.start_time <= target_timestamp) & (ChatHistory.end_time >= target_timestamp)
            query = query.where(time_filter)

        records = list(query.order_by(ChatHistory.start_time.desc()).limit(scan_limit))

        if keyword:
            keywords_list = parse_keywords_string(keyword)
            if not keywords_list:
                keywords_list = [keyword.strip()] if keyword.strip() else []
            keywords_lower = [kw.lower() for kw in keywords_list if kw.strip()]
            if not keywords_lower:
                return _failure("关键词为空", {"keyword": keyword})

            filtered_records = []
            for record in records:
                theme = (record.theme or "").lower()
                summary = (record.summary or "").lower()
                original_text = (record.original_text or "").lower()
                record_keywords_list = []
                if record.keywords:
                    try:
                        keywords_data = (
                            json.loads(record.keywords) if isinstance(record.keywords, str) else record.keywords
                        )
                        if isinstance(keywords_data, list):
                            record_keywords_list = [str(k).lower() for k in keywords_data]
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

                matched = False
                if fuzzy:
                    for kw in keywords_lower:
                        if (
                            kw in theme
                            or kw in summary
                            or kw in original_text
                            or any(kw in k for k in record_keywords_list)
                        ):
                            matched = True
                            break
                else:
                    matched = True
                    for kw in keywords_lower:
                        kw_matched = (
                            kw in theme
                            or kw in summary
                            or kw in original_text
                            or any(kw in k for k in record_keywords_list)
                        )
                        if not kw_matched:
                            matched = False
                            break
                if matched:
                    filtered_records.append(record)
            if not filtered_records:
                keywords_str = "、".join(keywords_list)
                match_mode = "包含任意一个关键词" if fuzzy else "包含所有关键词"
                return _failure(
                    f"未找到{match_mode}'{keywords_str}'的聊天记录概述",
                    {
                        "keyword": keywords_list,
                        "time_range": time_range,
                        "match_mode": "fuzzy" if fuzzy else "strict",
                    },
                )
            records = filtered_records

        if not records:
            message = "未找到指定时间范围内的聊天记录概述" if time_range else "未找到相关聊天记录概述"
            return _failure(message, {"time_range": time_range})

        records_to_use = records[:result_limit]
        for record in records_to_use:
            try:
                ChatHistory.update(count=ChatHistory.count + 1).where(ChatHistory.id == record.id).execute()
                record.count = (record.count or 0) + 1
            except Exception as update_error:
                logger.error(f"更新聊天记录概述计数失败: {update_error}")

        record_entries = [_build_chat_history_record_payload(record) for record in records_to_use]
        omitted_count = max(0, len(records) - len(records_to_use))
        message_suffix = f"，另有{omitted_count}条已省略" if omitted_count else ""
        return _success(
            f"找到{len(record_entries)}条聊天记录概述{message_suffix}",
            {
                "records": record_entries,
                "total_found": len(records),
                "omitted": omitted_count,
                "filters": {
                    "keyword": keyword,
                    "time_range": time_range,
                    "match_mode": "fuzzy" if fuzzy else "strict",
                },
            },
        )
    except Exception as exc:
        logger.error(f"查询聊天历史概述失败: {exc}")
        return _failure(f"查询失败: {str(exc)}")


def _format_group_nick_names(group_nick_name_field) -> List[Dict[str, str]]:
    if not group_nick_name_field:
        return []
    try:
        group_nick_names_data = (
            json.loads(group_nick_name_field) if isinstance(group_nick_name_field, str) else group_nick_name_field
        )
        if not isinstance(group_nick_names_data, list) or not group_nick_names_data:
            return []
        group_nick_list: List[Dict[str, str]] = []
        for item in group_nick_names_data:
            if isinstance(item, dict):
                group_nick_list.append(
                    {
                        "group_id": str(item.get("group_id", "未知群号")),
                        "group_nick_name": item.get("group_nick_name", "未知群昵称"),
                    }
                )
            elif isinstance(item, str):
                group_nick_list.append({"group_id": "", "group_nick_name": item})
        return group_nick_list
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(f"解析群昵称信息失败: {exc}")
        if isinstance(group_nick_name_field, str):
            return [{"group_id": "", "group_nick_name": _truncate_text(group_nick_name_field, 200), "raw": True}]
        return []


def _build_person_record_payload(record: PersonInfo, match_type: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "match_type": match_type,
        "person_name": record.person_name or "",
        "nickname": record.nickname or "",
        "person_id": record.person_id or "",
        "platform": record.platform or "",
        "user_id": record.user_id or "",
        "is_known": bool(record.is_known),
        "know_times": int(record.know_times or 0),
    }
    if record.name_reason:
        payload["name_reason"] = record.name_reason
    if record.know_since:
        payload["know_since"] = datetime.fromtimestamp(record.know_since).strftime("%Y-%m-%d %H:%M:%S")
    if record.last_know:
        payload["last_know"] = datetime.fromtimestamp(record.last_know).strftime("%Y-%m-%d %H:%M:%S")

    group_nicks = _format_group_nick_names(getattr(record, "group_nick_name", None))
    if group_nicks:
        payload["group_nick_names"] = group_nicks

    if record.memory_points:
        try:
            memory_points_data = (
                json.loads(record.memory_points) if isinstance(record.memory_points, str) else record.memory_points
            )
            if isinstance(memory_points_data, list):
                sorted_memories = sorted(
                    memory_points_data,
                    key=lambda x: float(x.get("weight", 0) if isinstance(x, dict) else 0),
                    reverse=True,
                )
                memory_entries = []
                for memory_point in sorted_memories[:15]:
                    if isinstance(memory_point, dict):
                        memory_entries.append(
                            {
                                "category": memory_point.get("category", "其他"),
                                "content": memory_point.get("content", ""),
                                "weight": memory_point.get("weight", 0),
                            }
                        )
                    elif isinstance(memory_point, str):
                        parts = memory_point.split(":", 2)
                        if len(parts) >= 3:
                            memory_entries.append(
                                {
                                    "category": parts[0].strip(),
                                    "content": parts[1].strip(),
                                    "weight": parts[2].strip(),
                                }
                            )
                        else:
                            memory_entries.append({"raw": memory_point})
                if memory_entries:
                    payload["top_memories"] = memory_entries
                    payload["memory_count"] = len(memory_points_data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(f"解析用户 {record.person_id} 的memory_points失败: {exc}")
            payload["memory_error"] = "记忆数据解析失败"

    return payload


async def fetch_person_info(person_name: str, *, result_limit: int = 20) -> Dict[str, Any]:
    try:
        person_name = str(person_name).strip()
        if not person_name:
            return _failure("用户名称为空")

        records = list(PersonInfo.select().where(PersonInfo.person_name.contains(person_name)).limit(result_limit))
        if not records:
            return _failure(f"未找到模糊匹配'{person_name}'的用户信息", {"person_name": person_name})

        exact_matches = []
        fuzzy_matches = []
        for record in records:
            if record.person_name and record.person_name.strip() == person_name:
                exact_matches.append(record)
            else:
                fuzzy_matches.append(record)

        payloads = [_build_person_record_payload(record, "exact") for record in exact_matches]
        payloads.extend(_build_person_record_payload(record, "fuzzy") for record in fuzzy_matches)

        data: Dict[str, Any] = {
            "person_name": person_name,
            "total": len(records),
            "exact_matches": len(exact_matches),
            "fuzzy_matches": len(fuzzy_matches),
            "records": payloads,
        }
        if len(records) >= result_limit:
            data["truncated"] = True
        return _success(f"找到 {len(records)} 条匹配的用户信息", data)
    except Exception as exc:
        logger.error(f"查询用户信息失败: {exc}")
        return _failure(f"查询失败: {str(exc)}")


async def fetch_person_memory(
    query: str, *, person_name: Optional[str] = None, result_limit: int = 5
) -> Dict[str, Any]:
    try:
        target_person_id = None
        if person_name:
            target_person_id = get_person_id_by_person_name(person_name) or get_person_id_by_alias(person_name)

        def filter_fn(meta):
            if target_person_id:
                return meta.get("person_id") == target_person_id
            return True

        results = await vector_manager.search_memory(query, k=result_limit, filter_fn=filter_fn)
        if not results:
            return _failure(f"未找到关于'{query}'的相关记忆", {"query": query, "person_name": person_name})

        formatted_results = [
            {
                "content": res.get("text", ""),
                "score": f"{res.get('similarity', 0):.4f}",
                "category": res.get("category", "unknown"),
                "person_id": res.get("person_id", "unknown"),
            }
            for res in results
        ]
        return _success(
            f"找到 {len(results)} 条相关记忆",
            {"query": query, "person_name": person_name, "results": formatted_results},
        )
    except Exception as exc:
        logger.error(f"向量记忆检索失败: {exc}")
        return _failure(f"检索失败: {str(exc)}")


async def fetch_jargon(query: str, *, result_limit: int = 5) -> Dict[str, Any]:
    try:
        results = await vector_manager.search_jargon(query, k=result_limit)
        if not results:
            return _failure(f"未找到关于'{query}'的相关黑话", {"query": query})

        formatted_results = [
            {
                "term": res.get("text", ""),
                "meaning": res.get("meaning", ""),
                "score": f"{res.get('similarity', 0):.4f}",
            }
            for res in results
        ]
        return _success(
            f"找到 {len(results)} 条相关黑话",
            {"query": query, "results": formatted_results},
        )
    except Exception as exc:
        logger.error(f"向量黑话检索失败: {exc}")
        return _failure(f"检索失败: {str(exc)}")


async def fetch_lpmm_knowledge(query: str, *, limit: int = 5) -> Dict[str, Any]:
    try:
        content = str(query).strip()
        if not content:
            return _failure("查询关键词为空")
        try:
            limit_value = int(limit)
        except (TypeError, ValueError):
            limit_value = 5
        limit_value = max(1, limit_value)

        if not global_config.lpmm_knowledge.enable:
            return _failure("LPMM知识库未启用")

        from src.chat.knowledge import get_qa_manager

        qa_manager = get_qa_manager()
        if qa_manager is None:
            return _failure("LPMM知识库未初始化")

        knowledge_info = await qa_manager.get_knowledge(content, limit=limit_value)
        if knowledge_info:
            return _success(
                "你从LPMM知识库中找到以下信息",
                {"query": content, "result": knowledge_info, "limit": limit_value},
            )
        return _failure(
            f"在LPMM知识库中未找到与“{content}”相关的信息",
            {"query": content},
        )
    except Exception as exc:
        logger.error(f"LPMM知识库查询失败: {exc}")
        return _failure(f"LPMM知识库查询失败：{str(exc)}")


__all__ = [
    "fetch_chat_history",
    "fetch_jargon",
    "fetch_lpmm_knowledge",
    "fetch_person_info",
    "fetch_person_memory",
]
