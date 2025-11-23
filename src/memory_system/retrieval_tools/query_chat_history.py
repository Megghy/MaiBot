"""
根据时间或关键词在chat_history中查询 - 工具实现
从ChatHistory表的聊天记录概述库中查询
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from src.common.logger import get_logger
from src.common.database.database_model import ChatHistory
from src.chat.utils.utils import parse_keywords_string
from .tool_registry import register_memory_retrieval_tool
from ..memory_utils import parse_datetime_to_timestamp, parse_time_range
from .tool_utils import format_tool_response, truncate_text

logger = get_logger("memory_retrieval_tools")


def _build_record_payload(record: ChatHistory) -> Dict[str, Any]:
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
        entry["summary"] = truncate_text(record.original_text, 200)

    if record.keywords:
        try:
            parsed_keywords = json.loads(record.keywords) if isinstance(record.keywords, str) else record.keywords
            if isinstance(parsed_keywords, list):
                entry["keywords"] = [str(k) for k in parsed_keywords]
        except (json.JSONDecodeError, TypeError, ValueError):
            entry["keywords_raw"] = truncate_text(record.keywords, 120)

    return entry


async def query_chat_history(
    chat_id: str, keyword: Optional[str] = None, time_range: Optional[str] = None, fuzzy: bool = True
) -> str:
    """根据时间或关键词在chat_history表中查询聊天记录概述

    Args:
        chat_id: 聊天ID
        keyword: 关键词（可选，支持多个关键词，可用空格、逗号等分隔）
        time_range: 时间范围或时间点，格式：
            - 时间范围："YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS"
            - 时间点："YYYY-MM-DD HH:MM:SS"（查询包含该时间点的记录）
        fuzzy: 是否使用模糊匹配模式（默认True）
            - True: 模糊匹配，只要包含任意一个关键词即匹配（OR关系）
            - False: 全匹配，必须包含所有关键词才匹配（AND关系）

    Returns:
        str: 查询结果
    """
    try:
        # 检查参数
        if not keyword and not time_range:
            return format_tool_response(
                False,
                "未指定查询参数（需要提供keyword或time_range之一）",
                {"hint": "传入keyword或time_range其中之一"},
            )

        # 构建查询条件
        query = ChatHistory.select().where(ChatHistory.chat_id == chat_id)

        # 时间过滤条件
        if time_range:
            # 判断是时间点还是时间范围
            if " - " in time_range:
                # 时间范围：查询与时间范围有交集的记录
                start_timestamp, end_timestamp = parse_time_range(time_range)
                # 交集条件：start_time < end_timestamp AND end_time > start_timestamp
                time_filter = (ChatHistory.start_time < end_timestamp) & (ChatHistory.end_time > start_timestamp)
            else:
                # 时间点：查询包含该时间点的记录（start_time <= time_point <= end_time）
                target_timestamp = parse_datetime_to_timestamp(time_range)
                time_filter = (ChatHistory.start_time <= target_timestamp) & (ChatHistory.end_time >= target_timestamp)
            query = query.where(time_filter)

        # 执行查询
        records = list(query.order_by(ChatHistory.start_time.desc()).limit(50))

        # 如果有关键词，进一步过滤
        if keyword:
            # 解析多个关键词（支持空格、逗号等分隔符）
            keywords_list = parse_keywords_string(keyword)
            if not keywords_list:
                keywords_list = [keyword.strip()] if keyword.strip() else []

            # 转换为小写以便匹配
            keywords_lower = [kw.lower() for kw in keywords_list if kw.strip()]

            if not keywords_lower:
                return format_tool_response(False, "关键词为空", {"keyword": keyword})

            filtered_records = []

            for record in records:
                # 在theme、keywords、summary、original_text中搜索
                theme = (record.theme or "").lower()
                summary = (record.summary or "").lower()
                original_text = (record.original_text or "").lower()

                # 解析record中的keywords JSON
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

                # 根据匹配模式检查关键词
                matched = False
                if fuzzy:
                    # 模糊匹配：只要包含任意一个关键词即匹配（OR关系）
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
                    # 全匹配：必须包含所有关键词才匹配（AND关系）
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
                return format_tool_response(
                    False,
                    f"未找到{match_mode}'{keywords_str}'的聊天记录概述",
                    {
                        "keyword": keywords_list,
                        "time_range": time_range,
                        "match_mode": "fuzzy" if fuzzy else "strict",
                    },
                )

            records = filtered_records

        # 如果没有记录（可能是时间范围查询但没有匹配的记录）
        if not records:
            message = "未找到指定时间范围内的聊天记录概述" if time_range else "未找到相关聊天记录概述"
            return format_tool_response(False, message, {"time_range": time_range})

        # 对即将返回的记录增加使用计数
        records_to_use = records[:3]
        for record in records_to_use:
            try:
                ChatHistory.update(count=ChatHistory.count + 1).where(ChatHistory.id == record.id).execute()
                record.count = (record.count or 0) + 1
            except Exception as update_error:
                logger.error(f"更新聊天记录概述计数失败: {update_error}")

        # 构建结果文本
        record_entries: List[Dict[str, Any]] = [_build_record_payload(record) for record in records_to_use]

        omitted_count = max(0, len(records) - len(records_to_use))
        message_suffix = f"，另有{omitted_count}条已省略" if omitted_count else ""
        msg = f"找到{len(record_entries)}条聊天记录概述{message_suffix}"

        return format_tool_response(
            True,
            msg,
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

    except Exception as e:
        logger.error(f"查询聊天历史概述失败: {e}")
        return format_tool_response(False, f"查询失败: {str(e)}")


def register_tool():
    """注册工具"""
    register_memory_retrieval_tool(
        name="query_chat_history",
        description="根据时间或关键词在聊天记录中查询。可以查询某个时间点发生了什么、某个时间范围内的事件，或根据关键词搜索消息概述。支持两种匹配模式：模糊匹配（默认，只要包含任意一个关键词即匹配）和全匹配（必须包含所有关键词才匹配）",
        parameters=[
            {
                "name": "keyword",
                "type": "string",
                "description": "关键词（可选，支持多个关键词，可用空格、逗号、斜杠等分隔，如：'麦麦 百度网盘' 或 '麦麦,百度网盘'。用于在主题、关键词、概括、原文中搜索）",
                "required": False,
            },
            {
                "name": "time_range",
                "type": "string",
                "description": "时间范围或时间点（可选）。格式：'YYYY-MM-DD HH:MM:SS - YYYY-MM-DD HH:MM:SS'（时间范围，查询与时间范围有交集的记录）或 'YYYY-MM-DD HH:MM:SS'（时间点，查询包含该时间点的记录）",
                "required": False,
            },
            {
                "name": "fuzzy",
                "type": "boolean",
                "description": "是否使用模糊匹配模式（默认True）。True表示模糊匹配（只要包含任意一个关键词即匹配，OR关系），False表示全匹配（必须包含所有关键词才匹配，AND关系）",
                "required": False,
            },
        ],
        execute_func=query_chat_history,
    )
