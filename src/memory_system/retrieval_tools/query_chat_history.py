"""根据时间或关键词在chat_history中查询 - 工具实现"""

from typing import Optional

from src.memory_system.retrieval_sources import CHAT_HISTORY_SOURCE, RetrievalSourceRequest

from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_source_result


async def query_chat_history(
    chat_id: str, keyword: Optional[str] = None, time_range: Optional[str] = None, fuzzy: bool = True
) -> str:
    result = await CHAT_HISTORY_SOURCE.collect(
        RetrievalSourceRequest(
            chat_id=chat_id,
            keyword=keyword or "",
            time_range=time_range or "",
            fuzzy=fuzzy,
        )
    )
    return format_source_result(result)


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
