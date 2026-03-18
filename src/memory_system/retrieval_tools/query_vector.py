"""向量检索工具 - 工具实现"""

from typing import Optional

from src.memory_system.retrieval_sources import JARGON_SOURCE, PERSON_MEMORY_SOURCE, RetrievalSourceRequest

from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_source_result


async def query_person_memory(query: str, person_name: Optional[str] = None, chat_id: str = None) -> str:
    """
    使用向量检索查询人物记忆（支持模糊语义匹配）

    Args:
        query: 查询语句/问题
        person_name: 可选，指定人物名称限制搜索范围
        chat_id: 聊天ID（上下文注入）

    Returns:
        查询结果
    """
    result = await PERSON_MEMORY_SOURCE.collect(
        RetrievalSourceRequest(
            chat_id=chat_id or "",
            question=query,
            person_name=person_name or "",
        )
    )
    return format_source_result(result)


async def query_jargon(query: str, chat_id: str = None) -> str:
    """
    使用向量检索查询黑话/知识（支持模糊语义匹配）

    Args:
        query: 查询语句
        chat_id: 聊天ID（可选，用于过滤）

    Returns:
        查询结果
    """
    result = await JARGON_SOURCE.collect(
        RetrievalSourceRequest(
            chat_id=chat_id or "",
            question=query,
        )
    )
    return format_source_result(result)


def register_tool():
    register_memory_retrieval_tool(
        name="query_person_memory",
        description="使用向量技术检索人物记忆，支持模糊语义匹配。当需要查询某人过去的经历、喜好、或者模糊的记忆时使用。",
        parameters=[
            {"name": "query", "type": "string", "description": "查询语句", "required": True},
            {"name": "person_name", "type": "string", "description": "人物名称（可选）", "required": False},
        ],
        execute_func=query_person_memory,
    )

    register_memory_retrieval_tool(
        name="query_jargon",
        description="使用向量技术检索黑话或专有名词含义，支持语义匹配。当遇到不懂的词汇或想了解某个概念时使用。",
        parameters=[{"name": "query", "type": "string", "description": "查询语句", "required": True}],
        execute_func=query_jargon,
    )
