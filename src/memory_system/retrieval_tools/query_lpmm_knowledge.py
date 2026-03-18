"""
通过LPMM知识库查询信息 - 工具实现
"""

from src.memory_system.retrieval_sources import LPMM_KNOWLEDGE_SOURCE, RetrievalSourceRequest

from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_source_result


async def query_lpmm_knowledge(query: str, limit: int = 5) -> str:
    """在LPMM知识库中查询相关信息

    Args:
        query: 查询关键词

    Returns:
        str: 查询结果
    """
    result = await LPMM_KNOWLEDGE_SOURCE.collect(
        RetrievalSourceRequest(
            chat_id="",
            question=query,
            result_limit=limit,
        )
    )
    return format_source_result(result)


def register_tool():
    """注册LPMM知识库查询工具"""
    register_memory_retrieval_tool(
        name="lpmm_search_knowledge",
        description="从LPMM知识库中搜索相关信息，适用于需要知识支持的场景。",
        parameters=[
            {
                "name": "query",
                "type": "string",
                "description": "需要查询的关键词或问题",
                "required": True,
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "希望返回的相关知识条数，默认为5",
                "required": False,
            },
        ],
        execute_func=query_lpmm_knowledge,
    )
