"""
根据关键词在jargon库中查询 - 工具实现
"""

from src.common.logger import get_logger
from src.jargon.jargon_miner import search_jargon
from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_tool_response

logger = get_logger("memory_retrieval_tools")


async def query_jargon(keyword: str, chat_id: str) -> str:
    """根据关键词在jargon库中查询

    Args:
        keyword: 关键词（黑话/俚语/缩写）
        chat_id: 聊天ID

    Returns:
        str: 查询结果
    """
    try:
        content = str(keyword).strip()
        if not content:
            return format_tool_response(False, "关键词为空")

        # 先尝试精确匹配
        results = search_jargon(keyword=content, chat_id=chat_id, limit=10, case_sensitive=False, fuzzy=False)

        is_fuzzy_match = False

        # 如果精确匹配未找到，尝试模糊搜索
        if not results:
            results = search_jargon(keyword=content, chat_id=chat_id, limit=10, case_sensitive=False, fuzzy=True)
            is_fuzzy_match = True

        if results:
            normalized = []
            for result in results:
                normalized.append(
                    {
                        "content": result.get("content", "").strip(),
                        "meaning": result.get("meaning", "").strip(),
                    }
                )

            return format_tool_response(
                True,
                "在jargon库中找到匹配",
                {
                    "keyword": content,
                    "results": normalized,
                    "match_mode": "fuzzy" if is_fuzzy_match else "exact",
                },
            )

        # 未命中
        logger.info(f"在jargon库中未找到匹配（当前会话或全局，精确匹配和模糊搜索都未找到）: {content}")
        return format_tool_response(False, f"未在jargon库中找到'{content}'的解释", {"keyword": content})

    except Exception as e:
        logger.error(f"查询jargon失败: {e}")
        return format_tool_response(False, f"查询失败: {str(e)}")


def register_tool():
    """注册工具"""
    register_memory_retrieval_tool(
        name="query_jargon",
        description="根据关键词在jargon库中查询黑话/俚语/缩写的含义。支持大小写不敏感搜索，默认会先尝试精确匹配，如果找不到则自动使用模糊搜索。仅搜索当前会话或全局jargon。",
        parameters=[{"name": "keyword", "type": "string", "description": "关键词（黑话/俚语/缩写）", "required": True}],
        execute_func=query_jargon,
    )
