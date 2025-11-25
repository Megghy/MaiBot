"""
向量检索工具 - 工具实现
"""

from typing import Optional
from src.common.logger import get_logger
from src.common.vector_store.manager import vector_manager
from src.person_info.person_info import get_person_id_by_person_name, get_person_id_by_alias
from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_tool_response

logger = get_logger("vector_retrieval_tools")

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
    try:
        target_person_id = None
        if person_name:
            target_person_id = get_person_id_by_person_name(person_name)
            if not target_person_id:
                target_person_id = get_person_id_by_alias(person_name)
        
        def filter_fn(meta):
            if target_person_id:
                return meta.get("person_id") == target_person_id
            return True
            
        results = await vector_manager.search_memory(query, k=5, filter_fn=filter_fn)
        
        if not results:
            return format_tool_response(False, f"未找到关于'{query}'的相关记忆", {"query": query})
            
        formatted_results = []
        for res in results:
            formatted_results.append({
                "content": res.get("text", ""),  # faiss_store stores as 'text'
                "score": f"{res.get('similarity', 0):.4f}",
                "category": res.get("category", "unknown"),
                "person_id": res.get("person_id", "unknown")
            })
            
        return format_tool_response(
            True, 
            f"找到 {len(results)} 条相关记忆", 
            {
                "query": query,
                "results": formatted_results
            }
        )
    except Exception as e:
        logger.error(f"向量记忆检索失败: {e}")
        return format_tool_response(False, f"检索失败: {str(e)}")

async def query_jargon(query: str, chat_id: str = None) -> str:
    """
    使用向量检索查询黑话/知识（支持模糊语义匹配）
    
    Args:
        query: 查询语句
        chat_id: 聊天ID（可选，用于过滤）
        
    Returns:
        查询结果
    """
    try:
        # Jargon usually global or chat specific. 
        # Vector search implies we want meaning.
        
        results = await vector_manager.search_jargon(query, k=5)
        
        if not results:
            return format_tool_response(False, f"未找到关于'{query}'的相关黑话", {"query": query})
            
        formatted_results = []
        for res in results:
            formatted_results.append({
                "term": res.get("text", ""),  # The jargon term itself
                "meaning": res.get("meaning", ""),
                "score": f"{res.get('similarity', 0):.4f}"
            })
            
        return format_tool_response(
            True, 
            f"找到 {len(results)} 条相关黑话", 
            {
                "query": query,
                "results": formatted_results
            }
        )
    except Exception as e:
        logger.error(f"向量黑话检索失败: {e}")
        return format_tool_response(False, f"检索失败: {str(e)}")

def register_tool():
    register_memory_retrieval_tool(
        name="query_person_memory",
        description="使用向量技术检索人物记忆，支持模糊语义匹配。当需要查询某人过去的经历、喜好、或者模糊的记忆时使用。",
        parameters=[
            {"name": "query", "type": "string", "description": "查询语句", "required": True},
            {"name": "person_name", "type": "string", "description": "人物名称（可选）", "required": False}
        ],
        execute_func=query_person_memory
    )
    
    register_memory_retrieval_tool(
        name="query_jargon",
        description="使用向量技术检索黑话或专有名词含义，支持语义匹配。当遇到不懂的词汇或想了解某个概念时使用。",
        parameters=[
            {"name": "query", "type": "string", "description": "查询语句", "required": True}
        ],
        execute_func=query_jargon
    )
