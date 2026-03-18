"""根据person_name查询用户信息 - 工具实现"""

from src.memory_system.retrieval_sources import PERSON_INFO_SOURCE, RetrievalSourceRequest

from .tool_registry import register_memory_retrieval_tool
from .tool_utils import format_source_result


async def query_person_info(person_name: str) -> str:
    result = await PERSON_INFO_SOURCE.collect(
        RetrievalSourceRequest(
            chat_id="",
            person_name=person_name,
        )
    )
    return format_source_result(result)


def register_tool():
    """注册工具"""
    register_memory_retrieval_tool(
        name="query_person_info",
        description="查询某个用户的静态档案信息，如名称、昵称、平台、用户ID、qq号、群昵称以及已登记的人格/标签等。不用于推断具体聊天事件或关系细节，仅用于补全人物背景。",
        parameters=[
            {"name": "person_name", "type": "string", "description": "用户名称，用于查询用户信息", "required": True}
        ],
        execute_func=query_person_info,
    )
