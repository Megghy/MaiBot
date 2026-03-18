import time
import random
import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.memory_system.memory_ingester import schedule_jargon_ingestion, schedule_question_answer_ingestion
from src.memory_system.retrieval_agent import _react_agent_solve_question
from src.memory_system.retrieval_cache import (
    _cleanup_stale_not_found_thinking_back,
    _query_thinking_back,
    _store_thinking_back,
)
from src.memory_system.retrieval_concepts import (
    _build_focus_summary,
    _extract_keywords,
)
from src.memory_system.retrieval_format import (
    _extract_answer_source,
    _format_answer_payload,
)
from src.memory_system.retrieval_merge import (
    build_cached_memory_block,
    build_question_memory_block,
    merge_ranked_memory_blocks,
)
from src.memory_system.retrieval_questioning import (
    _build_participant_hints,
    _parse_questions_json,
    _record_question_generation_trace,
)
from src.memory_system.retrieval_sources import (
    CACHED_MEMORIES_SOURCE,
    CONCEPT_EXPANSION_SOURCE,
    LPMM_KNOWLEDGE_SOURCE,
    RECENT_QUERY_HISTORY_SOURCE,
    RetrievalSourceRequest,
    collect_retrieval_source_results,
)
from src.memory_system.retrieval_tools import init_all_tools
from src.plugin_system.apis import llm_api

logger = get_logger("memory_retrieval")

MAX_QUESTIONS_PER_TURN = 2


@dataclass(slots=True)
class QuestionRetrievalResult:
    question: str
    answer: str
    thinking_steps: Optional[list[dict[str, Any]]] = None
    from_cache: bool = False
    should_ingest: bool = False


def init_memory_retrieval_prompt():
    """初始化记忆检索相关的 prompt 模板和工具"""
    # 首先注册所有工具
    init_all_tools()

    # 第一步：问题生成prompt
    Prompt(
        """
你的名字是 {bot_name}, 现在是 {time_now}.
你会拿到四段上下文:
1. 聊天主干 (最近几轮消息)
{chat_history}
2. 最近已查询/回答的问题摘要 (可追问, 可补充细节)
{recent_query_history}
3. 角色线索 (帮你处理指代和昵称)
{participant_hints}
4. 对话焦点概览 (自动提取的摘要与关键词)
{focus_summary}

当前 {sender} 说: {target_message}

任务: 判断是否需要回忆历史信息来回复, 并给出需要检索的关键词与问题。

操作步骤:
1. 先根据上下文拆分出角色/事件/时间/地点/术语等可能缺失的信息.
2. 如果这些信息会明显提升回答质量, 记录相应的概念关键词.
3. 只有在确实缺信息、且问题中存在明确的疑问信号/需要补充的点时, 才汇总 1~3 个检索问题 (按重要度降序). 若无需检索, 直接返回空数组。

编写问题时请注意:
- 聊天里若出现 "上次", "那个ta", "之前说" 等模糊指代, 需要展开成可检索的问题.
- 如果 recent_query_history 已有部分答案, 但你还需更多细节, 可以在问题里说明想补充的方向.
- 先确保真的无法用现有信息回复, 再提出检索; 没有疑问词、上下文足够时, 不要生成检索问题.
- 如果多个疑问都围绕同一对人物或同一件事的不同细节, 请尽量合并为 1~2 个综合问题, 避免拆分过细.

输出 JSON, 且只输出 JSON:
{{
  "concepts": ["概念A", "概念B"],
  "questions": ["问题1", "问题2", "问题3"]
}}

当无需检索时, concepts 与 questions 都输出 [] 即可.
""",
        name="memory_retrieval_question_prompt",
    )

    # 第二步：ReAct Agent prompt（使用function calling，要求先思考再行动）
    Prompt(
        """你的名字是 {bot_name}. 现在是 {time_now}.
你正在参与聊天, 你需要搜集信息来回答问题, 帮助你参与聊天.

**重要限制:**
- 最大查询轮数: {max_iterations} 轮(当前第 {current_iteration} 轮, 剩余 {remaining_iterations} 轮)
- 必须尽快得出答案, 避免不必要的查询
- 思考要简短, 直接切入要点
- 回答时遵循优先级:
  1. 优先使用检索到的记忆/知识库信息回答问题
  2. 如果确定问题与聊天上下文无关, 并且你确定自己知道高置信度答案, 可以在明确标注 source="model" 的前提下, 使用你作为大语言模型已有的常识或通用知识进行回答
  3. 如果多轮检索后仍然没有找到相关信息, 可以在明确标注 source="model" 的前提下, 使用你作为大语言模型已有的常识或通用知识进行回答
  4. 不允许胡乱编造明显错误的信息

当前问题: {question}
已收集的信息:
{collected_info}

**执行步骤:**
**第一步: 思考(Think)**
在思考中分析:
- 当前信息是否足够回答问题?
- 如果信息足够或能根据常识回答，决定调用 `found_answer` 工具。
- 如果信息不足且无法回答，决定调用 `not_enough_info` 工具。
- 如果需要更多信息，决定调用检索类工具（如 `search_chat_history` 等）。

**第二步: 行动(Action)**
- 如果涉及过往事件、态度、冲突或关系变化, 优先使用聊天记录查询工具查询相关片段或事件
- 如果涉及概念, 可以用 jargon 查询, 或根据关键词检索聊天记录
- 如果涉及人物的静态档案信息(如昵称、平台、用户ID、人格标签等), 可以使用人物信息查询工具查询人物信息; 不要依赖人物信息工具推断具体事件经过
{lpmm_hint}
- 如果信息不足且需要继续查询, 说明最需要查询什么, 并输出为纯文本说明, 然后调用相应工具查询(可并行调用多个工具)

**示例:**
Think: 用户问 "昨天群里谁提到了GitHub". 我需要查询昨天的聊天记录.
Action: 调用 `search_chat_history(keyword="GitHub", time_range="1d")`
Observation: 工具返回: [张三] 昨天 10:00 说: GitHub 上那个项目不错.
Think: 我找到了答案, 是张三. 我应该提交答案.
Action: 调用 `found_answer(answer="昨天张三提到了GitHub", source="memory")`

**重要规则:**
- **必须** 通过调用工具来执行操作，无论是查询还是提交答案。
- 当检索到明确、有关的信息并得出答案时, 调用 `found_answer(answer="...", source="memory")`
- 当检索不到相关信息, 但你根据常识/通用知识可以给出可靠、通用层面的回答时, 调用 `found_answer(answer="...", source="model")`
- 如果信息不足、问题过于具体、你无法给出可靠答案, 必须调用 `not_enough_info`, 不要强行回答
- 一旦调用了 `found_answer` 或 `not_enough_info`, 本轮对话即视为结束
""",
        name="memory_retrieval_react_prompt_head",
    )

    # 额外，如果最后一轮迭代：ReAct Agent prompt（使用function calling，要求先思考再行动）
    Prompt(
        """你的名字是 {bot_name}. 现在是 {time_now}.
你正在参与聊天, 你需要根据搜集到的信息判断问题是否可以回答.

当前问题: {question}
已收集的信息:
{collected_info}

**执行步骤:**
分析:
- 当前信息是否足够回答问题?
- **如果信息足够且能找到明确答案**, 调用 `found_answer` 工具提交答案:
  - 如果主要依据检索信息: source="memory"
  - 如果主要依据你的常识/通用知识: source="model"
- **如果信息不足或无法找到答案**, 调用 `not_enough_info` 工具提交原因

**重要规则:**
- 你已经经过几轮查询, 尝试了信息搜集, 现在你需要总结信息, 选择回答问题或判断问题无法回答
- 本轮中不应再调用任何查询类工具, 只能在 `found_answer` 和 `not_enough_info` 之间做出选择
- 优先使用检索到的信息回答问题; 在检索完全失败的情况下, 可以在明确标注 source="model" 的前提下, 使用你的常识/通用知识给出答案
- 不允许为了逃避 not_enough_info 而胡乱编造明显错误的信息
- 答案必须精简, 不要过多解释
- 如果你已经尝试检索仍然无法确定答案, 使用 not_enough_info
- 必须调用 `found_answer` 或 `not_enough_info` 之一来结束任务
""",
        name="memory_retrieval_react_final_prompt",
    )


async def _process_single_question(
    question: str,
    chat_id: str,
    context: str,
    initial_info: str = "",
) -> Optional[QuestionRetrievalResult]:
    """处理单个问题的查询（包含缓存检查逻辑）

    Args:
        question: 要查询的问题
        chat_id: 聊天ID
        context: 上下文信息
        initial_info: 初始信息（如概念检索结果），将传递给ReAct Agent

    Returns:
        Optional[QuestionRetrievalResult]: 如果找到答案，返回结构化结果，否则返回None
    """
    logger.info(f"开始处理问题: {question}")

    await asyncio.to_thread(_cleanup_stale_not_found_thinking_back)

    question_initial_info = initial_info or ""

    lpmm_results = await collect_retrieval_source_results(
        RetrievalSourceRequest(
            chat_id=chat_id,
            question=question,
            context=context,
            result_limit=2,
        ),
        [LPMM_KNOWLEDGE_SOURCE],
    )
    lpmm_result = lpmm_results[LPMM_KNOWLEDGE_SOURCE.name].content
    if lpmm_result:
        if question_initial_info:
            question_initial_info += "\n"
        question_initial_info += f"【LPMM知识库预查询】\n{lpmm_result}"
        logger.info(f"LPMM预查询命中，问题: {question[:50]}...")
    else:
        logger.info(f"LPMM预查询未命中或未找到信息，问题: {question[:50]}...")

    # 先检查thinking_back数据库中是否有现成答案
    cached_result = await asyncio.to_thread(_query_thinking_back, chat_id, question)
    should_requery = False

    if cached_result:
        cached_found_answer, cached_answer, cached_steps = cached_result

        if cached_found_answer:
            # 根据缓存信息的完整度来判断是否直接返回
            reuse_probability = 0.8
            if cached_steps:
                last_source = _extract_answer_source(cached_steps)
                if last_source == "model":
                    reuse_probability = 0.4  # 模型常识答案可靠度略低

            if cached_answer and random.random() <= reuse_probability:
                logger.info(f"直接复用thinking_back缓存答案，问题: {question[:50]}... 概率: {reuse_probability}")
                return QuestionRetrievalResult(
                    question=question,
                    answer=cached_answer,
                    thinking_steps=cached_steps,
                    from_cache=True,
                )

            should_requery = True
            logger.info(f"缓存命中但触发重新检索，问题: {question[:50]}... reuse_probability={reuse_probability:.2f}")
            if not cached_answer:
                logger.info(f"缓存答案为空，将强制重新查询，问题: {question[:50]}...")
        else:
            # found_answer == 0：不使用缓存，直接重新查询
            should_requery = True
            logger.info(f"thinking_back存在但未找到答案，忽略缓存重新查询，问题: {question[:50]}...")

    # 如果没有缓存答案或需要重新查询，使用ReAct Agent查询
    if not cached_result or should_requery:
        if should_requery:
            logger.info(f"概率触发重新查询，使用ReAct Agent查询，问题: {question[:50]}...")
        else:
            logger.info(f"未找到缓存答案，使用ReAct Agent查询，问题: {question[:50]}...")

        found_answer, answer, thinking_steps, is_timeout = await _react_agent_solve_question(
            question=question,
            chat_id=chat_id,
            max_iterations=global_config.memory.max_agent_iterations,
            timeout=120.0,
            initial_info=question_initial_info,
        )

        # 存储到数据库（超时时不存储）
        if not is_timeout:
            await asyncio.to_thread(
                _store_thinking_back,
                chat_id=chat_id,
                question=question,
                context=context,
                found_answer=found_answer,
                answer=answer,
                thinking_steps=thinking_steps,
            )
        else:
            logger.info(f"ReAct Agent超时，不存储到数据库，问题: {question[:50]}...")

        if found_answer and answer:
            return QuestionRetrievalResult(
                question=question,
                answer=answer,
                thinking_steps=thinking_steps,
                should_ingest=True,
            )

    return None


async def build_memory_retrieval_prompt(
    message: str,
    sender: str,
    target: str,
    chat_stream,
    tool_executor,
) -> str:
    """构建记忆检索提示
    使用两段式查询：第一步生成问题，第二步使用ReAct Agent查询答案

    Args:
        message: 聊天历史记录
        sender: 发送者名称
        target: 目标消息内容
        chat_stream: 聊天流对象
        tool_executor: 工具执行器（保留参数以兼容接口）

    Returns:
        str: 记忆检索结果字符串
    """
    start_time = time.time()

    logger.info(f"检测是否需要回忆，元消息：{message[:30]}...，消息长度: {len(message)}")
    try:
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        bot_name = global_config.bot.nickname
        chat_id = chat_stream.stream_id

        # 获取最近查询历史（最近1小时内的查询）
        history_results = await collect_retrieval_source_results(
            RetrievalSourceRequest(
                chat_id=chat_id,
                context=message,
                focus_summary="",
                time_window_seconds=300.0,
            ),
            [RECENT_QUERY_HISTORY_SOURCE],
        )
        recent_query_history = history_results[RECENT_QUERY_HISTORY_SOURCE.name].content or "最近没有查询记录。"

        participant_hints = await asyncio.to_thread(_build_participant_hints, chat_id)

        keywords = _extract_keywords(target)
        focus_summary = _build_focus_summary(message, target, keywords)

        # 第一步：生成问题
        question_prompt = await global_prompt_manager.format_prompt(
            "memory_retrieval_question_prompt",
            bot_name=bot_name,
            time_now=time_now,
            chat_history=message,
            recent_query_history=recent_query_history,
            participant_hints=participant_hints,
            focus_summary=focus_summary,
            sender=sender,
            target_message=target,
        )

        success, response, reasoning_content, model_name = await llm_api.generate_with_model(
            question_prompt,
            model_config=model_config.model_task_config.tool_use,
            request_type="memory.question",
        )

        if global_config.debug.show_memory_prompt:
            logger.info(f"记忆检索问题生成提示词: {question_prompt}")
        logger.info(f"记忆检索问题生成响应: {response}")

        if not success:
            logger.error(f"LLM生成问题失败: {response}")
            return ""

        # 解析概念列表和问题列表
        concepts, questions = _parse_questions_json(response)
        logger.info(f"解析到 {len(concepts)} 个概念: {concepts}")
        logger.info(f"解析到 {len(questions)} 个问题(原始): {questions}")

        # 对概念进行jargon检索，作为初始信息
        source_results = await collect_retrieval_source_results(
            RetrievalSourceRequest(
                chat_id=chat_id,
                context=message,
                focus_summary=focus_summary,
                concepts=tuple(concepts),
                time_window_seconds=300.0,
            ),
            [CONCEPT_EXPANSION_SOURCE, CACHED_MEMORIES_SOURCE],
        )
        concept_result = source_results[CONCEPT_EXPANSION_SOURCE.name]
        concept_info = concept_result.content
        learned_jargon = concept_result.payload or {}
        cached_memories = source_results[CACHED_MEMORIES_SOURCE.name].payload or []

        initial_info = ""
        if concept_info:
            initial_info += concept_info
            logger.info(f"概念检索完成，结果: {concept_info[:200]}...")
        elif concepts:
            logger.info("概念检索未找到任何结果")
        if learned_jargon:
            schedule_jargon_ingestion(learned_jargon, chat_id)

        if not questions:
            logger.info("问题列表为空，跳过记忆检索")

        # 控制单轮问题数量上限，避免问题拆分过多
        if questions and len(questions) > MAX_QUESTIONS_PER_TURN:
            logger.info(
                f"问题数量 {len(questions)} 超过上限 {MAX_QUESTIONS_PER_TURN}，仅保留前 {MAX_QUESTIONS_PER_TURN} 个问题"
            )
            questions = questions[:MAX_QUESTIONS_PER_TURN]
            logger.info(f"截断后的问题列表: {questions}")

        await _record_question_generation_trace(
            chat_stream,
            question_prompt,
            response,
            concepts,
            questions,
        )

        # 第二步：并行处理所有问题（使用配置的最大迭代次数/120秒超时）
        max_iterations = global_config.memory.max_agent_iterations
        logger.info(f"问题数量: {len(questions)}，设置最大迭代次数: {max_iterations}，超时时间: 120秒")

        # 并行处理所有问题，将概念检索结果作为初始信息传递
        question_tasks = [
            _process_single_question(question=question, chat_id=chat_id, context=message, initial_info=initial_info)
            for question in questions
        ]

        # 并行执行所有查询任务
        results = await asyncio.gather(*question_tasks, return_exceptions=True)

        # 显式构建 question/cached 两类命中块，再交给 merge 层去重排序
        question_blocks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"处理问题 '{questions[i]}' 时发生异常: {result}")
            elif result is not None:
                if result.should_ingest:
                    schedule_question_answer_ingestion(result.question, result.answer, chat_id)
                formatted_block = _format_answer_payload(
                    result.question,
                    result.answer,
                    result.thinking_steps,
                    from_cache=result.from_cache,
                )
                question_blocks.append(
                    build_question_memory_block(
                        question=result.question,
                        block=formatted_block,
                        raw_source=_extract_answer_source(result.thinking_steps or []),
                        from_cache=result.from_cache,
                        order=i,
                    )
                )
                logger.debug(f"提取到问题：{result.question[:50]}...")

        cached_blocks = []
        for i, cached_memory in enumerate(cached_memories):
            cached_block = build_cached_memory_block(
                question=cached_memory.question,
                block=cached_memory.block,
                order=i,
            )
            if cached_block is None:
                continue
            cached_blocks.append(cached_block)
            logger.debug(f"添加缓存记忆候选: {cached_block.question[:50]}...")

        all_results = merge_ranked_memory_blocks(
            initial_info=initial_info,
            question_blocks=question_blocks,
            cached_blocks=cached_blocks,
        )

        end_time = time.time()

        if all_results:
            retrieved_memory = "\n\n".join(all_results)
            logger.info(
                f"记忆检索成功，耗时: {(end_time - start_time):.3f}秒，包含 {len(all_results)} 条记忆（含缓存）"
            )
            return f"你回忆起了以下信息：\n{retrieved_memory}\n如果与回复内容相关，可以参考这些回忆的信息。\n"
        else:
            logger.debug("所有问题均未找到答案，且无缓存记忆")
            return ""

    except Exception as e:
        logger.error(f"记忆检索时发生异常: {str(e)}")
        return ""
