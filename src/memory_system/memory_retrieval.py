import time
import json
import re
import random
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.plugin_system.apis import llm_api, database_api, message_api
from src.common.database.database_model import ThinkingBack
from json_repair import repair_json
from src.memory_system.retrieval_tools import get_tool_registry, init_all_tools
from src.memory_system.retrieval_tools.query_lpmm_knowledge import query_lpmm_knowledge
from src.chat.knowledge import get_qa_manager
from src.person_info.person_info import Person


def _extract_keywords(text: str, *, max_keywords: int = 6) -> List[str]:
    if not text:
        return []

    tokens = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text)
    keywords: List[str] = []
    for token in tokens:
        if token in keywords:
            continue
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _build_focus_summary(chat_history: str, target_message: str, keywords: List[str]) -> str:
    summary_lines: List[str] = []
    if chat_history:
        recent_lines = chat_history.strip().splitlines()[-5:]
        summary_lines.append("最近对话片段：")
        summary_lines.extend(recent_lines)
    if target_message:
        summary_lines.append(f"当前消息：{_truncate_text(target_message, 80)}")
    if keywords:
        summary_lines.append("关键词提示：" + "、".join(keywords))
    return "\n".join(summary_lines) if summary_lines else "无可用摘要"


def _build_fallback_question(sender: str, target_message: str) -> Optional[str]:
    core = target_message.strip() if target_message else "这条消息"
    core = _truncate_text(core, 24) or "这条消息"
    sender = sender or "对方"
    return f"{sender} 提到的“{core}”指的具体事件或背景是什么？"


QUESTION_HINT_REGEX = re.compile(r"[？?]|谁|什么|啥|哪|哪里|哪儿|什么时候|何时|为何|为什么|咋|怎么|如何|多少|第几")


def _should_use_fallback_question(target_message: str, *, has_recent_query_history: bool, min_length: int = 6) -> bool:
    """基于启发式判断是否需要触发fallback问题"""

    if has_recent_query_history:
        return False

    text = (target_message or "").strip()
    if len(text) < min_length:
        return False

    return bool(QUESTION_HINT_REGEX.search(text))


async def _record_question_generation_trace(
    chat_stream,
    question_prompt: str,
    response: str,
    concepts: List[str],
    questions: List[str],
) -> None:
    try:
        action_data = {
            "prompt": question_prompt,
            "response": response,
            "concepts": concepts,
            "questions": questions,
        }
        await database_api.store_action_info(
            chat_stream=chat_stream,
            action_name="memory_question_generation",
            action_data=action_data,
            action_reasoning="question_generation",
        )
    except Exception as exc:
        logger.warning(f"记录问题生成追踪失败: {exc}")


def _build_participant_hints(chat_id: str, *, limit: int = 10) -> str:
    """聚合最近聊天参与者与人物记忆摘要，辅助question prompt理解指代"""

    try:
        end_time = time.time()
        start_time = max(0.0, end_time - 3600)
        recent_messages = message_api.get_messages_by_time_in_chat(
            chat_id=chat_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            filter_mai=False,
            filter_command=True,
        )
    except Exception as exc:
        logger.warning(f"获取参与者信息失败: {exc}")
        recent_messages = []

    participants: Dict[str, Dict[str, Any]] = {}
    for msg in recent_messages:
        user_info = msg.user_info
        if not user_info:
            continue
        key = f"{user_info.platform}:{user_info.user_id}"
        if key in participants:
            continue
        person = Person(platform=user_info.platform, user_id=user_info.user_id)
        if not person.person_name:
            continue
        memory_preview = ""
        if person.memory_points:
            sample = person.memory_points[0]
            memory_preview = sample if isinstance(sample, str) else ""

        participants[key] = {
            "name": person.person_name,
            "known": person.is_known,
            "memory": memory_preview,
        }

    if not participants:
        return "最近未识别出其他角色。"

    lines = [
        "最近角色线索：",
    ]
    for info in participants.values():
        note = "已登记" if info.get("known") else "未知"
        memory = info.get("memory")
        memory_str = f"；记忆片段：{memory}" if memory else ""
        lines.append(f"- {info['name']}（{note}）{memory_str}")

    return "\n".join(lines)


def _safe_str(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()
from src.llm_models.payload_content.message import MessageBuilder, RoleType, Message
from src.llm_models.payload_content.tool_option import ToolParamType

logger = get_logger("memory_retrieval")

THINKING_BACK_NOT_FOUND_RETENTION_SECONDS = 36000  # 未找到答案记录保留时长
THINKING_BACK_CLEANUP_INTERVAL_SECONDS = 3000  # 清理频率
_last_not_found_cleanup_ts: float = 0.0
MAX_OBSERVATION_LENGTH = 1000  # 工具返回结果最大长度
MAX_COLLECTED_INFO_LENGTH = 2000  # ReAct上下文最大长度
COLLECTED_INFO_TRUNCATE_NOTICE = "…（已截断，仅保留最近信息）\n"
MIN_QA_ANALYSIS_CONFIDENCE = 0.75  # 低于该置信度的结果将被丢弃

# 定义特殊工具用于提交答案
SPECIAL_TOOL_DEFINITIONS = [
    {
        "name": "found_answer",
        "description": "当找到足够的证据或根据常识可以回答问题时，调用此工具提交最终答案。调用此工具将结束思考过程。",
        "parameters": [
            (
                "answer",
                ToolParamType.STRING,
                "最终的答案内容。如果是基于检索到的信息，请综合整理；如果是基于常识，请直接给出。",
                True,
                None,
            ),
            (
                "source",
                ToolParamType.STRING,
                "答案的来源类型",
                True,
                ["memory", "model"],
            ),
        ],
    },
    {
        "name": "not_enough_info",
        "description": "当尝试了多种检索方式仍无法找到足够信息，且无法根据常识回答时，调用此工具结束思考过程。",
        "parameters": [
            (
                "reason",
                ToolParamType.STRING,
                "无法回答的具体原因",
                True,
                None,
            ),
        ],
    },
]


def _truncate_text(text: str, limit: int = 160) -> str:
    """对文本进行截断，避免日志或prompt过长"""

    if not text:
        return ""

    text = text.strip()
    if len(text) <= limit:
        return text
    ellipsis = "…"
    keep = max(0, limit - len(ellipsis))
    return text[:keep] + ellipsis


def _format_action_repr(action: Dict[str, Any]) -> str:
    action_type = action.get("action_type", "")
    params = action.get("action_params") or {}
    if not params:
        return action_type

    display_params = []
    for key, value in params.items():
        if key == "chat_id":
            continue
        value_str = _truncate_text(str(value), 40)
        display_params.append(f"{key}={value_str}")

    param_str = ", ".join(display_params)
    return f"{action_type}({param_str})" if display_params else action_type


def _format_agent_summary(thinking_steps: List[Dict[str, Any]], max_steps: int = 3) -> str:
    if not thinking_steps:
        return ""

    summary_lines: List[str] = []
    for step in thinking_steps[:max_steps]:
        iteration = step.get("iteration")
        line_parts = [f"第{iteration}轮"] if iteration is not None else []

        reasoning = step.get("reasoning") or step.get("thought") or ""
        if reasoning:
            line_parts.append(f"思考：{_truncate_text(reasoning, 80)}")

        actions = step.get("actions") or []
        if actions:
            action_text = "；".join(_format_action_repr(action) for action in actions)
            line_parts.append(f"动作：{action_text}")

        observations = step.get("observations") or []
        if observations:
            line_parts.append(f"观察：{_truncate_text(observations[0], 80)}")

        summary_lines.append("；".join(line_parts))

    if len(thinking_steps) > max_steps:
        summary_lines.append("……")

    return "\n".join(summary_lines)


def _extract_answer_source(thinking_steps: List[Dict[str, Any]]) -> str:
    for step in reversed(thinking_steps):
        for action in step.get("actions", []):
            if action.get("action_type") == "found_answer":
                params = action.get("action_params") or {}
                return params.get("source", "")
    return ""


def _describe_source(raw_source: Optional[str], *, from_cache: bool = False) -> str:
    if from_cache:
        return "历史缓存"

    if not raw_source:
        return "检索记忆"

    mapping = {"memory": "检索记忆", "model": "模型常识"}
    return mapping.get(raw_source, raw_source)


def _build_answer_block(
    question: str,
    answer: str,
    *,
    source_label: str,
    agent_summary: Optional[str] = None,
    extra_note: Optional[str] = None,
) -> str:
    lines = [
        f"问题：{question}",
        f"答案：{answer}",
    ]

    if source_label:
        lines.append(f"来源：{source_label}")

    if extra_note:
        lines.append(f"备注：{extra_note}")

    if agent_summary:
        lines.append("Agent步骤：")
        lines.append(agent_summary)

    return "\n".join(lines)


def _load_thinking_steps(raw_steps: Optional[str]) -> List[Dict[str, Any]]:
    if not raw_steps:
        return []

    try:
        loaded = json.loads(raw_steps)
        if isinstance(loaded, list):
            return loaded
    except Exception as exc:  # pragma: no cover - 防御性日志
        logger.warning(f"解析thinking_steps失败: {exc}")
    return []


def _format_answer_payload(
    question: str,
    answer: str,
    thinking_steps: Optional[List[Dict[str, Any]]] = None,
    *,
    from_cache: bool = False,
    extra_note: Optional[str] = None,
) -> str:
    steps = thinking_steps or []
    raw_source = _extract_answer_source(steps)
    source_label = _describe_source(raw_source, from_cache=from_cache)
    agent_summary = _format_agent_summary(steps)
    return _build_answer_block(
        question,
        answer,
        source_label=source_label,
        agent_summary=agent_summary or None,
        extra_note=extra_note,
    )


def _append_collected_info(current: str, addition: str) -> str:
    addition = (addition or "").strip()
    if not addition:
        return current

    combined = f"{current}\n{addition}" if current else addition
    combined = combined.strip()
    if len(combined) <= MAX_COLLECTED_INFO_LENGTH:
        return combined

    truncated = combined[-MAX_COLLECTED_INFO_LENGTH :]
    return f"{COLLECTED_INFO_TRUNCATE_NOTICE}{truncated}"


def _prepare_collected_info_for_prompt(collected_info: str, limit: int = 1800) -> str:
    if not collected_info:
        return "暂无信息"

    if len(collected_info) <= limit:
        return collected_info

    trimmed = collected_info[-limit:]
    return f"{COLLECTED_INFO_TRUNCATE_NOTICE}{trimmed}"


def _calculate_iteration_budget(question: str, initial_info: str) -> int:
    base = max(1, getattr(global_config.memory, "max_agent_iterations", 3))
    if base <= 2:
        return base

    normalized_question = (question or "").strip()
    complexity_score = 0

    if len(normalized_question) > 48:
        complexity_score += 1
    if len(re.findall(r"[?？]", normalized_question)) > 1:
        complexity_score += 1
    detail_keywords = ("为什么", "如何", "怎么", "原因", "步骤", "哪些", "哪几")
    if any(keyword in normalized_question for keyword in detail_keywords):
        complexity_score += 1
    if len(normalized_question) <= 16:
        complexity_score -= 1
    if initial_info:
        complexity_score -= 1

    if complexity_score <= -1:
        return 2
    if complexity_score == 0:
        return max(2, base - 1)
    return base


def _extract_question_from_block(block: str) -> Optional[str]:
    for line in block.splitlines():
        if line.startswith("问题："):
            return line.replace("问题：", "", 1).strip()
    return None


def _cleanup_stale_not_found_thinking_back() -> None:
    """定期清理过期的未找到答案记录"""
    global _last_not_found_cleanup_ts

    now = time.time()
    if now - _last_not_found_cleanup_ts < THINKING_BACK_CLEANUP_INTERVAL_SECONDS:
        return

    threshold_time = now - THINKING_BACK_NOT_FOUND_RETENTION_SECONDS
    try:
        deleted_rows = (
            ThinkingBack.delete()
            .where((ThinkingBack.found_answer == 0) & (ThinkingBack.update_time < threshold_time))
            .execute()
        )
        if deleted_rows:
            logger.info(f"清理过期的未找到答案thinking_back记录 {deleted_rows} 条")
        _last_not_found_cleanup_ts = now
    except Exception as e:
        logger.error(f"清理未找到答案的thinking_back记录失败: {e}")


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
- 先确保真的无法用现有信息回复, 再提出检索; 没有疑问词、上下文足够时, 不要生成检索问题。

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
- 如果涉及过往事件, 可以使用聊天记录查询工具查询过往事件
- 如果涉及概念, 可以用 jargon 查询, 或根据关键词检索聊天记录
- 如果涉及人物, 可以使用人物信息查询工具查询人物信息
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
- 优先使用检索到的信息回答问题; 在检索完全失败的情况下, 可以在明确标注 source="model" 的前提下, 使用你的常识/通用知识给出答案
- 不允许为了逃避 not_enough_info 而胡乱编造明显错误的信息
- 答案必须精简, 不要过多解释
- 如果你已经尝试检索仍然无法确定答案, 使用 not_enough_info
- 必须调用 `found_answer` 或 `not_enough_info` 之一来结束任务
""",
        name="memory_retrieval_react_final_prompt",
    )


async def _retrieve_concepts_with_jargon(concepts: List[str], chat_id: str) -> str:
    """对概念列表进行jargon检索

    Args:
        concepts: 概念列表
        chat_id: 聊天ID

    Returns:
        str: 检索结果字符串
    """
    if not concepts:
        return ""

    from src.jargon.jargon_miner import search_jargon

    results = []
    for concept in concepts:
        concept = concept.strip()
        if not concept:
            continue

        # 先尝试精确匹配
        jargon_results = await asyncio.to_thread(
            search_jargon, keyword=concept, chat_id=chat_id, limit=10, case_sensitive=False, fuzzy=False
        )

        is_fuzzy_match = False

        # 如果精确匹配未找到，尝试模糊搜索
        if not jargon_results:
            jargon_results = await asyncio.to_thread(
                search_jargon, keyword=concept, chat_id=chat_id, limit=10, case_sensitive=False, fuzzy=True
            )
            is_fuzzy_match = True

        if jargon_results:
            # 找到结果
            if is_fuzzy_match:
                # 模糊匹配
                output_parts = [f"未精确匹配到'{concept}'"]
                for result in jargon_results:
                    found_content = result.get("content", "").strip()
                    meaning = result.get("meaning", "").strip()
                    if found_content and meaning:
                        output_parts.append(f"找到 '{found_content}' 的含义为：{meaning}")
                results.append("，".join(output_parts))
                logger.info(f"在jargon库中找到匹配（模糊搜索）: {concept}，找到{len(jargon_results)}条结果")
            else:
                # 精确匹配
                output_parts = []
                for result in jargon_results:
                    meaning = result.get("meaning", "").strip()
                    if meaning:
                        output_parts.append(f"'{concept}' 为黑话或者网络简写，含义为：{meaning}")
                results.append("；".join(output_parts) if len(output_parts) > 1 else output_parts[0])
                logger.info(f"在jargon库中找到匹配（精确匹配）: {concept}，找到{len(jargon_results)}条结果")
        else:
            # 未找到，不返回占位信息，只记录日志
            logger.info(f"在jargon库中未找到匹配: {concept}")

    if results:
        return "【概念检索结果】\n" + "\n".join(results) + "\n"
    return ""


async def _react_agent_solve_question(
    question: str, chat_id: str, max_iterations: int = 5, timeout: float = 30.0, initial_info: str = ""
) -> Tuple[bool, str, List[Dict[str, Any]], bool]:
    """使用ReAct架构的Agent来解决问题

    Args:
        question: 要回答的问题
        chat_id: 聊天ID
        max_iterations: 最大迭代次数
        timeout: 超时时间（秒）
        initial_info: 初始信息（如概念检索结果），将作为collected_info的初始值

    Returns:
        Tuple[bool, str, List[Dict[str, Any]], bool]: (是否找到答案, 答案内容, 思考步骤列表, 是否超时)
    """
    start_time = time.time()
    collected_info = _append_collected_info("", initial_info)
    configured_budget = max(1, max_iterations or 1)
    dynamic_budget = _calculate_iteration_budget(question, collected_info)
    max_iterations = max(1, min(configured_budget, dynamic_budget or configured_budget))
    thinking_steps = []
    is_timeout = False
    conversation_messages: List[Message] = []

    for iteration in range(max_iterations):
        # 检查超时
        if time.time() - start_time > timeout:
            logger.warning(f"ReAct Agent超时，已迭代{iteration}次")
            is_timeout = True
            break

        # 获取工具注册器
        tool_registry = get_tool_registry()

        # 获取bot_name
        bot_name = global_config.bot.nickname

        # 获取当前时间
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 计算剩余迭代次数
        current_iteration = iteration + 1
        remaining_iterations = max_iterations - current_iteration
        is_final_iteration = current_iteration >= max_iterations

        # 准备本次迭代的工具列表
        tool_definitions = []
        if is_final_iteration:
            # 最后一次迭代，只提供结果提交工具
            tool_definitions = SPECIAL_TOOL_DEFINITIONS
            logger.info(
                f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}（最后一次迭代，仅提供提交工具）"
            )

            prompt = await global_prompt_manager.format_prompt(
                "memory_retrieval_react_final_prompt",
                bot_name=bot_name,
                time_now=time_now,
                question=question,
                collected_info=_prepare_collected_info_for_prompt(collected_info),
                current_iteration=current_iteration,
                remaining_iterations=remaining_iterations,
                max_iterations=max_iterations,
            )
        else:
            # 非最终迭代，提供所有检索工具 + 结果提交工具
            tool_definitions = tool_registry.get_tool_definitions()
            tool_definitions.extend(SPECIAL_TOOL_DEFINITIONS)
            
            logger.info(
                f"ReAct Agent 第 {iteration + 1} 次迭代，问题: {question}|可用工具数量: {len(tool_definitions)}"
            )

            # 检查LPMM是否有内容
            lpmm_hint = ""
            try:
                qa_manager = get_qa_manager()
                if (
                    qa_manager
                    and qa_manager.embed_manager
                    and qa_manager.embed_manager.paragraphs_embedding_store.store
                ):
                    lpmm_hint = "- 如果不确定查询类别, 也可以使用 lpmm 知识库查询"
            except Exception as e:
                logger.warning(f"检查LPMM内容失败: {e}")

            prompt = await global_prompt_manager.format_prompt(
                "memory_retrieval_react_prompt_head",
                bot_name=bot_name,
                time_now=time_now,
                question=question,
                collected_info=_prepare_collected_info_for_prompt(collected_info),
                current_iteration=current_iteration,
                remaining_iterations=remaining_iterations,
                max_iterations=max_iterations,
                lpmm_hint=lpmm_hint,
            )

        def message_factory(
            _client,
            *,
            _prompt: str = prompt,
            _conversation_messages: List[Message] = conversation_messages,
        ) -> List[Message]:
            messages: List[Message] = []

            system_builder = MessageBuilder()
            system_builder.set_role(RoleType.System)
            system_builder.add_text_content(_prompt)
            messages.append(system_builder.build())

            messages.extend(_conversation_messages)

            if global_config.debug.show_memory_prompt:
                # 优化日志展示 - 合并所有消息到一条日志
                log_lines = []
                for idx, msg in enumerate(messages, 1):
                    role_name = msg.role.value if hasattr(msg.role, "value") else str(msg.role)

                    # 处理内容 - 显示完整内容，不截断
                    if isinstance(msg.content, str):
                        full_content = msg.content
                        content_type = "文本"
                    elif isinstance(msg.content, list):
                        text_parts = [item for item in msg.content if isinstance(item, str)]
                        image_count = len([item for item in msg.content if isinstance(item, tuple)])
                        full_content = "".join(text_parts) if text_parts else ""
                        content_type = f"混合({len(text_parts)}段文本, {image_count}张图片)"
                    else:
                        full_content = str(msg.content)
                        content_type = "未知"

                    # 构建单条消息的日志信息
                    msg_info = f"\n[消息 {idx}] 角色: {role_name} 内容类型: {content_type}\n========================================"

                    if full_content:
                        msg_info += f"\n{full_content}"

                    if msg.tool_calls:
                        msg_info += f"\n  工具调用: {len(msg.tool_calls)}个"
                        for tool_call in msg.tool_calls:
                            msg_info += f"\n    - {tool_call}"

                    if msg.tool_call_id:
                        msg_info += f"\n  工具调用ID: {msg.tool_call_id}"

                    log_lines.append(msg_info)

                # 合并所有消息为一条日志输出
                logger.info(f"消息列表 (共{len(messages)}条):{''.join(log_lines)}")

            return messages

        (
            success,
            response,
            reasoning_content,
            model_name,
            tool_calls,
        ) = await llm_api.generate_with_model_with_tools_by_message_factory(
            message_factory,
            model_config=model_config.model_task_config.tool_use,
            tool_options=tool_definitions,
            request_type="memory.react",
        )

        logger.info(
            f"ReAct Agent 第 {iteration + 1} 次迭代 模型: {model_name} ，调用工具数量: {len(tool_calls) if tool_calls else 0} ，调用工具响应: {response}"
        )

        if not success:
            logger.error(f"ReAct Agent LLM调用失败: {response}")
            break

        assistant_message: Optional[Message] = None
        if tool_calls:
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            if response and response.strip():
                assistant_builder.add_text_content(response)
            assistant_builder.set_tool_calls(tool_calls)
            assistant_message = assistant_builder.build()
        elif response and response.strip():
            assistant_builder = MessageBuilder()
            assistant_builder.set_role(RoleType.Assistant)
            assistant_builder.add_text_content(response)
            assistant_message = assistant_builder.build()

        # 记录思考步骤
        step = {"iteration": iteration + 1, "thought": response, "actions": [], "observations": []}

        if assistant_message:
            conversation_messages.append(assistant_message)

        # 记录思考过程到collected_info中
        if reasoning_content or response:
            thought_summary = reasoning_content or (response[:200] if response else "")
            if thought_summary:
                collected_info = _append_collected_info(collected_info, f"[思考] {thought_summary}")

        # 处理工具调用
        if not tool_calls:
            # 没有工具调用，说明LLM在思考中已经给出了答案（但这是不应该的，因为我们强制用工具）
            # 或者只是纯思考
            if response and response.strip():
                step["observations"] = [f"思考完成，但未调用工具。响应: {response}"]
                logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 思考完成但未调用工具: {response[:100]}...")
                collected_info = _append_collected_info(collected_info, f"思考: {response}")
                thinking_steps.append(step)
                continue
            else:
                logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 无工具调用且无响应")
                step["observations"] = ["无响应且无工具调用"]
                thinking_steps.append(step)
                break

        # 处理工具调用
        tool_tasks = []
        found_answer_content = None
        not_enough_info_reason = None
        
        # 优先处理结束工具
        for tool_call in tool_calls:
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}
            
            if tool_name == "found_answer":
                found_answer_content = tool_args.get("answer", "")
                step["actions"].append({"action_type": "found_answer", "action_params": tool_args})
                step["observations"] = ["Agent调用found_answer提交答案"]
                thinking_steps.append(step)
                logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 调用found_answer: {found_answer_content[:100]}...")
                return True, found_answer_content, thinking_steps, False
            
            elif tool_name == "not_enough_info":
                not_enough_info_reason = tool_args.get("reason", "未知原因")
                step["actions"].append({"action_type": "not_enough_info", "action_params": tool_args})
                step["observations"] = ["Agent调用not_enough_info提交原因"]
                thinking_steps.append(step)
                logger.info(f"ReAct Agent 第 {iteration + 1} 次迭代 调用not_enough_info: {not_enough_info_reason[:100]}...")
                return False, not_enough_info_reason, thinking_steps, False

        # 普通工具调用
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.func_name
            tool_args = tool_call.args or {}

            logger.info(
                f"ReAct Agent 第 {iteration + 1} 次迭代 工具调用 {i + 1}/{len(tool_calls)}: {tool_name}({tool_args})"
            )

            tool = tool_registry.get_tool(tool_name)
            if tool:
                # 准备工具参数
                tool_params = tool_args.copy()

                # 如果工具函数签名需要chat_id，添加它
                import inspect
                sig = inspect.signature(tool.execute_func)
                if "chat_id" in sig.parameters:
                    tool_params["chat_id"] = chat_id

                # 创建异步任务
                async def execute_single_tool(tool_instance, params, tool_name_str, iter_num):
                    try:
                        observation = await tool_instance.execute(**params)
                        param_str = ", ".join([f"{k}={v}" for k, v in params.items() if k != "chat_id"])
                        return f"查询{tool_name_str}({param_str})的结果：{observation}"
                    except Exception as e:
                        error_msg = f"工具执行失败: {str(e)}"
                        logger.error(f"ReAct Agent 第 {iter_num + 1} 次迭代 工具 {tool_name_str} {error_msg}")
                        return f"查询{tool_name_str}失败: {error_msg}"

                tool_tasks.append(execute_single_tool(tool, tool_params, tool_name, iteration))
                step["actions"].append({"action_type": tool_name, "action_params": tool_args})
            else:
                error_msg = f"未知的工具类型: {tool_name}"
                logger.warning(f"ReAct Agent 第 {iteration + 1} 次迭代 工具 {i + 1}/{len(tool_calls)} {error_msg}")
                tool_tasks.append(asyncio.create_task(asyncio.sleep(0, result=f"查询{tool_name}失败: {error_msg}")))

        # 并行执行所有工具
        if tool_tasks:
            observations = await asyncio.gather(*tool_tasks, return_exceptions=True)

            # 处理执行结果
            for i, (tool_call_item, observation) in enumerate(zip(tool_calls, observations, strict=False)):
                if isinstance(observation, Exception):
                    observation = f"工具执行异常: {str(observation)}"
                    logger.error(f"ReAct Agent 第 {iteration + 1} 次迭代 工具 {i + 1} 执行异常: {observation}")

                observation_text = observation if isinstance(observation, str) else str(observation)
                
                # 长度截断
                if len(observation_text) > MAX_OBSERVATION_LENGTH:
                    observation_text = observation_text[:MAX_OBSERVATION_LENGTH] + "...(结果过长已截断)"
                
                step["observations"].append(observation_text)
                collected_info = _append_collected_info(collected_info, observation_text)
                if observation_text.strip():
                    tool_builder = MessageBuilder()
                    tool_builder.set_role(RoleType.Tool)
                    tool_builder.add_text_content(observation_text)
                    tool_builder.add_tool_call(tool_call_item.call_id)
                    conversation_messages.append(tool_builder.build())

        thinking_steps.append(step)

    # 达到最大迭代次数或超时，但Agent没有明确返回found_answer
    if collected_info:
        logger.warning(
            f"ReAct Agent达到最大迭代次数或超时，但未明确返回found_answer。已收集信息: {collected_info[:100]}..."
        )
    if is_timeout:
        logger.warning("ReAct Agent超时，直接视为not_enough_info")
    else:
        logger.warning("ReAct Agent达到最大迭代次数，直接视为not_enough_info")
    return False, "未找到相关信息", thinking_steps, is_timeout


def _get_recent_query_history(chat_id: str, time_window_seconds: float = 300.0) -> str:
    """获取最近一段时间内的查询历史

    Args:
        chat_id: 聊天ID
        time_window_seconds: 时间窗口（秒），默认10分钟

    Returns:
        str: 格式化的查询历史字符串
    """
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        # 查询最近时间窗口内的记录，按更新时间倒序
        records = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.update_time >= start_time))
            .order_by(ThinkingBack.update_time.desc())
            .limit(5)  # 最多返回5条最近的记录
        )

        if not records.exists():
            return ""

        history_lines = []
        history_lines.append("最近已查询的问题和结果：")

        for record in records:
            status = "✓ 已找到答案" if record.found_answer else "✗ 未找到答案"
            answer_preview = ""
            # 只有找到答案时才显示答案内容
            if record.found_answer and record.answer:
                # 截取答案前100字符
                answer_preview = record.answer[:100]
                if len(record.answer) > 100:
                    answer_preview += "..."

            history_lines.append(f"- 问题：{record.question}")
            history_lines.append(f"  状态：{status}")
            if answer_preview:
                history_lines.append(f"  答案：{answer_preview}")
            history_lines.append("")  # 空行分隔

        return "\n".join(history_lines)

    except Exception as e:
        logger.error(f"获取查询历史失败: {e}")
        return ""


def _get_cached_memories(chat_id: str, time_window_seconds: float = 300.0) -> List[str]:
    """获取最近一段时间内缓存的记忆（只返回找到答案的记录）

    Args:
        chat_id: 聊天ID
        time_window_seconds: 时间窗口（秒），默认300秒（5分钟）

    Returns:
        List[str]: 格式化的记忆列表，每个元素格式为 "问题：xxx\n答案：xxx"
    """
    try:
        current_time = time.time()
        start_time = current_time - time_window_seconds

        # 查询最近时间窗口内找到答案的记录，按更新时间倒序
        records = (
            ThinkingBack.select()
            .where(
                (ThinkingBack.chat_id == chat_id)
                & (ThinkingBack.update_time >= start_time)
                & (ThinkingBack.found_answer == 1)
            )
            .order_by(ThinkingBack.update_time.desc())
            .limit(5)  # 最多返回5条最近的记录
        )

        if not records.exists():
            return []

        cached_memories: List[str] = []
        for record in records:
            if not record.answer:
                continue

            thinking_steps = _load_thinking_steps(record.thinking_steps)
            cached_memories.append(
                _format_answer_payload(
                    record.question,
                    record.answer,
                    thinking_steps,
                    from_cache=True,
                )
            )

        return cached_memories

    except Exception as e:
        logger.error(f"获取缓存记忆失败: {e}")
        return []


def _query_thinking_back(chat_id: str, question: str) -> Optional[Tuple[bool, str, List[Dict[str, Any]]]]:
    """从thinking_back数据库中查询是否有现成的答案

    Args:
        chat_id: 聊天ID
        question: 问题

    Returns:
        Optional[Tuple[bool, str]]: 如果找到记录，返回(found_answer, answer)，否则返回None
            found_answer: 是否找到答案（True表示found_answer=1，False表示found_answer=0）
            answer: 答案内容
    """
    try:
        # 查询相同chat_id和问题的所有记录（包括found_answer为0和1的）
        # 按更新时间倒序，获取最新的记录
        records = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.question == question))
            .order_by(ThinkingBack.update_time.desc())
            .limit(1)
        )

        if records.exists():
            record = records.get()
            found_answer = bool(record.found_answer)
            answer = record.answer or ""
            steps = _load_thinking_steps(record.thinking_steps)
            logger.info(f"在thinking_back中找到记录，问题: {question[:50]}...，found_answer: {found_answer}")
            return found_answer, answer, steps

        return None

    except Exception as e:
        logger.error(f"查询thinking_back失败: {e}")
        return None


async def _analyze_question_answer(question: str, answer: str, chat_id: str) -> None:
    """异步分析问题和答案的类别，并存储到相应系统

    Args:
        question: 问题
        answer: 答案
        chat_id: 聊天ID
    """
    try:
        # 使用LLM分析类别
        analysis_prompt = (
            "你需要阅读一问一答并提取其中的关键信息，可同时包含多类：\n"
            "- 人物信息：包含某位用户的喜好、习惯、经历等事实\n"
            "- 黑话：解释网络黑话/缩写/自创词（如\"yyds\"、\"社死\"）\n"
            "- 其他：除上述两类之外\n\n"
            "输出一个 JSON 数组，每个元素描述一条命中结果，格式如下（字段均为字符串，confidence 为 0-1 之间的数字字符串）：\n"
            "[\n  {\n"
            '    "category": "人物信息" 或 "黑话" 或 "其他",\n'
            '    "person_name": "若为人物信息则给出人名，否则留空",\n'
            '    "memory_content": "若为人物信息则给出一句简短记忆，否则留空",\n'
            '    "jargon_keyword": "若为黑话则给出词条，否则留空",\n'
            '    "confidence": "0.0 到 1.0 的数字，表示判定置信度"\n'
            "  }\n]\n"
            "无命中请输出空数组 []。禁止输出除 JSON 外的任何内容。\n\n"
            f"问题：\"{question}\"\n"
            f"答案：\"{answer}\""
        )

        success, response, _, _ = await llm_api.generate_with_model(
            analysis_prompt,
            model_config=model_config.model_task_config.utils,
            request_type="memory.analyze_qa",
        )

        if not success:
            logger.error(f"分析问题和答案失败: {response}")
            return

        # 解析JSON响应
        try:
            json_pattern = r"```json\s*(.*?)\s*```"
            matches = re.findall(json_pattern, response, re.DOTALL)
            json_candidates = matches if matches else [response.strip()]

            parsed_entries: List[Dict[str, Any]] = []
            for candidate in reversed(json_candidates):
                try:
                    repaired_json = repair_json(candidate)
                    parsed_candidate = json.loads(repaired_json)
                    if isinstance(parsed_candidate, dict):
                        parsed_entries = [parsed_candidate]
                    elif isinstance(parsed_candidate, list):
                        parsed_entries = [item for item in parsed_candidate if isinstance(item, dict)]
                    else:
                        continue
                    if parsed_entries:
                        break
                except Exception:
                    continue

            if not parsed_entries:
                raise ValueError("LLM响应未返回有效的对象列表")

            stored_any = False
            for entry in parsed_entries:
                category = _safe_str(entry.get("category"))
                if category not in {"人物信息", "黑话", "其他"}:
                    logger.debug(
                        f"跳过未知分类: {category}，问题: {question[:50]}..."
                    )
                    continue

                confidence_raw = _safe_str(entry.get("confidence"))
                try:
                    confidence = float(confidence_raw) if confidence_raw else 0.0
                except ValueError:
                    confidence = 0.0

                if confidence < MIN_QA_ANALYSIS_CONFIDENCE:
                    logger.debug(
                        f"置信度过低({confidence:.2f}< {MIN_QA_ANALYSIS_CONFIDENCE}),跳过分类 {category}，问题: {question[:50]}..."
                    )
                    continue

                if category == "黑话":
                    jargon_keyword = _safe_str(entry.get("jargon_keyword"))
                    if not jargon_keyword:
                        continue
                    from src.jargon.jargon_miner import store_jargon_from_answer

                    await store_jargon_from_answer(jargon_keyword, answer, chat_id)
                    stored_any = True
                elif category == "人物信息":
                    person_name = _safe_str(entry.get("person_name"))[:32]
                    memory_content = _safe_str(entry.get("memory_content"))[:120]
                    if not (person_name and memory_content):
                        continue
                    from src.person_info.person_info import store_person_memory_from_answer

                    await store_person_memory_from_answer(person_name, memory_content, chat_id)
                    stored_any = True

            if not stored_any:
                logger.info(f"本次分析未提取可存储的黑话/人物信息，问题: {question[:50]}...")

        except Exception as e:
            logger.error(f"解析分析结果失败: {e}, 响应: {response[:200]}...")

    except Exception as e:
        logger.error(f"分析问题和答案时发生异常: {e}")


def _store_thinking_back(
    chat_id: str, question: str, context: str, found_answer: bool, answer: str, thinking_steps: List[Dict[str, Any]]
) -> None:
    """存储或更新思考过程到数据库（如果已存在则更新，否则创建）

    Args:
        chat_id: 聊天ID
        question: 问题
        context: 上下文信息
        found_answer: 是否找到答案
        answer: 答案内容
        thinking_steps: 思考步骤列表
    """
    try:
        now = time.time()

        # 先查询是否已存在相同chat_id和问题的记录
        existing = (
            ThinkingBack.select()
            .where((ThinkingBack.chat_id == chat_id) & (ThinkingBack.question == question))
            .order_by(ThinkingBack.update_time.desc())
            .limit(1)
        )

        if existing.exists():
            # 更新现有记录
            record = existing.get()
            record.context = context
            record.found_answer = found_answer
            record.answer = answer
            record.thinking_steps = json.dumps(thinking_steps, ensure_ascii=False)
            record.update_time = now
            record.save()
            logger.info(f"已更新思考过程到数据库，问题: {question[:50]}...")
        else:
            # 创建新记录
            ThinkingBack.create(
                chat_id=chat_id,
                question=question,
                context=context,
                found_answer=found_answer,
                answer=answer,
                thinking_steps=json.dumps(thinking_steps, ensure_ascii=False),
                create_time=now,
                update_time=now,
            )
            logger.info(f"已创建思考过程到数据库，问题: {question[:50]}...")
    except Exception as e:
        logger.error(f"存储思考过程失败: {e}")


async def _process_single_question(question: str, chat_id: str, context: str, initial_info: str = "") -> Optional[str]:
    """处理单个问题的查询（包含缓存检查逻辑）

    Args:
        question: 要查询的问题
        chat_id: 聊天ID
        context: 上下文信息
        initial_info: 初始信息（如概念检索结果），将传递给ReAct Agent

    Returns:
        Optional[str]: 如果找到答案，返回格式化的结果字符串，否则返回None
    """
    logger.info(f"开始处理问题: {question}")

    await asyncio.to_thread(_cleanup_stale_not_found_thinking_back)

    question_initial_info = initial_info or ""

    # 预先进行一次LPMM知识库查询，作为后续ReAct Agent的辅助信息
    if global_config.lpmm_knowledge.enable:
        try:
            lpmm_result = await query_lpmm_knowledge(question, limit=2)
            if lpmm_result and lpmm_result.startswith("你从LPMM知识库中找到"):
                if question_initial_info:
                    question_initial_info += "\n"
                question_initial_info += f"【LPMM知识库预查询】\n{lpmm_result}"
                logger.info(f"LPMM预查询命中，问题: {question[:50]}...")
            else:
                logger.info(f"LPMM预查询未命中或未找到信息，问题: {question[:50]}...")
        except Exception as e:
            logger.error(f"LPMM预查询失败，问题: {question[:50]}... 错误: {e}")

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
                return _format_answer_payload(
                    question,
                    cached_answer,
                    cached_steps,
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
            # 创建异步任务分析问题和答案
            asyncio.create_task(_analyze_question_answer(question, answer, chat_id))
            return _format_answer_payload(
                question,
                answer,
                thinking_steps,
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
        recent_query_history_raw = await asyncio.to_thread(
            _get_recent_query_history, chat_id, time_window_seconds=300.0
        )
        has_recent_query_history = bool(recent_query_history_raw)
        recent_query_history = recent_query_history_raw or "最近没有查询记录。"

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
        logger.info(f"解析到 {len(questions)} 个问题: {questions}")

        # 对概念进行jargon检索，作为初始信息
        initial_info = ""
        if concepts:
            logger.info(f"开始对 {len(concepts)} 个概念进行jargon检索")
            concept_info = await _retrieve_concepts_with_jargon(concepts, chat_id)
            if concept_info:
                initial_info += concept_info
                logger.info(f"概念检索完成，结果: {concept_info[:200]}...")
            else:
                logger.info("概念检索未找到任何结果")

        # 获取缓存的记忆（与question时使用相同的时间窗口和数量限制）
        cached_memories = await asyncio.to_thread(_get_cached_memories, chat_id, time_window_seconds=300.0)

        if not questions:
            if _should_use_fallback_question(target, has_recent_query_history=has_recent_query_history):
                fallback = _build_fallback_question(sender, target)
                if fallback:
                    logger.info(f"问题列表为空，使用自动回退问题: {fallback}")
                    questions = [fallback]
            else:
                logger.info("问题列表为空，且启发式判断无需触发回退问题")

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

        # 收集所有有效结果
        all_results = []
        current_questions: Set[str] = set()  # 用于去重，避免缓存和当次查询重复
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"处理问题 '{questions[i]}' 时发生异常: {result}")
            elif result is not None:
                all_results.append(result)
                question_line = _extract_question_from_block(result)
                if question_line:
                    current_questions.add(question_line)
                    logger.debug(f"提取到问题：{question_line[:50]}...")

        # 将缓存的记忆添加到结果中（排除当次查询已包含的问题，避免重复）
        for cached_memory in cached_memories:
            question_line = _extract_question_from_block(cached_memory)
            if not question_line:
                continue

            # 只有当次查询中没有相同问题时，才添加缓存记忆
            if question_line not in current_questions:
                all_results.append(cached_memory)
                logger.debug(f"添加缓存记忆: {question_line[:50]}...")

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


def _parse_questions_json(response: str) -> Tuple[List[str], List[str]]:
    """解析问题JSON，返回概念列表和问题列表

    Args:
        response: LLM返回的响应

    Returns:
        Tuple[List[str], List[str]]: (概念列表, 问题列表)
    """
    try:
        # 尝试提取JSON（可能包含在```json代码块中）
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            json_str = matches[0]
        else:
            # 尝试直接解析整个响应
            json_str = response.strip()

        # 修复可能的JSON错误
        repaired_json = repair_json(json_str)

        # 解析JSON
        parsed = json.loads(repaired_json)

        # 只支持新格式：包含concepts和questions的对象
        if not isinstance(parsed, dict):
            logger.warning(f"解析的JSON不是对象格式: {parsed}")
            return [], []

        concepts_raw = parsed.get("concepts", [])
        questions_raw = parsed.get("questions", [])

        # 确保是列表
        if not isinstance(concepts_raw, list):
            concepts_raw = []
        if not isinstance(questions_raw, list):
            questions_raw = []

        # 确保所有元素都是字符串
        concepts = [c for c in concepts_raw if isinstance(c, str) and c.strip()]
        questions = [q for q in questions_raw if isinstance(q, str) and q.strip()]

        return concepts, questions

    except Exception as e:
        logger.error(f"解析问题JSON失败: {e}, 响应内容: {response[:200]}...")
        return [], []
