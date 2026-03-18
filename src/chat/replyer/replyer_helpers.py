import hashlib
import json
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.common.logger import get_logger
from src.chat.utils.prompt_builder import global_prompt_manager
from src.config.config import global_config, model_config
from src.express.expression_selector import expression_selector
from src.mood.mood_manager import mood_manager
from src.person_info.person_info import Person, is_person_known
from src.plugin_system.apis import llm_api

logger = get_logger("base_replyer")


def format_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    if not tool_results:
        return ""

    lines = []
    for tool_result in tool_results:
        tool_name = tool_result.get("tool_name", "unknown")
        content = tool_result.get("content", "")
        result_type = tool_result.get("type", "tool_result")

        if isinstance(content, (dict, list)):
            preview = json.dumps(content, ensure_ascii=False)
        else:
            preview = str(content)

        preview = preview.strip()
        if len(preview) > 200:
            preview = f"{preview[:197]}..."

        lines.append(f"- 【{tool_name}】({result_type}): {preview}")

    if not lines:
        return ""
    return "以下是你通过工具获取到的实时信息：\n" + "\n".join(lines) + "\n以上信息可直接引用，不要额外总结。"


def parse_chat_prompt_config_to_chat_id(chat_prompt_str: str) -> Optional[tuple[str, str]]:
    try:
        parts = chat_prompt_str.split(":", 3)
        if len(parts) != 4:
            return None

        platform, id_str, stream_type, prompt_content = parts
        components = [platform, str(id_str)] if stream_type == "group" else [platform, str(id_str), "private"]
        chat_id = hashlib.md5("_".join(components).encode()).hexdigest()
        return chat_id, prompt_content
    except (ValueError, IndexError):
        return None


def get_chat_prompt_by_type(chat_id: str, target_type: str, chat_prompts: List[str]) -> str:
    if not chat_prompts:
        return ""

    for chat_prompt_str in chat_prompts:
        if not isinstance(chat_prompt_str, str):
            continue

        parts = chat_prompt_str.split(":", 3)
        if len(parts) != 4 or parts[2] != target_type:
            continue

        result = parse_chat_prompt_config_to_chat_id(chat_prompt_str)
        if result is None:
            continue

        config_chat_id, prompt_content = result
        if config_chat_id == chat_id:
            logger.debug(f"匹配到{target_type}聊prompt配置，chat_id: {chat_id}, prompt: {prompt_content[:50]}...")
            return prompt_content

    return ""


async def build_expression_habits_block(
    chat_stream_id: str,
    chat_history: str,
    target: str,
    reply_reason: str = "",
) -> Tuple[str, List[int]]:
    use_expression, _, _ = global_config.expression.get_expression_config_for_chat(chat_stream_id)
    if not use_expression:
        return "", []

    style_habits = []
    selected_expressions, selected_ids = await expression_selector.select_suitable_expressions(
        chat_stream_id,
        chat_history,
        max_num=8,
        target_message=target,
        reply_reason=reply_reason,
    )

    if selected_expressions:
        logger.debug(f"使用处理器选中的{len(selected_expressions)}个表达方式")
        for expr in selected_expressions:
            if isinstance(expr, dict) and "situation" in expr and "style" in expr:
                style_habits.append(f"当{expr['situation']}时，使用 {expr['style']}")
    else:
        logger.debug("没有从处理器获得表达方式，将使用空的表达方式")

    style_habits_str = "\n".join(style_habits)
    if not style_habits_str.strip():
        return "\n", selected_ids

    expression_habits_title = "在回复时,你可以参考以下的语言习惯，不要生硬使用："
    expression_habits_block = f"{style_habits_str}\n"
    return f"{expression_habits_title}\n{expression_habits_block}", selected_ids


async def build_mood_state_prompt_block(chat_stream_id: str) -> str:
    if not global_config.mood.enable_mood:
        return ""
    return await mood_manager.build_mood_prompt_block(chat_stream_id)


def build_keywords_reaction_prompt_text(target: Optional[str]) -> str:
    keywords_reaction_prompt = ""
    try:
        if target is None:
            return keywords_reaction_prompt

        for rule in global_config.keyword_reaction.keyword_rules:
            if any(keyword in target for keyword in rule.keywords):
                logger.info(f"检测到关键词规则：{rule.keywords}，触发反应：{rule.reaction}")
                keywords_reaction_prompt += f"{rule.reaction}，"

        for rule in global_config.keyword_reaction.regex_rules:
            for pattern_str in rule.regex:
                try:
                    pattern = re.compile(pattern_str)
                    if result := pattern.search(target):
                        reaction = rule.reaction
                        for name, content in result.groupdict().items():
                            reaction = reaction.replace(f"[{name}]", content)
                        logger.info(f"匹配到正则表达式：{pattern_str}，触发反应：{reaction}")
                        keywords_reaction_prompt += f"{reaction}，"
                        break
                except re.error as exc:
                    logger.error(f"正则表达式编译错误: {pattern_str}, 错误信息: {str(exc)}")
                    continue
    except Exception as exc:
        logger.error(f"关键词检测与反应时发生异常: {str(exc)}", exc_info=True)

    return keywords_reaction_prompt


def build_personality_prompt_text() -> str:
    bot_name = global_config.bot.nickname
    bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}" if global_config.bot.alias_names else ""

    prompt_personality = global_config.personality.personality
    if (
        global_config.personality.states
        and global_config.personality.state_probability > 0
        and random.random() < global_config.personality.state_probability
    ):
        prompt_personality = random.choice(global_config.personality.states)

    return f"你的名字是{bot_name}{bot_nickname}，你{prompt_personality};"


async def build_tool_results_block(
    tool_executor,
    chat_history: str,
    sender: str,
    target: str,
    *,
    enable_tool: bool = True,
) -> str:
    if not enable_tool:
        return ""

    try:
        tool_results, _, _ = await tool_executor.execute_from_chat_message(
            sender=sender,
            target_message=target,
            chat_history=chat_history,
            return_details=False,
        )
        if not tool_results:
            logger.debug("未获取到任何工具结果")
            return ""

        formatted = format_tool_results(tool_results)
        if formatted:
            logger.info(f"获取到 {len(tool_results)} 个工具结果")
        return formatted
    except Exception as exc:
        logger.error(f"工具信息获取失败: {exc}")
        return ""


async def get_prompt_info_block(tool_executor, message: str, sender: str, target: str) -> str:
    from src.plugins.built_in.knowledge.lpmm_get_knowledge import SearchKnowledgeFromLPMMTool

    logger.debug(f"获取知识库内容，元消息：{message[:30]}...，消息长度: {len(message)}")
    try:
        if not global_config.lpmm_knowledge.enable:
            logger.debug("LPMM知识库未启用，跳过获取知识库内容")
            return ""
        if global_config.lpmm_knowledge.lpmm_mode == "agent":
            return ""

        prompt = await global_prompt_manager.format_prompt(
            "lpmm_get_knowledge_prompt",
            bot_name=global_config.bot.nickname,
            time_now=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            chat_history=message,
            sender=sender,
            target_message=target,
        )
        _, _, _, _, tool_calls = await llm_api.generate_with_model_with_tools(
            prompt,
            model_config=model_config.model_task_config.tool_use,
            tool_options=[SearchKnowledgeFromLPMMTool.get_tool_definition()],
        )
        if not tool_calls:
            logger.debug("模型认为不需要使用LPMM知识库")
            return ""

        start_time = time.time()
        result = await tool_executor.execute_tool_call(tool_calls[0], SearchKnowledgeFromLPMMTool())
        if not result or not result.get("content"):
            logger.debug("从LPMM知识库获取知识失败，返回空知识...")
            return ""

        found_knowledge = result.get("content", "")
        logger.info(f"从LPMM知识库获取知识，相关信息：{found_knowledge[:100]}...，信息长度: {len(found_knowledge)}")
        logger.debug(f"获取知识库内容耗时: {(time.time() - start_time):.3f}秒")
        return f"你有以下这些**知识**：\n{found_knowledge}\n请你**记住上面的知识**，之后可能会用到。\n"
    except Exception as exc:
        logger.error(f"获取知识库内容时发生异常: {str(exc)}")
        return ""


async def build_relation_info_block(chat_content: str, sender: str) -> str:
    if not global_config.relationship.enable_relationship:
        return ""
    if not sender or sender == global_config.bot.nickname:
        return ""
    if not is_person_known(person_name=sender):
        logger.warning(f"未找到用户 {sender} 的ID，跳过信息提取")
        return f"你完全不认识{sender}，不理解ta的相关信息。"

    person = Person(person_name=sender)
    return await person.build_relationship(chat_content)
