import asyncio
import json
import re
from typing import Dict, List

from json_repair import repair_json

from src.common.logger import get_logger
from src.config.config import model_config
from src.memory_system.retrieval_format import _truncate_text
from src.plugin_system.apis import llm_api

logger = get_logger("memory_retrieval")

MIN_JARGON_FALLBACK_CONFIDENCE = 0.75


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


async def _fallback_explain_concepts_with_llm(concepts: List[str], context: str, chat_id: str) -> Dict[str, str]:
    if not concepts:
        return {}

    from src.jargon.jargon_miner import store_jargon_from_answer

    safe_context = (context or "").strip()
    if safe_context:
        safe_context = _truncate_text(safe_context, 800)

    prompt_parts: List[str] = []
    prompt_parts.append("你是一个帮助解释词语含义的助手。")
    prompt_parts.append(
        "请阅读给定的概念列表和可选的对话上下文，给出每个概念在当前语境下最可能的含义，并评估你的置信度（0-1）。"
    )
    prompt_parts.append("只在你认为自己有足够把握时返回该概念；如果不确定或上下文不足，请不要为该概念返回结果。")
    prompt_parts.append("输出一个 JSON 数组，每个元素是一个对象，格式如下：")
    prompt_parts.append('[{"term": "概念", "meaning": "含义", "confidence": 0.9}]')
    prompt_parts.append("要求：")
    prompt_parts.append("- 只输出 JSON，不要有任何多余文字；")
    prompt_parts.append("- confidence 为 0 到 1 之间的数字，0 表示完全不确定，1 表示非常确定；")
    prompt_parts.append("- 不要编造与常识明显冲突的含义；")
    prompt_parts.append("- 当无法确定某个概念的含义时，直接省略该条目。")
    concepts_text = ", ".join(concepts)
    prompt_parts.append(f"概念列表：{concepts_text}")
    if safe_context:
        prompt_parts.append("相关上下文：")
        prompt_parts.append(safe_context)
    prompt = "\n".join(prompt_parts)

    try:
        success, response, _, _ = await llm_api.generate_with_model(
            prompt,
            model_config=model_config.model_task_config.utils_small,
            request_type="memory.jargon_fallback",
        )
    except Exception as exc:
        logger.error(f"jargon fallback LLM 调用失败: {exc}")
        return {}

    if not success or not response:
        logger.error(f"jargon fallback LLM 返回失败: {response}")
        return {}

    try:
        json_pattern = r"```json\s*(.*?)\s*```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        json_str = matches[0] if matches else response.strip()
        repaired = repair_json(json_str)
        parsed = json.loads(repaired)
    except Exception as exc:
        logger.error(f"解析 jargon fallback JSON 失败: {exc}, 响应: {response[:200]}...")
        return {}

    if isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        logger.warning(f"jargon fallback JSON 不是数组或对象: {parsed}")
        return {}

    results: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or item.get("concept") or "").strip()
        meaning = str(item.get("meaning") or "").strip()
        if not term or not meaning:
            continue
        confidence_raw = item.get("confidence")
        confidence = 0.0
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < MIN_JARGON_FALLBACK_CONFIDENCE:
            continue
        results[term] = meaning

    if not results:
        return {}

    for term, meaning in results.items():
        try:
            asyncio.create_task(store_jargon_from_answer(term, meaning, chat_id))
        except Exception as exc:
            logger.error(f"异步写入 jargon 失败: {exc}")

    return results


async def _retrieve_concepts_with_jargon(concepts: List[str], chat_id: str, context: str = "") -> str:
    """对概念列表进行jargon检索"""
    if not concepts:
        return ""

    from src.common.vector_store.manager import vector_manager
    from src.jargon.jargon_miner import search_jargon

    results = []
    unknown_concepts: List[str] = []
    for concept in concepts:
        concept = concept.strip()
        if not concept:
            continue

        search_method = None
        jargon_results = []

        try:
            vector_results = await vector_manager.search_jargon(concept, k=3)
            if vector_results:
                jargon_results = [
                    {"content": r.get("text", ""), "meaning": r.get("meaning", "")}
                    for r in vector_results
                    if r.get("similarity", 0) >= 0.5 and r.get("meaning")
                ]
                if jargon_results:
                    search_method = "vector"
                    logger.info(f"向量搜索找到 {len(jargon_results)} 条jargon结果: {concept}")
        except Exception as exc:
            logger.warning(f"向量搜索jargon失败: {exc}")

        if not jargon_results:
            jargon_results = await asyncio.to_thread(
                search_jargon, keyword=concept, chat_id=chat_id, limit=5, case_sensitive=False, fuzzy=False
            )
            if jargon_results:
                search_method = "exact"
                logger.info(f"数据库精确匹配找到 {len(jargon_results)} 条jargon结果: {concept}")

        if not jargon_results:
            jargon_results = await asyncio.to_thread(
                search_jargon, keyword=concept, chat_id=chat_id, limit=5, case_sensitive=False, fuzzy=True
            )
            if jargon_results:
                search_method = "fuzzy"
                logger.info(f"数据库模糊搜索找到 {len(jargon_results)} 条jargon结果: {concept}")

        if jargon_results:
            if search_method == "vector":
                output_parts = []
                for result in jargon_results:
                    content = result.get("content", "").strip()
                    meaning = result.get("meaning", "").strip()
                    if content and meaning:
                        output_parts.append(f"'{content}' 的含义为：{meaning}")
                if output_parts:
                    results.append("；".join(output_parts))
            elif search_method == "fuzzy":
                output_parts = [f"未精确匹配到'{concept}'"]
                for result in jargon_results:
                    found_content = result.get("content", "").strip()
                    meaning = result.get("meaning", "").strip()
                    if found_content and meaning:
                        output_parts.append(f"找到 '{found_content}' 的含义为：{meaning}")
                results.append("，".join(output_parts))
            else:
                output_parts = []
                for result in jargon_results:
                    meaning = result.get("meaning", "").strip()
                    if meaning:
                        output_parts.append(f"'{concept}' 为黑话或者网络简写，含义为：{meaning}")
                if output_parts:
                    results.append("；".join(output_parts) if len(output_parts) > 1 else output_parts[0])
        else:
            logger.info(f"在jargon库中未找到匹配: {concept}")
            unknown_concepts.append(concept)

    if unknown_concepts:
        try:
            fallback_map = await _fallback_explain_concepts_with_llm(unknown_concepts, context, chat_id)
        except Exception as exc:
            logger.error(f"jargon fallback 解释失败: {exc}")
            fallback_map = {}
        if fallback_map:
            for concept in unknown_concepts:
                meaning = fallback_map.get(concept)
                if meaning:
                    meaning_text = meaning.strip()
                    if meaning_text:
                        results.append(f"'{concept}' 的含义为：{meaning_text}")

    if results:
        return "【概念检索结果】\n" + "\n".join(results) + "\n"
    return ""
