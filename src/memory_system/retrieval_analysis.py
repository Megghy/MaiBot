from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from json_repair import repair_json

from src.common.logger import get_logger
from src.config.config import model_config
from src.memory_system.retrieval_format import _safe_str
from src.plugin_system.apis import llm_api

logger = get_logger("memory_retrieval")

MIN_QA_ANALYSIS_CONFIDENCE = 0.75
VALID_QA_ANALYSIS_CATEGORIES = {"人物信息", "黑话", "其他"}


@dataclass(slots=True)
class QAAnalysisEntry:
    category: str
    confidence: float
    person_name: str = ""
    memory_content: str = ""
    jargon_keyword: str = ""


def _parse_qa_analysis_entries(response: str, question: str) -> list[QAAnalysisEntry]:
    json_pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(json_pattern, response, re.DOTALL)
    json_candidates = matches if matches else [response.strip()]

    parsed_entries: list[dict[str, Any]] | None = None
    for candidate in reversed(json_candidates):
        try:
            repaired_json = repair_json(candidate)
            parsed_candidate = json.loads(repaired_json)
        except Exception:
            continue

        if isinstance(parsed_candidate, dict):
            parsed_entries = [parsed_candidate]
            break
        if isinstance(parsed_candidate, list):
            parsed_entries = [item for item in parsed_candidate if isinstance(item, dict)]
            if not parsed_candidate:
                return []
            break

    if parsed_entries is None:
        raise ValueError("LLM响应未返回有效的对象列表")

    normalized_entries: list[QAAnalysisEntry] = []
    question_preview = question[:50]

    for entry in parsed_entries:
        category = _safe_str(entry.get("category"))
        if category not in VALID_QA_ANALYSIS_CATEGORIES:
            logger.debug(f"跳过未知分类: {category}，问题: {question_preview}...")
            continue

        confidence_raw = _safe_str(entry.get("confidence"))
        try:
            confidence = float(confidence_raw) if confidence_raw else 0.0
        except ValueError:
            confidence = 0.0

        if confidence < MIN_QA_ANALYSIS_CONFIDENCE:
            logger.debug(
                f"置信度过低({confidence:.2f}< {MIN_QA_ANALYSIS_CONFIDENCE}),跳过分类 {category}，问题: {question_preview}..."
            )
            continue

        if category == "黑话":
            jargon_keyword = _safe_str(entry.get("jargon_keyword"))[:64]
            if not jargon_keyword:
                continue
            normalized_entries.append(
                QAAnalysisEntry(
                    category=category,
                    confidence=confidence,
                    jargon_keyword=jargon_keyword,
                )
            )
            continue

        if category == "人物信息":
            person_name = _safe_str(entry.get("person_name"))[:32]
            memory_content = _safe_str(entry.get("memory_content"))[:120]
            if not (person_name and memory_content):
                continue
            normalized_entries.append(
                QAAnalysisEntry(
                    category=category,
                    confidence=confidence,
                    person_name=person_name,
                    memory_content=memory_content,
                )
            )

    return normalized_entries


async def analyze_question_answer(question: str, answer: str) -> list[QAAnalysisEntry]:
    analysis_prompt = (
        "你需要阅读一问一答并提取其中的关键信息，可同时包含多类：\n"
        "- 人物信息：包含某位用户的喜好、习惯、经历等事实\n"
        '- 黑话：解释网络黑话/缩写/自创词（如"yyds"、"社死"）\n'
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
        f'问题："{question}"\n'
        f'答案："{answer}"'
    )

    try:
        success, response, _, _ = await llm_api.generate_with_model(
            analysis_prompt,
            model_config=model_config.model_task_config.utils,
            request_type="memory.analyze_qa",
        )
    except Exception as exc:
        logger.error(f"分析问题和答案时发生异常: {exc}")
        return []

    if not success:
        logger.error(f"分析问题和答案失败: {response}")
        return []

    try:
        return _parse_qa_analysis_entries(response, question)
    except Exception as exc:
        logger.error(f"解析分析结果失败: {exc}, 响应: {response[:200]}...")
        return []


__all__ = ["QAAnalysisEntry", "analyze_question_answer"]
