import asyncio
import json
from typing import Optional

from json_repair import repair_json

from src.chat.message_receive.chat_stream import get_chat_manager
from src.common.database.database import db
from src.common.database.database_model import PersonInfo
from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest
from src.person_info.memory_schema import (
    get_person_id,
    get_person_id_by_alias,
    get_person_id_by_person_name,
    infer_memory_category,
)

logger = get_logger("person_info")


class PersonInfoManager:
    def __init__(self):
        self.person_name_list = {}
        self.qv_name_llm = LLMRequest(
            model_set=model_config.model_task_config.utils,
            request_type="relation.qv_name",
        )
        try:
            db.connect(reuse_if_open=True)
            if hasattr(db, "execute_sql"):
                db.execute_sql("PRAGMA cache_size = -64000")
                db.execute_sql("PRAGMA temp_store = memory")
                db.execute_sql("PRAGMA mmap_size = 268435456")
            db.create_tables([PersonInfo], safe=True)
        except Exception as exc:
            logger.error(f"数据库连接或 PersonInfo 表创建失败: {exc}")

        try:
            for record in PersonInfo.select(PersonInfo.person_id, PersonInfo.person_name).where(
                PersonInfo.person_name.is_null(False)
            ):
                if record.person_name:
                    self.person_name_list[record.person_id] = record.person_name
            logger.debug(f"已加载 {len(self.person_name_list)} 个用户名称 (Peewee)")
        except Exception as exc:
            logger.error(f"从 Peewee 加载 person_name_list 失败: {exc}")

    @staticmethod
    def _extract_json_from_text(text: str) -> dict:
        try:
            fixed_json = repair_json(text)
            parsed_json = json.loads(fixed_json) if isinstance(fixed_json, str) else fixed_json
            if isinstance(parsed_json, list) and parsed_json:
                parsed_json = parsed_json[0]
            if isinstance(parsed_json, dict):
                return parsed_json
        except Exception as exc:
            logger.warning(f"JSON提取失败: {exc}")

        logger.warning(f"无法从文本中提取有效的JSON字典: {text}")
        logger.info(f"文本: {text}")
        return {"nickname": "", "reason": ""}

    async def _generate_unique_person_name(self, base_name: str) -> str:
        if not base_name or base_name.isspace():
            base_name = "空格"
        if base_name not in self.person_name_list.values():
            return base_name

        counter = 1
        while True:
            new_name = f"{base_name}[{counter}]"
            if new_name not in self.person_name_list.values():
                return new_name
            counter += 1

    async def qv_person_name(
        self,
        person_id: str,
        user_nickname: str,
        user_cardname: str,
        user_avatar: str,
        request: str = "",
    ):
        if not person_id:
            logger.debug("取名失败：person_id不能为空")
            return None

        from src.person_info.person_info import Person

        person = Person(person_id=person_id)
        old_name = person.person_name
        old_reason = person.name_reason
        max_retries = 8
        current_try = 0
        existing_names_str = ""
        current_name_set = set(self.person_name_list.values())

        while current_try < max_retries:
            bot_name = global_config.bot.nickname
            qv_name_prompt = f"你是{bot_name}，一个伪装成人类的AI，你不能让别人发现这一点，"
            qv_name_prompt += f"现在你想给一个用户取一个昵称，用户的qq昵称是{user_nickname}，"
            qv_name_prompt += f"用户的qq群昵称名是{user_cardname}，"
            if user_avatar:
                qv_name_prompt += f"用户的qq头像是{user_avatar}，"
            if old_name:
                qv_name_prompt += f"你之前叫他{old_name}，是因为{old_reason}，"

            qv_name_prompt += f"\n其他取名的要求是：{request}，不要太浮夸，简短，"
            qv_name_prompt += "\n请根据以上用户信息，想想你叫他什么比较好，不要太浮夸，请最好使用用户的qq昵称或群昵称原文，可以稍作修改，优先使用原文。优先使用用户的qq昵称或者群昵称原文。"
            if existing_names_str:
                qv_name_prompt += f"\n请注意，以下名称已被你尝试过或已知存在，请避免：{existing_names_str}。\n"
            if len(current_name_set) < 50 and current_name_set:
                qv_name_prompt += f"已知的其他昵称有: {', '.join(list(current_name_set)[:10])}等。\n"

            qv_name_prompt += "请用json给出你的想法，并给出理由，示例如下："
            qv_name_prompt += """{
                "nickname": "昵称",
                "reason": "理由"
            }"""
            response, _ = await self.qv_name_llm.generate_response_async(qv_name_prompt)
            result = self._extract_json_from_text(response)
            if not result or not result.get("nickname"):
                logger.error("生成的昵称为空或结果格式不正确，重试中...")
                current_try += 1
                continue

            generated_nickname = result["nickname"]
            is_duplicate = generated_nickname in current_name_set
            if is_duplicate:
                logger.info(f"尝试给用户{user_nickname} {person_id} 取名，但是 {generated_nickname} 已存在，重试中...")
            else:

                def _db_check_name_exists_sync(name_to_check: str) -> bool:
                    return PersonInfo.select().where(PersonInfo.person_name == name_to_check).exists()

                if await asyncio.to_thread(_db_check_name_exists_sync, generated_nickname):
                    is_duplicate = True
                    current_name_set.add(generated_nickname)

            if not is_duplicate:
                person.person_name = generated_nickname
                person.name_reason = result.get("reason", "未提供理由")
                person.sync_to_database()
                logger.info(
                    f"成功给用户{user_nickname} {person_id} 取名 {generated_nickname}，理由：{result.get('reason', '未提供理由')}"
                )
                self.person_name_list[person_id] = generated_nickname
                return result

            existing_names_str = (
                f"{existing_names_str}、{generated_nickname}" if existing_names_str else generated_nickname
            )
            logger.debug(f"生成的昵称 {generated_nickname} 已存在，重试中...")
            current_try += 1

        unique_nickname = await self._generate_unique_person_name(user_nickname)
        logger.warning(f"在{max_retries}次尝试后未能生成唯一昵称，使用默认昵称 {unique_nickname}")
        person.person_name = unique_nickname
        person.name_reason = "使用用户原始昵称作为默认值"
        person.sync_to_database()
        self.person_name_list[person_id] = unique_nickname
        return {"nickname": unique_nickname, "reason": "使用用户原始昵称作为默认值"}


person_info_manager = PersonInfoManager()


def _resolve_person_for_memory(person_name: str, chat_stream):
    from src.person_info.person_info import Person

    if person_name == global_config.bot.nickname:
        return Person(platform=global_config.bot.platform, user_id=global_config.bot.qq_account)

    person_id = get_person_id_by_person_name(person_name) or get_person_id_by_alias(person_name)
    if not person_id and chat_stream.user_info:
        user_nickname = (chat_stream.user_info.user_nickname or "").strip()
        if user_nickname and user_nickname == person_name:
            person_id = get_person_id(chat_stream.platform, chat_stream.user_info.user_id)

    if not person_id:
        logger.warning(f"无法确定person_id，person_name: {person_name}, chat_id: {chat_stream.stream_id}")
        return None

    person = Person(person_id=person_id)
    if not person.is_known:
        logger.warning(f"用户 {person_name} (person_id: {person_id}) 尚未认识，无法存储记忆")
        return None

    return person


async def store_person_memory_from_answer(person_name: str, memory_content: str, chat_id: str) -> None:
    try:
        person_name = (person_name or "").strip()
        memory_content = (memory_content or "").strip()
        if not person_name or not memory_content:
            logger.debug("人物名称或记忆内容为空，跳过存储")
            return

        chat_stream = get_chat_manager().get_stream(chat_id)
        if not chat_stream:
            logger.warning(f"无法获取chat_stream for chat_id: {chat_id}")
            return

        group_id: Optional[str] = None
        if getattr(chat_stream, "group_info", None):
            try:
                group_id = str(chat_stream.group_info.group_id)
            except Exception:
                group_id = None

        person = _resolve_person_for_memory(person_name, chat_stream)
        if not person:
            return

        from src.memory_system.memory_writer import MemoryWriteRequest, persist_memory_request

        request = MemoryWriteRequest(
            person=person,
            memory_content=memory_content,
            category=infer_memory_category(memory_content),
            weight=1.0,
            group_id=group_id,
        )
        added = await persist_memory_request(request)

        if added:
            scope = f"群 {group_id}" if group_id else "全局"
            logger.info(f"成功添加人物记忆（{scope}）：{person_name} -> {memory_content}")
        else:
            logger.debug(f"未能添加人物记忆（可能重复）：{person_name} -> {memory_content}")
    except Exception as exc:
        logger.error(f"存储人物记忆失败: {exc}")
