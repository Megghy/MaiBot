import time
import asyncio
from typing import Final

from src.common.logger import get_logger
from src.config.config import global_config, model_config
from src.chat.message_receive.chat_stream import get_chat_manager
from src.chat.utils.prompt_builder import Prompt, global_prompt_manager
from src.chat.utils.chat_message_builder import build_readable_messages, get_raw_msg_by_timestamp_with_chat_inclusive
from src.llm_models.utils_model import LLMRequest
from src.manager.async_task_manager import AsyncTask, async_task_manager


logger = get_logger("mood")

# 多阶段情绪回归间隔（秒），参考记忆遗忘的分阶段调度
_MOOD_DECAY_PHASES: Final[tuple[int, ...]] = (180, 600, 1800)
_DEFAULT_STATE_REFRESH_SECONDS: Final[int] = 120


def init_prompt():
    Prompt(
        """
{chat_talking_prompt}
以上是群里正在进行的聊天记录

{identity_block}
你先前的情绪状态是：{mood_state}
你的情绪特点是:{emotion_style}

现在，请你根据先前的情绪状态和现在的聊天内容，总结推断你现在的情绪状态，用简短的词句来描述情绪状态
请只输出新的情绪状态，不要输出其他内容：
""",
        "get_mood_prompt",
    )

    Prompt(
        """
{chat_talking_prompt}
以上是群里最近的聊天记录

{identity_block}
你之前的情绪状态是：{mood_state}

距离你上次关注群里消息已经过去了一段时间，你冷静了下来，请你输出一句话或几个词来描述你现在的情绪状态
你的情绪特点是:{emotion_style}
请只输出新的情绪状态，不要输出其他内容：
""",
        "regress_mood_prompt",
    )


class ChatMood:
    def __init__(self, chat_id: str):
        self.chat_id: str = chat_id

        chat_manager = get_chat_manager()
        self.chat_stream = chat_manager.get_stream(self.chat_id)

        if not self.chat_stream:
            raise ValueError(f"Chat stream for chat_id {chat_id} not found")

        self.log_prefix = f"[{self.chat_stream.group_info.group_name if self.chat_stream.group_info else self.chat_stream.user_info.user_nickname}]"

        self.mood_state: str = "感觉很平静"

        self.regression_count: int = 0

        self.mood_model = LLMRequest(model_set=model_config.model_task_config.utils, request_type="mood")

        self.last_change_time: float = 0
        self._state_lock = asyncio.Lock()

    async def get_mood(self) -> str:
        self.regression_count = 0

        current_time = time.time()

        logger.info(f"{self.log_prefix} 获取情绪状态")
        message_list_before_now = get_raw_msg_by_timestamp_with_chat_inclusive(
            chat_id=self.chat_id,
            timestamp_start=self.last_change_time,
            timestamp_end=current_time,
            limit=int(global_config.chat.max_context_size / 3),
            limit_mode="last",
        )

        chat_talking_prompt = build_readable_messages(
            message_list_before_now,
            replace_bot_name=True,
            timestamp_mode="normal_no_YMD",
            read_mark=0.0,
            truncate=True,
            show_actions=True,
        )

        bot_name = global_config.bot.nickname
        if global_config.bot.alias_names:
            bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}"
        else:
            bot_nickname = ""

        identity_block = f"你的名字是{bot_name}{bot_nickname}"

        prompt = await global_prompt_manager.format_prompt(
            "get_mood_prompt",
            chat_talking_prompt=chat_talking_prompt,
            identity_block=identity_block,
            mood_state=self.mood_state,
            emotion_style=global_config.mood.emotion_style,
        )

        response, (reasoning_content, _, _) = await self.mood_model.generate_response_async(
            prompt=prompt, temperature=0.7
        )
        if global_config.debug.show_prompt:
            logger.info(f"{self.log_prefix} prompt: {prompt}")
            logger.info(f"{self.log_prefix} response: {response}")
            logger.info(f"{self.log_prefix} reasoning_content: {reasoning_content}")

        logger.info(f"{self.log_prefix} 情绪状态更新为: {response}")

        self.mood_state = response

        self.last_change_time = current_time

        return response

    async def ensure_recent_state(self, refresh_interval: float = _DEFAULT_STATE_REFRESH_SECONDS) -> None:
        """按需刷新情绪，避免频繁触发LLM"""
        if refresh_interval <= 0:
            await self.get_mood()
            return

        now = time.time()
        if now - self.last_change_time < refresh_interval:
            return

        async with self._state_lock:
            # 双重检查，避免并发重复刷新
            if time.time() - self.last_change_time < refresh_interval:
                return
            await self.get_mood()

    def _describe_trend(self) -> str:
        stage_desc = ["情绪刚刚更新", "正在平复", "趋于平静"]
        idx = min(self.regression_count, len(stage_desc) - 1)
        idle_minutes = max(0, int((time.time() - self.last_change_time) / 60))
        if idle_minutes <= 1:
            idle_text = "刚刚变化"
        else:
            idle_text = f"已持续{idle_minutes}分钟"
        return f"{stage_desc[idx]}，{idle_text}"

    def build_prompt_block(self) -> str:
        if not self.mood_state:
            return ""
        trend = self._describe_trend()
        return f"当前心情：{self.mood_state}（{trend}）"

    async def regress_mood(self):
        message_time = time.time()
        message_list_before_now = get_raw_msg_by_timestamp_with_chat_inclusive(
            chat_id=self.chat_id,
            timestamp_start=self.last_change_time,
            timestamp_end=message_time,
            limit=15,
            limit_mode="last",
        )

        chat_talking_prompt = build_readable_messages(
            message_list_before_now,
            replace_bot_name=True,
            timestamp_mode="normal_no_YMD",
            read_mark=0.0,
            truncate=True,
            show_actions=True,
        )

        bot_name = global_config.bot.nickname
        if global_config.bot.alias_names:
            bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}"
        else:
            bot_nickname = ""

        identity_block = f"你的名字是{bot_name}{bot_nickname}"

        prompt = await global_prompt_manager.format_prompt(
            "regress_mood_prompt",
            chat_talking_prompt=chat_talking_prompt,
            identity_block=identity_block,
            mood_state=self.mood_state,
            emotion_style=global_config.mood.emotion_style,
        )

        response, (reasoning_content, _, _) = await self.mood_model.generate_response_async(
            prompt=prompt, temperature=0.7
        )

        if global_config.debug.show_prompt:
            logger.info(f"{self.log_prefix} prompt: {prompt}")
            logger.info(f"{self.log_prefix} response: {response}")
            logger.info(f"{self.log_prefix} reasoning_content: {reasoning_content}")

        logger.info(f"{self.log_prefix} 情绪状态转变为: {response}")

        self.mood_state = response
        self.last_change_time = message_time

        self.regression_count += 1


class MoodRegressionTask(AsyncTask):
    def __init__(self, mood_manager: "MoodManager"):
        super().__init__(task_name="MoodRegressionTask", run_interval=45)
        self.mood_manager = mood_manager

    async def run(self):
        logger.debug("开始情绪回归任务...")
        now = time.time()
        threshold_factor = max(0.1, float(global_config.mood.mood_update_threshold))
        for mood in self.mood_manager.mood_list:
            if mood.last_change_time == 0:
                continue

            if mood.regression_count >= len(_MOOD_DECAY_PHASES):
                continue

            required_idle = _MOOD_DECAY_PHASES[mood.regression_count] * threshold_factor
            activity_factor = self._calculate_activity_factor(mood, now)
            adjusted_required_idle = required_idle / activity_factor
            idle_duration = now - mood.last_change_time

            if idle_duration < adjusted_required_idle:
                continue

            logger.debug(
                f"{mood.log_prefix} 开始情绪回归, 阶段 {mood.regression_count + 1}, "
                f"已空闲 {idle_duration:.0f}s / 阈值 {required_idle:.0f}s"
            )
            await mood.regress_mood()

    def _calculate_activity_factor(self, mood: ChatMood, now: float) -> float:
        sample_size = max(3, int(getattr(global_config.mood, "regression_activity_sample", 12)))
        try:
            recent_messages = get_raw_msg_by_timestamp_with_chat_inclusive(
                chat_id=mood.chat_id,
                timestamp_start=mood.last_change_time,
                timestamp_end=now,
                limit=sample_size,
                limit_mode="last",
            )
        except Exception as exc:
            logger.debug(f"获取情绪回归消息样本失败: {exc}")
            return 1.0

        activity_count = len(recent_messages or [])
        if activity_count == 0:
            return 1.0

        # 最多将等待时间缩短至原来的一半
        activity_ratio = min(activity_count / sample_size, 1.0)
        return 1.0 + activity_ratio


class MoodManager:
    def __init__(self):
        self.mood_list: list[ChatMood] = []
        """当前情绪状态"""
        self.task_started: bool = False

    async def start(self):
        """启动情绪回归后台任务"""
        if self.task_started:
            return

        task = MoodRegressionTask(self)
        await async_task_manager.add_task(task)
        self.task_started = True
        logger.info("情绪回归任务已启动")

    def get_mood_by_chat_id(self, chat_id: str) -> ChatMood:
        for mood in self.mood_list:
            if mood.chat_id == chat_id:
                return mood

        new_mood = ChatMood(chat_id)
        self.mood_list.append(new_mood)
        return new_mood

    def reset_mood_by_chat_id(self, chat_id: str):
        for mood in self.mood_list:
            if mood.chat_id == chat_id:
                mood.mood_state = "感觉很平静"
                mood.regression_count = 0
                return
        self.mood_list.append(ChatMood(chat_id))

    async def build_mood_prompt_block(self, chat_id: str, refresh_interval: float | None = None) -> str:
        if not global_config.mood.enable_mood:
            return ""

        interval = (
            refresh_interval
            if refresh_interval is not None
            else getattr(global_config.mood, "state_refresh_interval", _DEFAULT_STATE_REFRESH_SECONDS)
        )

        chat_mood = self.get_mood_by_chat_id(chat_id)
        try:
            await chat_mood.ensure_recent_state(interval)
        except Exception as exc:
            logger.warning(f"刷新情绪失败: {exc}")
        return chat_mood.build_prompt_block()


init_prompt()

mood_manager = MoodManager()
"""全局情绪管理器"""
