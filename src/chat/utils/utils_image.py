import base64
import os
import time
import hashlib
import uuid
import io
import numpy as np
from typing import Optional, Tuple, List

from PIL import Image
from rich.traceback import install

from src.common.logger import get_logger
from src.common.database.database import db
from src.common.database.database_model import Images, ImageDescriptions
from src.config.config import global_config, model_config
from src.llm_models.utils_model import LLMRequest

install(extra_lines=3)

logger = get_logger("chat_image")

PROMPTS = {
    "emoji": (
        "这是一个聊天表情包{gif_hint}。请分析：\n"
        "1. 画面内容（人物/动作/文字）。\n"
        "2. 情绪态度与笑点（反差/热梗/夸张）。\n"
        "3. 适用聊天场景（吐槽/自嘲/阴阳怪气/赞同等）。\n"
        "请用自然口语化的简短中文, 从互联网Meme角度给出描述, 一段话给出, **不进行分段分点**"
    ),
    "image": (
        "分析这张聊天图片。一段话给出, **不进行分段分点**, 尽可能简短但是要足够清晰. \n"
        "先判断类型（表情包/截图/生活照/其他）：\n"
        "- 表情包/梗图：说明笑点、梗及适用场景。\n"
        "- 截图：概括关键文字信息和对话大意，勿罗列细节。\n"
        "- 生活照/风景：描述场景、人物关系及氛围。\n"
        "- 其他：大概描述图片。\n"
        "最后总结对聊天最有用的关键信息。"
    ),
    "emotion": (
        "基于表情包描述提取3-12个聊天风格和内容(目标)的标签\n"
        "要求：标签简练，可以使用词语或短语，逗号分隔，不要解释，去重。\n"
        "描述：'{description}'"
    )
}


def _decode_image_bytes_and_hash(image_base64: str) -> Tuple[bytes, str]:
    if isinstance(image_base64, str):
        image_base64 = image_base64.encode("ascii", errors="ignore").decode("ascii")
    image_bytes = base64.b64decode(image_base64)
    image_hash = hashlib.md5(image_bytes).hexdigest()
    return image_bytes, image_hash


class ImageManager:
    _instance = None
    IMAGE_DIR = "data"  # 图像存储根目录

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._ensure_image_dir()
            self._init_models()
            self._init_db()
            self._initialized = True

    def _ensure_image_dir(self):
        """确保图像存储目录存在"""
        os.makedirs(self.IMAGE_DIR, exist_ok=True)

    def _init_models(self):
        self.vlm = LLMRequest(model_set=model_config.model_task_config.vlm, request_type="image")
        self.vlm_is_gemini_only = False
        try:
            vlm_task = model_config.model_task_config.vlm
            if vlm_task.model_list:
                self.vlm_is_gemini_only = all(
                    model_config.get_provider(model_config.get_model_info(model_name).api_provider).client_type
                    == "gemini"
                    for model_name in vlm_task.model_list
                )
        except Exception as e:
            logger.warning(f"检测 VLM 模型类型失败: {e}")

    def _init_db(self):
        try:
            db.connect(reuse_if_open=True)
            db.create_tables([Images, ImageDescriptions], safe=True)
            self._cleanup_invalid_descriptions()
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")

    def _decode_image(self, image_base64: str) -> Tuple[bytes, str, str]:
        """解码并获取图片信息 -> (bytes, hash, format)"""
        image_bytes, image_hash = _decode_image_bytes_and_hash(image_base64)

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                image_format = (img.format or "PNG").lower()
        except Exception:
            image_format = "unknown"
            
        return image_bytes, image_hash, image_format

    @staticmethod
    def _get_description_from_db(image_hash: str, description_type: str) -> Optional[str]:
        """从数据库获取图片描述

        Args:
            image_hash: 图片哈希值
            description_type: 描述类型 ('emoji' 或 'image')

        Returns:
            Optional[str]: 描述文本，如果不存在则返回None
        """
        try:
            record = ImageDescriptions.get_or_none(
                (ImageDescriptions.image_description_hash == image_hash) & (ImageDescriptions.type == description_type)
            )
            return record.description if record else None
        except Exception as e:
            logger.error(f"从数据库获取描述失败 (Peewee): {str(e)}")
            return None

    @staticmethod
    def _save_description_to_db(image_hash: str, description: str, description_type: str) -> None:
        """保存图片描述到数据库

        Args:
            image_hash: 图片哈希值
            description: 描述文本
            description_type: 描述类型 ('emoji' 或 'image')
        """
        try:
            current_timestamp = time.time()
            defaults = {"description": description, "timestamp": current_timestamp}
            desc_obj, created = ImageDescriptions.get_or_create(
                image_description_hash=image_hash, type=description_type, defaults=defaults
            )
            if not created:  # 如果记录已存在，则更新
                desc_obj.description = description
                desc_obj.timestamp = current_timestamp
                desc_obj.save()
        except Exception as e:
            logger.error(f"保存描述到数据库失败 (Peewee): {str(e)}")

    @staticmethod
    def _cleanup_invalid_descriptions():
        """清理数据库中 description 为空或为 'None' 的记录"""
        invalid_values = ["", "None"]

        # 清理 Images 表
        deleted_images = (
            Images.delete().where((Images.description >> None) | (Images.description << invalid_values)).execute()
        )

        # 清理 ImageDescriptions 表
        deleted_descriptions = (
            ImageDescriptions.delete()
            .where((ImageDescriptions.description >> None) | (ImageDescriptions.description << invalid_values))
            .execute()
        )

        if deleted_images or deleted_descriptions:
            logger.info(f"[清理完成] 删除 Images: {deleted_images} 条, ImageDescriptions: {deleted_descriptions} 条")
        else:
            logger.info("[清理完成] 未发现无效描述记录")

    async def get_emoji_tag(self, image_base64: str) -> str:
        from src.chat.emoji_system.emoji_manager import get_emoji_manager
        _, image_hash, _ = self._decode_image(image_base64)
        
        if emoji := await get_emoji_manager().get_emoji_from_manager(image_hash):
            return f"[表情包：{','.join(emoji.emotion)}]"
        return "[表情包：未知]"

    async def get_emoji_description(self, image_base64: str) -> str:
        """获取表情包描述"""
        try:
            image_bytes, image_hash, image_format = self._decode_image(image_base64)

            # 1. 优先检查 EmojiManager
            from src.chat.emoji_system.emoji_manager import get_emoji_manager
            if tags := await get_emoji_manager().get_emoji_tag_by_hash(image_hash):
                logger.info(f"[缓存命中] 使用已注册表情包描述: {tags}...")
                return f"[表情包：{','.join(tags)}]"

            # 2. 检查 DB 缓存
            if cached := self._get_description_from_db(image_hash, "emoji"):
                logger.info(f"[缓存命中] 使用ImageDescriptions表: {cached[:50]}...")
                return f"[表情包：{cached}]"

            # 3. VLM 生成详细描述
            is_gif = image_format in ["gif", "gif"]
            gif_hint = ""
            final_image_data = image_base64
            target_format = image_format

            if is_gif:
                gif_hint = "（动态图）"
                if not self.vlm_is_gemini_only:
                    if processed := self.transform_gif(image_base64):
                        final_image_data = processed
                        target_format = "jpg"
                        gif_hint = "（动态图关键帧）"

            prompt = PROMPTS["emoji"].format(gif_hint=gif_hint)
            description, _ = await self.vlm.generate_response_for_image(
                prompt, final_image_data, target_format, temperature=0.4
            )

            if not description:
                logger.warning("VLM未能生成表情包详细描述")
                return "[表情包(VLM描述生成失败)]"

            # 4. LLM 生成情感标签
            emotion_prompt = PROMPTS["emotion"].format(description=description)
            emotion_llm = LLMRequest(model_set=model_config.model_task_config.utils, request_type="emoji")
            emotion_result, _ = await emotion_llm.generate_response_async(emotion_prompt, temperature=0.3)

            if not emotion_result:
                # 简单回退策略：取描述的前几个字
                emotion_result = description[:10]

            # 处理标签
            emotions = [e.strip() for e in emotion_result.replace("，", ",").split(",") if e.strip()]
            final_emotion = "，".join(list(dict.fromkeys(emotions))[:10]) or "表情"

            logger.debug(f"[emoji识别] 描述: {description[:30]}... -> 标签: {final_emotion}")

            # 5. 保存结果
            self._save_emoji_record(image_hash, image_bytes, image_format, description, final_emotion)
            return f"[表情包：{final_emotion}]"

        except Exception as e:
            logger.error(f"获取表情包描述失败: {e}")
            return "[表情包(处理失败)]"

    def _save_emoji_record(self, image_hash, image_bytes, image_format, description, final_emotion):
        """保存表情包记录"""
        try:
            filename = f"{int(time.time())}_{image_hash[:8]}.{image_format}"
            emoji_dir = os.path.join(self.IMAGE_DIR, "emoji")
            os.makedirs(emoji_dir, exist_ok=True)
            file_path = os.path.join(emoji_dir, filename)
            
            with open(file_path, "wb") as f:
                f.write(image_bytes)

            defaults = {
                "image_id": str(uuid.uuid4()),
                "path": file_path,
                "type": "emoji",
                "description": description,
                "timestamp": time.time(),
                "vlm_processed": True
            }
            record, created = Images.get_or_create(emoji_hash=image_hash, type="emoji", defaults=defaults)
            if not created:
                record.description = description
                record.path = file_path
                record.save()

            self._save_description_to_db(image_hash, final_emotion, "emoji")
        except Exception as e:
            logger.error(f"保存表情包记录失败: {e}")

    def _build_chat_context_for_image(
        self,
        chat_id: Optional[str],
        message_time: Optional[float],
    ) -> str:
        """为图片解析构建简短的聊天上下文"""
        if not chat_id or message_time is None:
            return ""
        try:
            from src.chat.utils.chat_message_builder import (
                get_raw_msg_before_timestamp_with_chat,
                build_readable_messages,
            )

            messages = get_raw_msg_before_timestamp_with_chat(
                chat_id=chat_id,
                timestamp=message_time,
                limit=int(global_config.chat.max_context_size * 0.3),
            )
            if not messages:
                return ""

            chat_context = build_readable_messages(
                messages=messages,
                replace_bot_name=True,
                timestamp_mode="normal_no_YMD",
                read_mark=0.0,
                truncate=True,
                show_actions=False,
                show_pic=True,
                show_pic_mapping_header=False,
            )
            return chat_context
        except Exception as e:
            logger.error(f"构建图片上下文失败: {e}")
            return ""

    async def get_image_description(
        self,
        image_base64: str,
        chat_id: Optional[str] = None,
        message_time: Optional[float] = None,
    ) -> str:
        """获取图片描述"""
        try:
            image_bytes, image_hash, image_format = self._decode_image(image_base64)

            # 1. 检查 Images 表 (包括增加计数)
            existing_image = Images.get_or_none(Images.emoji_hash == image_hash)
            if existing_image:
                existing_image.count = (existing_image.count or 0) + 1
                existing_image.save()
                if existing_image.description:
                    logger.debug(f"[缓存命中] Images表: {existing_image.description[:50]}...")
                    return f"[图片：{existing_image.description}]"

            # 2. 检查 Cache 表
            if cached := self._get_description_from_db(image_hash, "image"):
                logger.debug(f"[缓存命中] ImageDescriptions表: {cached[:50]}...")
                # 如果 Images 表里有记录但没描述，顺便更新一下
                if existing_image:
                    existing_image.description = cached
                    existing_image.vlm_processed = True
                    existing_image.save()
                return f"[图片：{cached}]"

            # 3. VLM 生成
            prompt = PROMPTS["image"]
            if chat_id and message_time:
                chat_context = self._build_chat_context_for_image(chat_id, message_time)
                if chat_context:
                    prompt = (
                        prompt
                        + "\n\n以下是与这张图片相关的聊天上下文，仅用于帮助你理解图片含义，不要逐句复述：\n"
                        + chat_context
                    )

            prompt += f"\n\n{global_config.personality.visual_style}"
            logger.info(f"[VLM调用] 生成图片描述 (Hash: {image_hash[:8]}...)")

            description, _ = await self.vlm.generate_response_for_image(
                prompt, image_base64, image_format, temperature=0.4
            )

            if not description:
                logger.warning("AI未能生成图片描述")
                return "[图片(描述生成失败)]"

            # 4. 保存结果
            self._save_image_record(existing_image, image_hash, image_bytes, image_format, description)
            
            return f"[图片：{description}]"

        except Exception as e:
            logger.error(f"获取图片描述失败: {e}")
            return "[图片(处理失败)]"

    def _save_image_record(self, existing_image, image_hash, image_bytes, image_format, description):
        """保存图片记录"""
        try:
            current_time = time.time()
            filename = f"{int(current_time)}_{image_hash[:8]}.{image_format}"
            image_dir = os.path.join(self.IMAGE_DIR, "image")
            os.makedirs(image_dir, exist_ok=True)
            file_path = os.path.join(image_dir, filename)

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            if existing_image:
                existing_image.path = file_path
                existing_image.description = description
                existing_image.timestamp = current_time
                existing_image.vlm_processed = True
                if not existing_image.image_id:
                    existing_image.image_id = str(uuid.uuid4())
                existing_image.save()
            else:
                Images.create(
                    image_id=str(uuid.uuid4()),
                    emoji_hash=image_hash,
                    path=file_path,
                    type="image",
                    description=description,
                    timestamp=current_time,
                    vlm_processed=True,
                    count=1,
                )
            
            self._save_description_to_db(image_hash, description, "image")
            
        except Exception as e:
            logger.error(f"保存图片记录失败: {e}")

    async def process_image(
        self,
        image_base64: str,
        chat_id: Optional[str] = None,
        message_time: Optional[float] = None,
    ) -> Tuple[str, str]:
        """处理图片并返回图片ID和描述占位符"""
        try:
            _, image_hash, _ = self._decode_image(image_base64)
            
            # 检查现有记录
            if existing_image := Images.get_or_none(Images.emoji_hash == image_hash):
                if not existing_image.image_id:
                    existing_image.image_id = str(uuid.uuid4())
                    existing_image.save()
                
                existing_image.count = (existing_image.count or 0) + 1
                existing_image.save()
                return existing_image.image_id, f"[picid:{existing_image.image_id}]"

            # 新图片
            image_id = str(uuid.uuid4())
            
            # 先创建占位记录以返回 ID
            try:
                image_bytes = base64.b64decode(image_base64)
                filename = f"{image_id}.png"
                file_path = os.path.join(self.IMAGE_DIR, "images", filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                Images.create(
                    image_id=image_id,
                    emoji_hash=image_hash,
                    path=file_path,
                    type="image",
                    timestamp=time.time(),
                    vlm_processed=False,
                    count=1,
                )
            except Exception as e:
                logger.error(f"创建图片初始记录失败: {e}")

            # 启动异步处理
            await self._process_image_with_vlm(
                image_id,
                image_base64,
                chat_id=chat_id,
                message_time=message_time,
            )
            return image_id, f"[picid:{image_id}]"

        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            return "", "[图片]"

    async def _process_image_with_vlm(
        self,
        image_id: str,
        image_base64: str,
        chat_id: Optional[str] = None,
        message_time: Optional[float] = None,
    ) -> None:
        """使用VLM处理图片并更新数据库"""
        try:
            description_text = await self.get_image_description(
                image_base64,
                chat_id=chat_id,
                message_time=message_time,
            )

            description = ""
            if description_text.startswith("[图片：") and description_text.endswith("]"):
                description = description_text[4:-1]

            if description:
                Images.update(description=description, vlm_processed=True).where(Images.image_id == image_id).execute()

        except Exception as e:
            logger.error(f"VLM后台处理图片失败: {e}")

    @staticmethod
    def transform_gif(gif_base64: str, similarity_threshold: float = 1000.0, max_frames: int = 15) -> Optional[str]:
        """将GIF转换为水平拼接的静态图像"""
        try:
            if isinstance(gif_base64, str):
                gif_base64 = gif_base64.encode("ascii", errors="ignore").decode("ascii")
            
            gif = Image.open(io.BytesIO(base64.b64decode(gif_base64)))
            
            all_frames = []
            try:
                while True:
                    gif.seek(len(all_frames))
                    all_frames.append(gif.convert("RGB").copy())
            except EOFError:
                pass

            if not all_frames:
                return None

            # 帧选择
            selected_frames = []
            last_frame_np = None
            
            for i, frame in enumerate(all_frames):
                frame_np = np.array(frame)
                if i == 0:
                    selected_frames.append(frame)
                    last_frame_np = frame_np
                    continue

                if last_frame_np is not None:
                    mse = np.mean((frame_np - last_frame_np) ** 2)
                    if mse > similarity_threshold:
                        selected_frames.append(frame)
                        last_frame_np = frame_np
                        if len(selected_frames) >= max_frames:
                            break

            if not selected_frames:
                return None

            # 拼接
            target_height = 200
            frame_width, frame_height = selected_frames[0].size
            if frame_height == 0: return None
            
            target_width = int((target_height / frame_height) * frame_width) or 1
            
            resized_frames = [f.resize((target_width, target_height), Image.Resampling.LANCZOS) for f in selected_frames]
            total_width = target_width * len(resized_frames)
            
            if total_width == 0: return None
            
            combined = Image.new("RGB", (total_width, target_height))
            for idx, frame in enumerate(resized_frames):
                combined.paste(frame, (idx * target_width, 0))

            buffer = io.BytesIO()
            combined.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"GIF转换失败: {e}")
            return None


image_manager: Optional[ImageManager] = None


def get_image_manager() -> ImageManager:
    """获取全局图片管理器单例"""
    global image_manager
    if image_manager is None:
        image_manager = ImageManager()
    return image_manager


def image_path_to_base64(image_path: str) -> str:
    """将图片路径转换为base64编码
    Args:
        image_path: 图片文件路径
    Returns:
        str: base64编码的图片数据
    Raises:
        FileNotFoundError: 当图片文件不存在时
        IOError: 当读取图片文件失败时
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    with open(image_path, "rb") as f:
        if image_data := f.read():
            return base64.b64encode(image_data).decode("utf-8")
        else:
            raise IOError(f"读取图片文件失败: {image_path}")


def base64_to_image(image_base64: str, output_path: str) -> bool:
    """将base64编码的图片保存为文件

    Args:
        image_base64: 图片的base64编码
        output_path: 输出文件路径

    Returns:
        bool: 是否成功保存

    Raises:
        ValueError: 当base64编码无效时
        IOError: 当保存文件失败时
    """
    try:
        # 确保base64字符串只包含ASCII字符
        if isinstance(image_base64, str):
            image_base64 = image_base64.encode("ascii", errors="ignore").decode("ascii")

        # 解码base64
        image_bytes = base64.b64decode(image_base64)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存文件
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        return True

    except Exception as e:
        logger.error(f"保存base64图片失败: {e}")
        return False
