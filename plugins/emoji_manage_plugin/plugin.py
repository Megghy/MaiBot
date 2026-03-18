from typing import List, Tuple, Type
from src.plugin_system.apis import emoji_api
from src.plugin_system.base import (
    BasePlugin,
    BaseCommand,
    ComponentInfo,
    ConfigField,
    ReplyContentType,
)
from maim_message import Seg
from src.common.logger import get_logger

logger = get_logger("emoji_manage_plugin")


class AddEmojiCommand(BaseCommand):
    command_name = "add_emoji"
    command_description = "添加表情包"
    command_pattern = r".*/emoji add.*"

    async def execute(self) -> Tuple[bool, str, bool]:
        # 查找消息中的表情包
        # logger.info(f"查找消息中的表情包: {self.message.message_segment}")

        emoji_base64_list = self.find_and_return_emoji_in_message(self.message.message_segment)

        if not emoji_base64_list:
            return False, "未在消息中找到表情包或图片", False

        # 注册找到的表情包
        success_count = 0
        fail_count = 0
        results = []

        for i, emoji_base64 in enumerate(emoji_base64_list):
            try:
                # 使用emoji_api注册表情包（让API自动生成唯一文件名）
                result = await emoji_api.register_emoji(emoji_base64)

                if result["success"]:
                    success_count += 1
                    description = result.get("description", "未知描述")
                    emotions = result.get("emotions", [])
                    replaced = result.get("replaced", False)

                    result_msg = f"表情包 {i + 1} 注册成功{'(替换旧表情包)' if replaced else '(新增表情包)'}"
                    if description:
                        result_msg += f"\n描述: {description}"
                    if emotions:
                        result_msg += f"\n情感标签: {', '.join(emotions)}"

                    results.append(result_msg)
                else:
                    fail_count += 1
                    error_msg = result.get("message", "注册失败")
                    results.append(f"表情包 {i + 1} 注册失败: {error_msg}")

            except Exception as e:
                fail_count += 1
                results.append(f"表情包 {i + 1} 注册时发生错误: {str(e)}")

        # 构建返回消息
        total_count = success_count + fail_count
        summary_msg = f"表情包注册完成: 成功 {success_count} 个，失败 {fail_count} 个，共处理 {total_count} 个"

        # 如果有结果详情，添加到返回消息中
        details_msg = ""
        if results:
            details_msg = "\n" + "\n".join(results)
            final_msg = summary_msg + details_msg
        else:
            final_msg = summary_msg

        # 使用表达器重写回复
        try:
            from src.plugin_system.apis import generator_api

            # 构建重写数据
            rewrite_data = {
                "raw_reply": summary_msg,
                "reason": f"注册了表情包：{details_msg}\n",
            }

            # 调用表达器重写
            result_status, data = await generator_api.rewrite_reply(
                chat_stream=self.message.chat_stream,
                reply_data=rewrite_data,
            )

            if result_status:
                # 发送重写后的回复
                for reply_seg in data.reply_set.reply_data:
                    send_data = reply_seg.content
                    await self.send_text(send_data)

                return success_count > 0, final_msg, success_count > 0
            else:
                # 如果重写失败，发送原始消息
                await self.send_text(final_msg)
                return success_count > 0, final_msg, success_count > 0

        except Exception as e:
            # 如果表达器调用失败，发送原始消息
            logger.error(f"[add_emoji] 表达器重写失败: {e}")
            await self.send_text(final_msg)
            return success_count > 0, final_msg, success_count > 0

    def find_and_return_emoji_in_message(self, message_segments) -> List[str]:
        emoji_base64_list = []

        # 处理单个Seg对象的情况
        if isinstance(message_segments, Seg):
            if message_segments.type == "emoji":
                emoji_base64_list.append(message_segments.data)
            elif message_segments.type == "image":
                # 假设图片数据是base64编码的
                emoji_base64_list.append(message_segments.data)
            elif message_segments.type == "seglist":
                # 递归处理嵌套的Seg列表
                emoji_base64_list.extend(self.find_and_return_emoji_in_message(message_segments.data))
            return emoji_base64_list

        # 处理Seg列表的情况
        for seg in message_segments:
            if seg.type == "emoji":
                emoji_base64_list.append(seg.data)
            elif seg.type == "image":
                # 假设图片数据是base64编码的
                emoji_base64_list.append(seg.data)
            elif seg.type == "seglist":
                # 递归处理嵌套的Seg列表
                emoji_base64_list.extend(self.find_and_return_emoji_in_message(seg.data))
        return emoji_base64_list


class ListEmojiCommand(BaseCommand):
    """列表表情包Command - 响应/emoji list命令"""

    command_name = "emoji_list"
    command_description = "列表表情包"

    # === 命令设置（必须填写）===
    command_pattern = r"^/emoji list(\s+\d+)?$"  # 匹配 "/emoji list" 或 "/emoji list 数量"

    async def execute(self) -> Tuple[bool, str, bool]:
        """执行列表表情包"""
        from src.plugin_system.apis import emoji_api
        import datetime

        # 解析命令参数
        import re

        match = re.match(r"^/emoji list(?:\s+(\d+))?$", self.message.raw_message)
        max_count = 10  # 默认显示10个
        if match and match.group(1):
            max_count = min(int(match.group(1)), 50)  # 最多显示50个

        # 获取当前时间
        time_format: str = self.get_config("time.format", "%Y-%m-%d %H:%M:%S")  # type: ignore
        now = datetime.datetime.now()
        time_str = now.strftime(time_format)

        # 获取表情包信息
        emoji_count = emoji_api.get_count()
        emoji_info = emoji_api.get_info()

        # 构建返回消息
        message_lines = [
            f"📊 表情包统计信息 ({time_str})",
            f"• 总数: {emoji_count} / {emoji_info['max_count']}",
            f"• 可用: {emoji_info['available_emojis']}",
        ]

        if emoji_count == 0:
            message_lines.append("\n❌ 暂无表情包")
            final_message = "\n".join(message_lines)
            await self.send_text(final_message)
            return True, final_message, True

        # 获取所有表情包
        all_emojis = await emoji_api.get_all()
        if not all_emojis:
            message_lines.append("\n❌ 无法获取表情包列表")
            final_message = "\n".join(message_lines)
            await self.send_text(final_message)
            return False, final_message, True

        # 显示前N个表情包
        display_emojis = all_emojis[:max_count]
        message_lines.append(f"\n📋 显示前 {len(display_emojis)} 个表情包:")

        for i, (_, description, emotion) in enumerate(display_emojis, 1):
            # 截断过长的描述
            short_desc = description[:50] + "..." if len(description) > 50 else description
            message_lines.append(f"{i}. {short_desc} [{emotion}]")

        # 如果还有更多表情包，显示总数
        if len(all_emojis) > max_count:
            message_lines.append(f"\n💡 还有 {len(all_emojis) - max_count} 个表情包未显示")

        final_message = "\n".join(message_lines)

        # 直接发送文本消息
        await self.send_text(final_message)

        return True, final_message, True


class DeleteEmojiCommand(BaseCommand):
    command_name = "delete_emoji"
    command_description = "删除表情包"
    command_pattern = r".*/emoji delete.*"

    async def execute(self) -> Tuple[bool, str, bool]:
        # 查找消息中的表情包图片
        logger.info(f"查找消息中的表情包用于删除: {self.message.message_segment}")

        emoji_base64_list = self.find_and_return_emoji_in_message(self.message.message_segment)

        if not emoji_base64_list:
            return False, "未在消息中找到表情包或图片", False

        # 删除找到的表情包
        success_count = 0
        fail_count = 0
        results = []

        for i, emoji_base64 in enumerate(emoji_base64_list):
            try:
                # 计算图片的哈希值来查找对应的表情包
                import base64
                import hashlib

                # 确保base64字符串只包含ASCII字符
                if isinstance(emoji_base64, str):
                    emoji_base64_clean = emoji_base64.encode("ascii", errors="ignore").decode("ascii")
                else:
                    emoji_base64_clean = str(emoji_base64)

                # 计算哈希值
                image_bytes = base64.b64decode(emoji_base64_clean)
                emoji_hash = hashlib.md5(image_bytes).hexdigest()

                # 使用emoji_api删除表情包
                result = await emoji_api.delete_emoji(emoji_hash)

                if result["success"]:
                    success_count += 1
                    description = result.get("description", "未知描述")
                    count_before = result.get("count_before", 0)
                    count_after = result.get("count_after", 0)
                    emotions = result.get("emotions", [])

                    result_msg = f"表情包 {i + 1} 删除成功"
                    if description:
                        result_msg += f"\n描述: {description}"
                    if emotions:
                        result_msg += f"\n情感标签: {', '.join(emotions)}"
                    result_msg += f"\n表情包数量: {count_before} → {count_after}"

                    results.append(result_msg)
                else:
                    fail_count += 1
                    error_msg = result.get("message", "删除失败")
                    results.append(f"表情包 {i + 1} 删除失败: {error_msg}")

            except Exception as e:
                fail_count += 1
                results.append(f"表情包 {i + 1} 删除时发生错误: {str(e)}")

        # 构建返回消息
        total_count = success_count + fail_count
        summary_msg = f"表情包删除完成: 成功 {success_count} 个，失败 {fail_count} 个，共处理 {total_count} 个"

        # 如果有结果详情，添加到返回消息中
        details_msg = ""
        if results:
            details_msg = "\n" + "\n".join(results)
            final_msg = summary_msg + details_msg
        else:
            final_msg = summary_msg

        # 使用表达器重写回复
        try:
            from src.plugin_system.apis import generator_api

            # 构建重写数据
            rewrite_data = {
                "raw_reply": summary_msg,
                "reason": f"删除了表情包：{details_msg}\n",
            }

            # 调用表达器重写
            result_status, data = await generator_api.rewrite_reply(
                chat_stream=self.message.chat_stream,
                reply_data=rewrite_data,
            )

            if result_status:
                # 发送重写后的回复
                for reply_seg in data.reply_set.reply_data:
                    send_data = reply_seg.content
                    await self.send_text(send_data)

                return success_count > 0, final_msg, success_count > 0
            else:
                # 如果重写失败，发送原始消息
                await self.send_text(final_msg)
                return success_count > 0, final_msg, success_count > 0

        except Exception as e:
            # 如果表达器调用失败，发送原始消息
            logger.error(f"[delete_emoji] 表达器重写失败: {e}")
            await self.send_text(final_msg)
            return success_count > 0, final_msg, success_count > 0

    def find_and_return_emoji_in_message(self, message_segments) -> List[str]:
        emoji_base64_list = []

        # 处理单个Seg对象的情况
        if isinstance(message_segments, Seg):
            if message_segments.type == "emoji":
                emoji_base64_list.append(message_segments.data)
            elif message_segments.type == "image":
                # 假设图片数据是base64编码的
                emoji_base64_list.append(message_segments.data)
            elif message_segments.type == "seglist":
                # 递归处理嵌套的Seg列表
                emoji_base64_list.extend(self.find_and_return_emoji_in_message(message_segments.data))
            return emoji_base64_list

        # 处理Seg列表的情况
        for seg in message_segments:
            if seg.type == "emoji":
                emoji_base64_list.append(seg.data)
            elif seg.type == "image":
                # 假设图片数据是base64编码的
                emoji_base64_list.append(seg.data)
            elif seg.type == "seglist":
                # 递归处理嵌套的Seg列表
                emoji_base64_list.extend(self.find_and_return_emoji_in_message(seg.data))
        return emoji_base64_list


class RandomEmojis(BaseCommand):
    command_name = "random_emojis"
    command_description = "发送多张随机表情包"
    command_pattern = r"^/random_emojis$"

    async def execute(self):
        emojis = await emoji_api.get_random(5)
        if not emojis:
            return False, "未找到表情包", False
        emoji_base64_list = []
        for emoji in emojis:
            emoji_base64_list.append(emoji[0])
        return await self.forward_images(emoji_base64_list)

    async def forward_images(self, images: List[str]):
        """
        把多张图片用合并转发的方式发给用户
        """
        success = await self.send_forward([("0", "神秘用户", [(ReplyContentType.IMAGE, img)]) for img in images])
        return (True, "已发送随机表情包", True) if success else (False, "发送随机表情包失败", False)


# ===== 插件注册 =====


class EmojiManagePlugin(BasePlugin):
    """表情包管理插件 - 管理表情包"""

    # 插件基本信息
    plugin_name: str = "emoji_manage_plugin"  # 内部标识符
    enable_plugin: bool = False
    dependencies: List[str] = []  # 插件依赖列表
    python_dependencies: List[str] = []  # Python包依赖列表
    config_file_name: str = "config.toml"  # 配置文件名

    # 配置节描述
    config_section_descriptions = {"plugin": "插件基本信息", "emoji": "表情包功能配置"}

    # 配置Schema定义
    config_schema: dict = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
            "config_version": ConfigField(type=str, default="1.0.1", description="配置文件版本"),
        },
    }

    def get_plugin_components(self) -> List[Tuple[ComponentInfo, Type]]:
        return [
            (RandomEmojis.get_command_info(), RandomEmojis),
            (AddEmojiCommand.get_command_info(), AddEmojiCommand),
            (ListEmojiCommand.get_command_info(), ListEmojiCommand),
            (DeleteEmojiCommand.get_command_info(), DeleteEmojiCommand),
        ]


PLUGIN_CLASS = EmojiManagePlugin
