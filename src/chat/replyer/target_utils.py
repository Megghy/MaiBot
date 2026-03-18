import re
from typing import Optional, Tuple

from src.chat.utils.chat_message_builder import replace_user_references
from src.plugin_system.apis.message_api import translate_pid_to_description


def parse_reply_target(target_message: Optional[str]) -> Tuple[str, str]:
    sender = ""
    target = ""
    if target_message is None:
        return sender, target
    if ":" in target_message or "：" in target_message:
        parts = re.split(pattern=r"[:：]", string=target_message, maxsplit=1)
        if len(parts) == 2:
            sender = parts[0].strip()
            target = parts[1].strip()
    return sender, target


def replace_picids_with_descriptions(text: str) -> str:
    pic_pattern = r"\[picid:([^\]]+)\]"

    def replace_pic_id(match: re.Match) -> str:
        pic_id = match.group(1)
        description = translate_pid_to_description(pic_id)
        return f"[图片：{description}]"

    return re.sub(pic_pattern, replace_pic_id, text)


def analyze_target_content(target: str) -> Tuple[bool, bool, str, str]:
    if not target or not target.strip():
        return False, False, "", ""

    picid_pattern = r"\[picid:[^\]]+\]"
    picid_matches = re.findall(picid_pattern, target)
    text_without_picids = re.sub(picid_pattern, "", target).strip()

    has_only_pics = len(picid_matches) > 0 and not text_without_picids
    has_text = bool(text_without_picids)

    pic_part = ""
    if picid_matches:
        pic_descriptions = []
        for picid_match in picid_matches:
            pic_id = picid_match[7:-1]
            description = translate_pid_to_description(pic_id)
            if description == "[图片]":
                pic_descriptions.append(description)
            else:
                pic_descriptions.append(f"[图片:{description}]")
        pic_part = "".join(pic_descriptions)

    return has_only_pics, has_text, pic_part, text_without_picids


def process_target_content(target: Optional[str], platform: str) -> Tuple[str, bool, bool, str, str]:
    normalized_target = replace_user_references(target or "", platform, replace_bot_name=True)
    has_only_pics, has_text, pic_part, text_part = analyze_target_content(normalized_target)
    normalized_target = replace_picids_with_descriptions(normalized_target)
    return normalized_target, has_only_pics, has_text, pic_part, text_part


def build_reply_target_block_text(
    *,
    is_group_chat: bool,
    sender: Optional[str],
    target: str,
    has_only_pics: bool,
    has_text: bool,
    pic_part: str,
    text_part: str,
) -> str:
    sender = sender or ""
    if is_group_chat:
        if not sender:
            return ""
        if has_only_pics and not has_text:
            return f"现在{sender}发送的图片：{pic_part}。引起了你的注意"
        if has_text and pic_part:
            return f"现在{sender}发送了图片：{pic_part}，并说：{text_part}。引起了你的注意"
        if has_text:
            return f"现在{sender}说的：{text_part}。引起了你的注意"
        return f"现在{sender}说的：{target}。引起了你的注意"

    display_sender = sender or "对方"
    if has_only_pics and not has_text:
        return f"现在{display_sender}发送的图片：{pic_part}。引起了你的注意"
    if has_text and pic_part:
        return f"现在{display_sender}发送了图片：{pic_part}，并说：{text_part}。引起了你的注意"
    if has_text:
        return f"现在{display_sender}说的：{text_part}。引起了你的注意"
    if target:
        return f"现在{display_sender}说的：{target}。引起了你的注意"
    return ""
