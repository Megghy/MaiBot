from math import ceil
from pathlib import Path
from typing import Optional, Union

import base64
import io

from PIL import Image as PILImage, ImageOps as PILImageOps, ImageSequence
import numpy as np

from src.common.logger import get_logger

logger = get_logger("image_utils")

MODEL_MIN_IMAGE_SIDE = 64
MODEL_MAX_UPSCALED_IMAGE_SIDE = 2048


class ImageUtils:
    @staticmethod
    def normalize_image_base64_for_model(
        image_base64: str,
        image_format: str,
        *,
        min_side: int = MODEL_MIN_IMAGE_SIDE,
        max_upscaled_side: int = MODEL_MAX_UPSCALED_IMAGE_SIDE,
    ) -> tuple[str, str, bool]:
        """确保发给视觉模型的图片不低于常见最小识别尺寸。"""
        if min_side <= 0:
            raise ValueError("模型图片最小边长必须大于0")

        image_bytes = base64.b64decode(image_base64, validate=True)
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            normalized_image = PILImageOps.exif_transpose(image)
            width, height = normalized_image.size
            if width <= 0 or height <= 0:
                raise ValueError("图片尺寸无效，无法发送给视觉模型")
            if width >= min_side and height >= min_side:
                return image_base64, image_format, False

            if normalized_image.mode in ("RGBA", "LA") or (
                normalized_image.mode == "P" and "transparency" in normalized_image.info
            ):
                working_image = normalized_image.convert("RGBA")
                canvas_mode = "RGBA"
                background_color = (255, 255, 255, 0)
            else:
                working_image = normalized_image.convert("RGB")
                canvas_mode = "RGB"
                background_color = (255, 255, 255)

            scale = max(1, ceil(min_side / min(width, height)))
            if max_upscaled_side > 0:
                max_scale = max(1, max_upscaled_side // max(width, height))
                scale = min(scale, max_scale)

            resized_width = max(1, width * scale)
            resized_height = max(1, height * scale)
            resized_image = working_image.resize((resized_width, resized_height), PILImage.Resampling.NEAREST)

            canvas_width = max(min_side, resized_width)
            canvas_height = max(min_side, resized_height)
            if (canvas_width, canvas_height) != resized_image.size:
                canvas = PILImage.new(canvas_mode, (canvas_width, canvas_height), background_color)
                paste_box = ((canvas_width - resized_width) // 2, (canvas_height - resized_height) // 2)
                if resized_image.mode == "RGBA":
                    canvas.paste(resized_image, paste_box, resized_image)
                else:
                    canvas.paste(resized_image, paste_box)
                resized_image = canvas

            output_buffer = io.BytesIO()
            resized_image.save(output_buffer, format="PNG")
            resized_base64 = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
            return resized_base64, "png", True

    @staticmethod
    def animated_image_to_static_image(
        image_bytes: bytes,
        similarity_threshold: float = 1000.0,
        max_frames: int = 15,
        target_height: int = 200,
    ) -> bytes:
        """将动态图（GIF/动态WebP等）关键帧水平拼接为静态 JPEG 图像。"""
        with PILImage.open(io.BytesIO(image_bytes)) as anim_image:
            selected_frames: list[PILImage.Image] = []
            last_selected_frame_np: Optional[np.ndarray] = None
            frame_index = 0

            for frame in ImageSequence.Iterator(anim_image):
                frame_rgb = ImageUtils._prepare_image_for_receive_compression(frame)
                # float32 避免 uint8 减法下溢和平方溢出（uint8 下 MSE 恒 <= 255）
                frame_np = np.asarray(frame_rgb, dtype=np.float32)

                if frame_index == 0:
                    selected_frames.append(frame_rgb.copy())
                    last_selected_frame_np = frame_np
                elif last_selected_frame_np is not None:
                    mse = float(np.mean((frame_np - last_selected_frame_np) ** 2))
                    if mse > similarity_threshold:
                        selected_frames.append(frame_rgb.copy())
                        last_selected_frame_np = frame_np
                        if len(selected_frames) >= max_frames:
                            frame_index += 1
                            break
                frame_index += 1

            # 局部微动动图（如仅文字/小区域变化导致帧间整体 MSE 未达阈值）降级为均匀抽样
            if len(selected_frames) == 1 and frame_index > 1 and max_frames > 1 and last_selected_frame_np is not None:
                fallback_count = min(max_frames, 5, frame_index)
                sample_indices = {
                    int(i * (frame_index - 1) / (fallback_count - 1))
                    for i in range(1, fallback_count)
                }
                for idx, frame in enumerate(ImageSequence.Iterator(anim_image)):
                    if idx not in sample_indices:
                        continue
                    frame_rgb = ImageUtils._prepare_image_for_receive_compression(frame)
                    frame_np = np.asarray(frame_rgb, dtype=np.float32)
                    if float(np.mean((frame_np - last_selected_frame_np) ** 2)) > 20.0:
                        selected_frames.append(frame_rgb.copy())
                        last_selected_frame_np = frame_np
                        if len(selected_frames) >= max_frames:
                            break

        if not selected_frames:
            logger.error("未能抽取到任何有效帧")
            raise ValueError("未能抽取到任何有效帧")

        frame_width, frame_height = selected_frames[0].size
        if frame_height == 0:
            raise ValueError("帧高度为0，无法计算缩放尺寸")

        target_width = int((target_height / frame_height) * frame_width)
        if target_width == 0:
            logger.warning(f"计算出的目标宽度为0 (原始尺寸 {frame_width}x{frame_height})，调整为1")
            target_width = 1
        resized_frames = [
            frame.resize((target_width, target_height), PILImage.Resampling.LANCZOS) for frame in selected_frames
        ]

        total_width = target_width * len(resized_frames)
        combined_image = PILImage.new("RGB", (total_width, target_height))
        for idx, frame in enumerate(resized_frames):
            combined_image.paste(frame, (idx * target_width, 0))
        buffer = io.BytesIO()
        combined_image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    @staticmethod
    def prepare_image_for_vlm(
        image_bytes: bytes,
        image_format: str,
        *,
        similarity_threshold: float = 1000.0,
        max_frames: int = 15,
        target_height: int = 200,
    ) -> tuple[bytes, str, bool]:
        """为视觉模型准备图片：GIF 与多帧动图统一转为横向关键帧拼接 JPG。"""
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            detected_format = (img.format or image_format or "").lower()
            is_animated = bool(getattr(img, "is_animated", False) or getattr(img, "n_frames", 1) > 1)
        if is_animated or detected_format == "gif":
            converted = ImageUtils.animated_image_to_static_image(
                image_bytes,
                similarity_threshold=similarity_threshold,
                max_frames=max_frames,
                target_height=target_height,
            )
            return converted, "jpg", is_animated
        return image_bytes, image_format, False

    @staticmethod
    def gif_2_static_image(gif_bytes: bytes, similarity_threshold: float = 1000.0, max_frames: int = 15) -> bytes:
        """
        将GIF图片水平拼接为静态图像，跳过相似帧

        Args:
            gif_bytes (bytes): 输入的GIF图片字节数据
            similarity_threshold (float): 判定帧相似的阈值 (MSE)，越小表示要求差异越大才算不同帧，默认1000.0
            max_frames (int): 最大抽取的帧数，默认15
        Returns:
            bytes: 拼接后的静态图像字节数据，格式为JPEG
        Raises:
            ValueError: 如果输入的GIF无效或无法处理
            MemoryError: 如果处理过程中内存不足
            Exception: 其他异常
        """
        with PILImage.open(io.BytesIO(gif_bytes)) as gif_image:
            if not gif_image.format or gif_image.format.lower() != "gif":
                logger.error("输入的图片不是有效的GIF格式")
                raise ValueError("输入的图片不是有效的GIF格式")
        return ImageUtils.animated_image_to_static_image(
            gif_bytes,
            similarity_threshold=similarity_threshold,
            max_frames=max_frames,
        )

    @staticmethod
    def compress_image_to_size(image_bytes: bytes, target_size: int) -> bytes:
        """将图片压缩到目标大小以内，失败时保持原图数据。"""
        if not image_bytes:
            raise ValueError("输入的图片字节数据无效")
        if target_size <= 0 or len(image_bytes) <= target_size:
            return image_bytes

        try:
            with PILImage.open(io.BytesIO(image_bytes)) as image:
                image.seek(0)
                working_image = ImageUtils._prepare_image_for_receive_compression(image)
        except Exception as exc:
            logger.warning(f"接收图片压缩失败，无法识别图片格式: {exc}")
            return image_bytes

        compressed = ImageUtils._compress_static_image_to_size(working_image, target_size)
        if len(compressed) < len(image_bytes):
            return compressed
        return image_bytes

    @staticmethod
    def _prepare_image_for_receive_compression(image: PILImage.Image) -> PILImage.Image:
        """将任意图片整理成适合接收链路压缩的 RGB 静态图。"""
        normalized_image = PILImageOps.exif_transpose(image)
        if normalized_image.mode in ("RGBA", "LA") or (
            normalized_image.mode == "P" and "transparency" in normalized_image.info
        ):
            alpha_image = normalized_image.convert("RGBA")
            background = PILImage.new("RGB", alpha_image.size, (255, 255, 255))
            background.paste(alpha_image, mask=alpha_image.getchannel("A"))
            return background
        return normalized_image.convert("RGB")

    @staticmethod
    def _compress_static_image_to_size(image: PILImage.Image, target_size: int) -> bytes:
        """通过降低质量和缩放尺寸压缩静态图片。"""
        working_image = image.copy()
        quality = 85
        last_output = b""

        for _ in range(16):
            output_buffer = io.BytesIO()
            working_image.save(output_buffer, format="JPEG", quality=quality, optimize=True)
            output_bytes = output_buffer.getvalue()
            last_output = output_bytes
            if len(output_bytes) <= target_size:
                return output_bytes

            if quality > 55:
                quality = max(55, quality - 10)
                continue

            scale = max(0.1, min(0.95, (target_size / len(output_bytes)) ** 0.5 * 0.95))
            new_width = max(1, int(working_image.width * scale))
            new_height = max(1, int(working_image.height * scale))
            if (new_width, new_height) == working_image.size:
                break
            working_image = working_image.resize((new_width, new_height), PILImage.Resampling.LANCZOS)

        return last_output

    @staticmethod
    def image_bytes_to_base64(image_bytes: bytes) -> str:
        """
        将图片字节数据转换为Base64编码字符串

        Args:
            image_bytes (bytes): 输入的图片字节数据
        Returns:
            str: Base64编码的图片字符串
        Raises:
            ValueError: 如果输入的图片字节数据无效
        """
        if not image_bytes:
            logger.error("输入的图片字节数据无效")
            raise ValueError("输入的图片字节数据无效")
        return base64.b64encode(image_bytes).decode("utf-8")

    @staticmethod
    def image_path_to_base64(image_path: Union[str, Path]) -> Optional[str]:
        """读取图片文件并转换为 Base64 编码字符串"""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"图片文件不存在: {path}")
                return None
            image_bytes = path.read_bytes()
            return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"读取图片文件失败: {e}")
            return None

    @staticmethod
    def base64_to_image(base64_str: str, save_path: Union[str, Path]) -> bool:
        """将 Base64 编码字符串解码并保存为图片文件"""
        try:
            image_bytes = base64.b64decode(base64_str)
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_bytes)
            return True
        except Exception as e:
            logger.error(f"保存图片文件失败: {e}")
            return False
