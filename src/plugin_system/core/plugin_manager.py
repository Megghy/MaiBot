from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from inspect import getfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional, Tuple, Type

from src.common.logger import get_logger
from src.plugin_system.base.plugin_base import PluginBase
from src.plugin_system.utils.manifest_utils import VersionComparator

from .component_registry import component_registry

logger = get_logger("plugin_manager")


class PluginManager:
    """插件加载器。

    活动注册数据已经进入 runtime registries，这里只保留：
    - 插件类发现与实例化
    - 外部插件目录扫描
    - 运行期加载/卸载/重载接口

    外部插件约定：
    - 每个插件包包含 `plugin.py`
    - `plugin.py` 必须显式导出 `PLUGIN_CLASS`
    """

    def __init__(self, plugin_directories: Iterable[str] | None = None) -> None:
        self.built_in_root = Path("src/plugins/built_in").resolve()
        self.plugin_directories: list[str] = []
        self.plugin_classes: dict[str, Type[PluginBase]] = {}
        self.plugin_paths: dict[str, str] = {}
        self.loaded_plugins: dict[str, PluginBase] = {}
        self.failed_plugins: dict[str, str] = {}

        self._ensure_plugin_directories(["plugins"] if plugin_directories is None else plugin_directories)
        logger.info("插件管理器初始化完成")

    def add_plugin_directory(self, directory: str) -> bool:
        resolved = str(Path(directory).resolve())
        if not Path(resolved).exists():
            logger.warning(f"插件目录不存在: {directory}")
            return False
        if resolved in self.plugin_directories:
            logger.warning(f"插件目录已存在: {resolved}")
            return False
        self.plugin_directories.append(resolved)
        logger.debug(f"已添加插件目录: {resolved}")
        return True

    def register_plugin_class(self, plugin_class: Type[PluginBase], *, plugin_dir: str | None = None) -> str:
        if not issubclass(plugin_class, PluginBase):
            raise TypeError(f"{plugin_class.__name__} 不是 PluginBase 的子类")

        plugin_name = plugin_class.plugin_name  # type: ignore[attr-defined]
        if not plugin_name:
            raise ValueError(f"插件类 {plugin_class.__name__} 缺少 plugin_name")
        if "." in plugin_name:
            raise ValueError(f"插件名称 '{plugin_name}' 包含非法字符 '.'，请使用下划线替代")

        resolved_dir = Path(plugin_dir or self._resolve_plugin_dir(plugin_class)).resolve()
        self.plugin_classes[plugin_name] = plugin_class
        self.plugin_paths[plugin_name] = str(resolved_dir)
        logger.debug(f"插件类已注册: {plugin_name}, 路径: {resolved_dir}")
        return plugin_name

    def load_builtin_plugins(self, plugin_classes: Iterable[Type[PluginBase]]) -> Tuple[int, int]:
        plugin_names = [self.register_plugin_class(plugin_class) for plugin_class in plugin_classes]
        return self._load_registered_plugins(plugin_names)

    def load_all_plugins(self) -> Tuple[int, int]:
        discovered, failed = self.rescan_plugin_directory()
        loaded, registration_failed = self._load_registered_plugins(self.list_registered_plugins())
        self._show_stats(
            discovered=discovered, import_failed=failed, loaded=loaded, registration_failed=registration_failed
        )
        return loaded, failed + registration_failed

    def load_registered_plugin_classes(self, plugin_name: str) -> Tuple[bool, int]:
        if plugin_name in self.loaded_plugins:
            return True, 0

        plugin_class = self.plugin_classes.get(plugin_name)
        if plugin_class is None:
            logger.error(f"插件 {plugin_name} 的插件类未注册")
            return False, 1

        plugin_dir = self.plugin_paths.get(plugin_name)
        if not plugin_dir:
            logger.error(f"插件 {plugin_name} 缺少插件目录")
            return False, 1

        try:
            plugin_instance = plugin_class(plugin_dir=plugin_dir)
            plugin_instance.plugin_info.is_built_in = self._is_built_in_path(plugin_dir)

            if not plugin_instance.enable_plugin:
                logger.info(f"插件 {plugin_name} 已禁用，跳过加载")
                return False, 0

            compatible, error_message = self._check_plugin_version_compatibility(
                plugin_name,
                plugin_instance.manifest_data,
            )
            if not compatible:
                self.failed_plugins[plugin_name] = error_message
                logger.error(f"插件加载失败: {plugin_name} - {error_message}")
                return False, 1

            if not plugin_instance.register_plugin():
                self.failed_plugins[plugin_name] = "插件注册失败"
                logger.error(f"插件注册失败: {plugin_name}")
                return False, 1

            self.loaded_plugins[plugin_name] = plugin_instance
            self.failed_plugins.pop(plugin_name, None)
            self._show_plugin_components(plugin_name)
            return True, 1
        except FileNotFoundError as exc:
            error_message = f"缺少 manifest 文件: {exc}"
        except ValueError as exc:
            error_message = f"manifest 验证失败: {exc}"
        except Exception as exc:  # pragma: no cover - 启动边界日志
            error_message = f"未知错误: {exc}"
            logger.exception(f"插件加载失败: {plugin_name}")

        self.failed_plugins[plugin_name] = error_message
        logger.error(f"插件加载失败: {plugin_name} - {error_message}")
        return False, 1

    async def remove_registered_plugin(self, plugin_name: str) -> bool:
        if not plugin_name:
            raise ValueError("插件名称不能为空")
        plugin_instance = self.loaded_plugins.get(plugin_name)
        if plugin_instance is None:
            logger.warning(f"插件 {plugin_name} 未加载")
            return False

        success = True
        for component in plugin_instance.plugin_info.components:
            success &= await component_registry.remove_component(component.name, component.component_type, plugin_name)
        success &= component_registry.remove_plugin_registry(plugin_name)
        del self.loaded_plugins[plugin_name]
        return success

    async def reload_registered_plugin(self, plugin_name: str) -> bool:
        if plugin_name in self.loaded_plugins and not await self.remove_registered_plugin(plugin_name):
            return False
        success, _ = self.load_registered_plugin_classes(plugin_name)
        return success

    def rescan_plugin_directory(self) -> Tuple[int, int]:
        discovered = 0
        failed = 0
        for directory in self.plugin_directories:
            current_discovered, current_failed = self._load_plugin_modules_from_directory(directory)
            discovered += current_discovered
            failed += current_failed
        return discovered, failed

    def get_plugin_instance(self, plugin_name: str) -> Optional[PluginBase]:
        return self.loaded_plugins.get(plugin_name)

    def list_loaded_plugins(self) -> list[str]:
        return list(self.loaded_plugins.keys())

    def list_registered_plugins(self) -> list[str]:
        return list(self.plugin_classes.keys())

    def get_plugin_path(self, plugin_name: str) -> Optional[str]:
        return self.plugin_paths.get(plugin_name)

    def reset(self, *, clear_registered_classes: bool = False) -> None:
        self.loaded_plugins.clear()
        self.failed_plugins.clear()
        if clear_registered_classes:
            self.plugin_classes.clear()
            self.plugin_paths.clear()

    def _ensure_plugin_directories(self, directories: Iterable[str]) -> None:
        for directory in directories:
            resolved = Path(directory).resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            resolved_str = str(resolved)
            if resolved_str not in self.plugin_directories:
                self.plugin_directories.append(resolved_str)

    def _load_registered_plugins(self, plugin_names: Iterable[str]) -> Tuple[int, int]:
        loaded = 0
        failed = 0
        for plugin_name in plugin_names:
            success, count = self.load_registered_plugin_classes(plugin_name)
            if success:
                loaded += count
            elif count:
                failed += count
        return loaded, failed

    def _load_plugin_modules_from_directory(self, directory: str) -> Tuple[int, int]:
        directory_path = Path(directory).resolve()
        if self._is_built_in_path(directory_path):
            return 0, 0
        if not directory_path.exists():
            logger.warning(f"插件目录不存在: {directory_path}")
            return 0, 1

        discovered = 0
        failed = 0
        for item in directory_path.iterdir():
            if not item.is_dir() or item.name.startswith(".") or item.name.startswith("__"):
                continue
            plugin_file = item / "plugin.py"
            if not plugin_file.exists():
                continue
            if self._load_plugin_module_file(plugin_file):
                discovered += 1
            else:
                failed += 1
        return discovered, failed

    def _load_plugin_module_file(self, plugin_file: Path) -> bool:
        try:
            try:
                relative_parent = plugin_file.resolve().parent.relative_to(Path.cwd().resolve())
                module_name = ".".join(relative_parent.parts)
            except ValueError:
                module_name = ".".join(plugin_file.resolve().parent.parts[-2:])

            spec = spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                raise ValueError(f"无法创建模块规范: {plugin_file}")

            module = module_from_spec(spec)
            module.__package__ = module_name
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            self._register_external_plugin_module(module, plugin_file)
            logger.debug(f"插件模块加载成功: {plugin_file}")
            return True
        except Exception as exc:
            error_message = f"加载插件模块失败: {plugin_file} - {exc}"
            self.failed_plugins[plugin_file.parent.name] = error_message
            logger.error(error_message)
            return False

    def _register_external_plugin_module(self, module: ModuleType, plugin_file: Path) -> str:
        plugin_class = getattr(module, "PLUGIN_CLASS", None)
        if plugin_class is None:
            raise ValueError(f"插件模块必须导出 PLUGIN_CLASS: {plugin_file}")
        return self.register_plugin_class(plugin_class, plugin_dir=str(plugin_file.parent))

    def _resolve_plugin_dir(self, plugin_class: Type[PluginBase]) -> str:
        try:
            return str(Path(getfile(plugin_class)).resolve().parent)
        except TypeError:
            module = sys.modules.get(plugin_class.__module__)
            module_file = getattr(module, "__file__", None) if module is not None else None
            if module_file:
                return str(Path(module_file).resolve().parent)
            raise

    def _check_plugin_version_compatibility(self, plugin_name: str, manifest_data: dict[str, Any]) -> Tuple[bool, str]:
        host_app = manifest_data.get("host_application")
        if not isinstance(host_app, dict):
            return True, ""

        min_version = host_app.get("min_version", "")
        max_version = host_app.get("max_version", "")
        if not min_version and not max_version:
            return True, ""

        try:
            current_version = VersionComparator.get_current_host_version()
            compatible, error_message = VersionComparator.is_version_in_range(current_version, min_version, max_version)
            if compatible:
                return True, ""
            return False, f"版本不兼容: {error_message}"
        except Exception as exc:
            logger.warning(f"插件 {plugin_name} 版本兼容性检查失败: {exc}")
            return False, f"插件 {plugin_name} 版本兼容性检查失败: {exc}"

    def _is_built_in_path(self, plugin_path: str | Path) -> bool:
        try:
            return Path(plugin_path).resolve().is_relative_to(self.built_in_root)
        except ValueError:
            return False

    def _show_stats(self, *, discovered: int, import_failed: int, loaded: int, registration_failed: int) -> None:
        stats = component_registry.get_registry_stats()
        logger.info(
            "插件启动完成: "
            f"发现 {discovered} 个外部插件包, 导入失败 {import_failed}, "
            f"本轮加载 {loaded}, 注册失败 {registration_failed}, "
            f"当前共 {stats['plugins']} 个插件 / {stats['total_components']} 个组件"
        )
        if self.failed_plugins:
            for plugin_name, error_message in self.failed_plugins.items():
                logger.warning(f"插件异常: {plugin_name} - {error_message}")

    def _show_plugin_components(self, plugin_name: str) -> None:
        plugin_info = component_registry.get_plugin_info(plugin_name)
        if plugin_info is None:
            logger.info(f"插件加载成功: {plugin_name}")
            return
        logger.info(
            f"插件加载成功: {plugin_info.display_name or plugin_name} "
            f"(built-in={plugin_info.is_built_in}, components={len(plugin_info.components)})"
        )


plugin_manager = PluginManager()
