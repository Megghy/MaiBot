import asyncio
import hashlib
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback
import shutil

WORKER_FLAG = "--worker"
RESTART_EXIT_CODE = 23
SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR / ".env"
TEMPLATE_ENV_PATH = SCRIPT_DIR / "template" / "template.env"

logger = None
confirm_logger = None
shutdown_logging = None


def _get_logger():
    if logger is None:
        raise RuntimeError("主日志器尚未初始化")
    return logger


def _get_confirm_logger():
    if confirm_logger is None:
        raise RuntimeError("确认日志器尚未初始化")
    return confirm_logger


def _load_env_file() -> None:
    from dotenv import load_dotenv

    if ENV_PATH.exists():
        load_dotenv(str(ENV_PATH), override=True)
        print("成功加载环境变量配置")
        return

    try:
        if TEMPLATE_ENV_PATH.exists():
            shutil.copyfile(TEMPLATE_ENV_PATH, ENV_PATH)
            print("未找到.env，已从 template/template.env 自动创建")
            load_dotenv(str(ENV_PATH), override=True)
            return

        print("未找到.env文件，也未找到模板 template/template.env")
        raise FileNotFoundError(".env 文件不存在，请创建并配置所需的环境变量")
    except Exception as e:
        print(f"自动创建 .env 失败: {e}")
        raise


def _bootstrap_worker_runtime() -> None:
    global logger, confirm_logger, shutdown_logging

    os.chdir(SCRIPT_DIR)
    _load_env_file()

    from rich.traceback import install
    from src.common.logger import initialize_logging, get_logger, shutdown_logging as shutdown_logging_fn

    initialize_logging()
    install(extra_lines=3)

    logger = get_logger("main")
    confirm_logger = get_logger("confirm")
    shutdown_logging = shutdown_logging_fn

    _get_logger().info(f"已设置工作目录为: {SCRIPT_DIR}")


def easter_egg():
    # 彩蛋
    from colorama import init, Fore

    init()
    text = "多年以后，面对AI行刑队，张三将会回想起他2023年在会议上讨论人工智能的那个下午"
    rainbow_colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.CYAN, Fore.BLUE, Fore.MAGENTA]
    rainbow_text = ""
    for i, char in enumerate(text):
        rainbow_text += rainbow_colors[i % len(rainbow_colors)] + char
    print(rainbow_text)


async def graceful_shutdown():  # sourcery skip: use-named-expression
    shutdown_logger = _get_logger()

    try:
        shutdown_logger.info("正在优雅关闭麦麦...")

        from src.plugin_system.base.component_types import EventType
        from src.runtime import events_manager, get_runtime_manager
        from src.common.server import get_global_server
        from src.manager.async_task_manager import async_task_manager

        # 触发 ON_STOP 事件
        await events_manager.handle_mai_events(event_type=EventType.ON_STOP)

        # 关闭运行时会话
        await get_runtime_manager().shutdown()

        # 停止所有异步任务
        await async_task_manager.stop_and_wait_all_tasks()

        # 关闭 HTTP/WebUI 层
        try:
            server = get_global_server()
            await server.shutdown()
        except Exception as e:
            shutdown_logger.warning(f"关闭 HTTP 服务器时出错: {e}")

        try:
            from src.webui.webui_server import get_webui_server

            webui_server = get_webui_server()
            if webui_server and webui_server._server:
                await webui_server.shutdown()
        except Exception as e:
            shutdown_logger.warning(f"关闭 WebUI 服务器时出错: {e}")

        # 获取所有剩余任务，排除当前任务
        remaining_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        if remaining_tasks:
            shutdown_logger.info(f"正在取消 {len(remaining_tasks)} 个剩余任务...")

            # 取消所有剩余任务
            for task in remaining_tasks:
                if not task.done():
                    task.cancel()

            # 等待所有任务完成，设置超时
            try:
                await asyncio.wait_for(asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=15.0)
                shutdown_logger.info("所有剩余任务已成功取消")
            except asyncio.TimeoutError:
                shutdown_logger.warning("等待任务取消超时，强制继续关闭")
            except Exception as e:
                shutdown_logger.error(f"等待任务取消时发生异常: {e}")

        shutdown_logger.info("麦麦优雅关闭完成")

    except Exception as e:
        shutdown_logger.error(f"麦麦关闭失败: {e}", exc_info=True)


def _calculate_file_hash(file_path: Path, file_type: str) -> str:
    """计算文件的MD5哈希值"""
    if not file_path.exists():
        _get_logger().error(f"{file_type} 文件不存在")
        raise FileNotFoundError(f"{file_type} 文件不存在")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _check_agreement_status(file_hash: str, confirm_file: Path, env_var: str) -> tuple[bool, bool]:
    """检查协议确认状态

    Returns:
        tuple[bool, bool]: (已确认, 未更新)
    """
    # 检查环境变量确认
    if file_hash == os.getenv(env_var):
        return True, False

    # 检查确认文件
    if confirm_file.exists():
        with open(confirm_file, "r", encoding="utf-8") as f:
            confirmed_content = f.read()
        if file_hash == confirmed_content:
            return True, False

    return False, True


def _prompt_user_confirmation(eula_hash: str, privacy_hash: str) -> None:
    """提示用户确认协议"""
    _get_confirm_logger().critical("EULA或隐私条款内容已更新，请在阅读后重新确认，继续运行视为同意更新后的以上两款协议")
    _get_confirm_logger().critical(
        f'输入"同意"或"confirmed"或设置环境变量"EULA_AGREE={eula_hash}"和"PRIVACY_AGREE={privacy_hash}"继续运行'
    )

    while True:
        user_input = input().strip().lower()
        if user_input in ["同意", "confirmed"]:
            return
        _get_confirm_logger().critical('请输入"同意"或"confirmed"以继续运行')


def _save_confirmations(eula_updated: bool, privacy_updated: bool, eula_hash: str, privacy_hash: str) -> None:
    """保存用户确认结果"""
    if eula_updated:
        _get_logger().info(f"更新EULA确认文件{eula_hash}")
        Path("eula.confirmed").write_text(eula_hash, encoding="utf-8")

    if privacy_updated:
        _get_logger().info(f"更新隐私条款确认文件{privacy_hash}")
        Path("privacy.confirmed").write_text(privacy_hash, encoding="utf-8")


def check_eula():
    """检查EULA和隐私条款确认状态"""
    # 计算文件哈希值
    eula_hash = _calculate_file_hash(Path("EULA.md"), "EULA.md")
    privacy_hash = _calculate_file_hash(Path("PRIVACY.md"), "PRIVACY.md")

    # 检查确认状态
    eula_confirmed, eula_updated = _check_agreement_status(eula_hash, Path("eula.confirmed"), "EULA_AGREE")
    privacy_confirmed, privacy_updated = _check_agreement_status(
        privacy_hash, Path("privacy.confirmed"), "PRIVACY_AGREE"
    )

    # 早期返回：如果都已确认且未更新
    if eula_confirmed and privacy_confirmed:
        return

    # 如果有更新，需要重新确认
    if eula_updated or privacy_updated:
        _prompt_user_confirmation(eula_hash, privacy_hash)
        _save_confirmations(eula_updated, privacy_updated, eula_hash, privacy_hash)


def raw_main():
    from src.main import MainSystem

    # 利用 TZ 环境变量设定程序工作的时区
    if platform.system().lower() != "windows":
        time.tzset()  # type: ignore

    check_eula()
    _get_logger().info("检查EULA和隐私条款完成")

    easter_egg()

    # 返回MainSystem实例
    return MainSystem()


def _build_worker_command() -> list[str]:
    worker_args = [arg for arg in sys.argv[1:] if arg != WORKER_FLAG]
    return [sys.executable, os.path.abspath(__file__), WORKER_FLAG, *worker_args]


def _wait_for_worker_exit(process: subprocess.Popen) -> int:
    try:
        return process.wait()
    except KeyboardInterrupt:
        print("[launcher] 收到中断信号，等待 worker 优雅退出...")
        try:
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print("[launcher] Worker 长时间未退出，尝试终止进程...")
            process.terminate()
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[launcher] Worker 终止超时，强制结束进程")
                process.kill()
                return process.wait()


def launcher_main() -> int:
    worker_command = _build_worker_command()
    print(f"[launcher] 启动 worker: {' '.join(worker_command)}")

    while True:
        process = subprocess.Popen(worker_command, cwd=str(SCRIPT_DIR))
        exit_code = _wait_for_worker_exit(process)

        if exit_code == RESTART_EXIT_CODE:
            print("[launcher] 检测到重启请求，正在重新拉起 worker...")
            continue

        print(f"[launcher] worker 已退出，退出码: {exit_code}")
        return exit_code


def worker_main() -> int:
    exit_code = 0
    worker_loop = None

    try:
        _bootstrap_worker_runtime()

        from src.runtime.lifecycle import LifecycleAction, wait_for_request, consume_requested_action, reset_lifecycle_state
        from src.common.logger import initialize_ws_handler

        reset_lifecycle_state()
        main_system = raw_main()

        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)

        initialize_ws_handler(worker_loop)

        try:
            worker_loop.run_until_complete(main_system.initialize())

            main_tasks = worker_loop.create_task(main_system.schedule_tasks())
            lifecycle_task = worker_loop.create_task(wait_for_request())

            done, _ = worker_loop.run_until_complete(
                asyncio.wait({main_tasks, lifecycle_task}, return_when=asyncio.FIRST_COMPLETED)
            )

            if lifecycle_task in done:
                requested_action = consume_requested_action()
                logger.info(f"收到生命周期请求: {requested_action}，开始优雅关闭")

                if not main_tasks.done():
                    main_tasks.cancel()
                worker_loop.run_until_complete(asyncio.gather(main_tasks, return_exceptions=True))
                worker_loop.run_until_complete(graceful_shutdown())

                exit_code = RESTART_EXIT_CODE if requested_action == LifecycleAction.RESTART else 0
            else:
                lifecycle_task.cancel()
                worker_loop.run_until_complete(asyncio.gather(lifecycle_task, return_exceptions=True))

                try:
                    main_tasks.result()
                    logger.warning("主服务任务意外结束")
                except asyncio.CancelledError:
                    logger.warning("主服务任务被取消")
                except Exception as task_error:
                    logger.error(f"主服务任务异常退出: {task_error}")

                worker_loop.run_until_complete(graceful_shutdown())
                exit_code = 1

        except KeyboardInterrupt:
            _get_logger().warning("收到中断信号，正在优雅关闭...")
            if worker_loop and not worker_loop.is_closed():
                try:
                    worker_loop.run_until_complete(graceful_shutdown())
                except Exception as ge:
                    _get_logger().error(f"优雅关闭时发生错误: {ge}")
            exit_code = 0

    except Exception as e:
        if logger is not None:
            _get_logger().error(f"主程序发生异常: {str(e)} {str(traceback.format_exc())}")
        else:
            print(f"[worker] 主程序发生异常: {str(e)} {str(traceback.format_exc())}")
        exit_code = 1
    finally:
        if worker_loop and not worker_loop.is_closed():
            worker_loop.close()
            print("[主程序] 事件循环已关闭")

        if shutdown_logging is not None:
            try:
                shutdown_logging()
            except Exception as e:
                print(f"关闭日志系统时出错: {e}")

        print("[主程序] 准备退出...")

    return exit_code


if __name__ == "__main__":
    if WORKER_FLAG in sys.argv[1:]:
        os._exit(worker_main())

    os._exit(launcher_main())
