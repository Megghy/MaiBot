import asyncio
import time
from maim_message import MessageServer

from src.config.config import global_config
from src.common.logger import get_logger
from src.common.server import get_global_server, Server

# 导入消息API和traceback模块
from src.common.message import get_global_api
from src.runtime.startup import build_service_tasks, initialize_runtime_services, setup_webui_server

# 插件系统现在使用统一的插件加载器

logger = get_logger("main")


class MainSystem:
    def __init__(self):
        # 使用消息API替代直接的FastAPI实例
        self.app: MessageServer = get_global_api()
        self.server: Server = get_global_server()
        self.webui_server = setup_webui_server()

    async def initialize(self):
        """初始化系统组件"""
        logger.info(f"正在唤醒{global_config.bot.nickname}......")

        init_start_time = time.time()
        await initialize_runtime_services(self.app)

        logger.info(f"""
--------------------------------
全部系统初始化完成，{global_config.bot.nickname}已成功唤醒
--------------------------------
如果想要自定义{global_config.bot.nickname}的功能,请查阅：https://docs.mai-mai.org/manual/usage/
或者遇到了问题，请访问我们的文档:https://docs.mai-mai.org/
--------------------------------
如果你想要编写或了解插件相关内容，请访问开发文档https://docs.mai-mai.org/develop/
--------------------------------
如果你需要查阅模型的消耗以及麦麦的统计数据，请访问根目录的maibot_statistics.html文件
""")
        try:
            init_time = int(1000 * (time.time() - init_start_time))
            logger.info(f"初始化完成，神经元放电{init_time}次")
        except Exception as e:
            logger.error(f"启动大脑和外部世界失败: {e}")
            raise

    async def schedule_tasks(self):
        """调度定时任务"""
        await asyncio.gather(*build_service_tasks(self.app, self.server, self.webui_server))

    # async def forget_memory_task(self):
    #     """记忆遗忘任务"""
    #     while True:
    #         await asyncio.sleep(global_config.memory.forget_memory_interval)
    #         logger.info("[记忆遗忘] 开始遗忘记忆...")
    #         await self.hippocampus_manager.forget_memory(percentage=global_config.memory.memory_forget_percentage)  # type: ignore
    #         logger.info("[记忆遗忘] 记忆遗忘完成")


async def main():
    """主函数"""
    system = MainSystem()
    await asyncio.gather(
        system.initialize(),
        system.schedule_tasks(),
    )


if __name__ == "__main__":
    asyncio.run(main())
