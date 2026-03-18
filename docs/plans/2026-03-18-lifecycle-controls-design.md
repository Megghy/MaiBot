# Lifecycle Controls Design

**Problem**

当前 WebUI 的重启接口在请求处理中直接调用 `os.execv`，没有经过主程序统一的优雅关闭流程。重启后控制台信号链变得不稳定，用户也无法通过 WebUI 做受控关闭。系统控制接口同时缺少和其它管理接口一致的 token 鉴权。

**Goals**

- 提供统一的 `restart` / `shutdown` 生命周期控制能力。
- WebUI 触发重启和关闭时必须经过主程序自己的优雅关闭链路。
- 保持“退出后重新拉起”发生在同一个控制台中。
- 为系统控制接口补齐 token 鉴权。
- 在网页控制台增加重启和关闭按钮，并提供明确的状态提示。

**Recommended Approach**

采用 `launcher + worker` 双层模式。

- launcher 负责在当前控制台中启动 worker 子进程，并根据退出码决定结束还是重新拉起。
- worker 负责运行现有 MaiBot 主逻辑。
- WebUI 不再直接 `execv`；它只向 worker 内部的生命周期控制器发出“请求重启”或“请求关闭”命令。
- worker 收到命令后停止主服务任务，执行现有 `graceful_shutdown()`，然后以约定退出码结束。
- launcher 观察 worker 退出码：
  - `0`：正常关闭，launcher 一起退出
  - `RESTART_EXIT_CODE`：重新拉起 worker
  - 其它非零：异常退出，launcher 透传退出码

**Back-end Design**

- 新增一个轻量生命周期控制模块，负责：
  - 记录当前请求类型：`none` / `shutdown` / `restart`
  - 暴露 `request_shutdown()` / `request_restart()`
  - 暴露 `wait_for_request()`，供 worker 主循环感知退出请求
- worker 主循环不再只有无限 `gather(...)`，而是并行等待：
  - 运行中的服务任务
  - 生命周期请求事件
- 一旦生命周期事件触发：
  - 先取消服务任务
  - 再执行 `graceful_shutdown()`
  - 最后返回对应退出码给 launcher

**WebUI Design**

- `/api/webui/system/restart` 改为“提交重启请求并立即返回 accepted”。
- 新增 `/api/webui/system/shutdown`，同样只提交关闭请求。
- 两个接口都必须验证 Bearer token。
- 前端沿用现有系统控制页面和遮罩组件，新增：
  - `重启后端`
  - `关闭后端`
- 两个操作都加确认提示。
- 重启时展示“正在重启，请稍候”。
- 关闭时展示“后端已关闭，需要手动重新打开程序”。

**Trade-offs**

- 双层模式比单进程多一层启动逻辑，但它把“进程重启”和“服务关闭”解耦，Windows 下更稳。
- 现有启动命令保持不变，用户仍然从 `python bot.py` 启动，只是内部多了 launcher。
- 这次不做“单进程热重建所有全局状态”，避免在现有全局单例架构下引入更难定位的残留状态。

**Validation**

- 本地配置加载仍然成功。
- `POST /api/webui/system/restart` 需要有效 token，且能触发 worker 退出后由 launcher 拉起。
- `POST /api/webui/system/shutdown` 需要有效 token，且能触发 worker 优雅关闭并结束 launcher。
- 控制台 `Ctrl+C` 仍走优雅关闭，不受新逻辑破坏。
