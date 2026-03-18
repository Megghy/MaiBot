"""
系统控制路由

提供系统重启、关闭、状态查询等功能
"""

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel

from src.config.config import MMC_VERSION
from src.config.runtime_reload import hot_reload_all_configs
from src.webui.token_manager import get_token_manager

from src.runtime.lifecycle import request_restart, request_shutdown

router = APIRouter(prefix="/system", tags=["system"])

# 记录启动时间
_start_time = time.time()


class RestartResponse(BaseModel):
    """重启响应"""

    success: bool
    message: str


class ShutdownResponse(BaseModel):
    """关闭响应"""

    success: bool
    message: str


class StatusResponse(BaseModel):
    """状态响应"""

    running: bool
    uptime: float
    version: str
    start_time: str


def verify_auth_token(authorization: Optional[str]) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未提供有效的认证信息")

    token = authorization.replace("Bearer ", "")
    token_manager = get_token_manager()
    if not token_manager.verify_token(token):
        raise HTTPException(status_code=401, detail="Token 无效或已过期")


async def _request_lifecycle_after_response(action: str) -> None:
    request_ok = request_restart() if action == "restart" else request_shutdown()
    if not request_ok:
        print(f"[{datetime.now()}] 生命周期请求已存在，忽略重复请求: {action}")


@router.post("/restart", response_model=RestartResponse)
async def restart_maibot(background_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    """
    重启麦麦主程序

    请求主进程优雅关闭后由 launcher 自动重新拉起。
    """
    try:
        verify_auth_token(authorization)
        background_tasks.add_task(_request_lifecycle_after_response, "restart")
        return RestartResponse(success=True, message="麦麦正在重启中...")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重启失败: {str(e)}") from e


@router.post("/shutdown", response_model=ShutdownResponse)
async def shutdown_maibot(background_tasks: BackgroundTasks, authorization: Optional[str] = Header(None)):
    """
    请求主进程优雅关闭。
    """
    try:
        verify_auth_token(authorization)
        background_tasks.add_task(_request_lifecycle_after_response, "shutdown")
        return ShutdownResponse(success=True, message="麦麦正在关闭中...")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"关闭失败: {str(e)}") from e


@router.get("/status", response_model=StatusResponse)
async def get_maibot_status():
    """
    获取麦麦运行状态

    返回麦麦的运行状态、运行时长和版本信息。
    """
    try:
        uptime = time.time() - _start_time

        # 尝试获取版本信息（需要根据实际情况调整）
        version = MMC_VERSION  # 可以从配置或常量中读取

        return StatusResponse(
            running=True, uptime=uptime, version=version, start_time=datetime.fromtimestamp(_start_time).isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}") from e


# 可选：添加更多系统控制功能


@router.post("/reload-config")
async def reload_config():
    """
    热重载配置（不重启进程）
    """
    try:
        reload_result = hot_reload_all_configs()
        message = "配置已重新加载并热生效"
        if reload_result.pending_restart:
            message += f"，以下配置节仍需重启: {', '.join(reload_result.pending_restart)}"
        return {"success": True, "message": message, "reload": reload_result.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"热重载配置失败: {str(e)}") from e
