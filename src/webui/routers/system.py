"""
系统控制路由

提供系统重启、状态查询等功能
"""

import os
import sys
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.config.config import MMC_VERSION
from src.config.runtime_reload import hot_reload_all_configs

router = APIRouter(prefix="/system", tags=["system"])

# 记录启动时间
_start_time = time.time()


class RestartResponse(BaseModel):
    """重启响应"""

    success: bool
    message: str


class StatusResponse(BaseModel):
    """状态响应"""

    running: bool
    uptime: float
    version: str
    start_time: str


@router.post("/restart", response_model=RestartResponse)
async def restart_maibot():
    """
    重启麦麦主程序

    使用 os.execv 重启当前进程，配置更改将在重启后生效。
    注意：此操作会使麦麦暂时离线。
    """
    try:
        # 记录重启操作
        print(f"[{datetime.now()}] WebUI 触发重启操作")

        # 使用 os.execv 重启当前进程
        # 这会替换当前进程，保持相同的 PID
        python = sys.executable
        args = [python] + sys.argv

        # 返回成功响应（实际上这个响应可能不会发送，因为进程会立即重启）
        # 但我们仍然返回它以保持 API 一致性
        os.execv(python, args)

        return RestartResponse(success=True, message="麦麦正在重启中...")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重启失败: {str(e)}") from e


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
