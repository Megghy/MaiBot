from __future__ import annotations

import socket


def ensure_tcp_port_available(host: str, port: int, *, service_name: str) -> None:
    """在真正启动服务前做一次显式 bind 预检，端口被占用时直接失败。"""

    bind_host = host or "0.0.0.0"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, int(port)))
        except OSError as exc:
            raise RuntimeError(f"{service_name} 启动失败：端口 {bind_host}:{port} 已被占用或无法绑定") from exc


__all__ = ["ensure_tcp_port_available"]
