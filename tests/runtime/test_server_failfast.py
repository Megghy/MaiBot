from __future__ import annotations

import pytest

from src.common.server import Server
from src.webui.webui_server import WebUIServer


@pytest.mark.asyncio
async def test_server_run_fails_fast_when_port_check_fails(monkeypatch):
    server = Server(host="127.0.0.1", port=8000)

    monkeypatch.setattr(
        "src.common.server.ensure_tcp_port_available",
        lambda host, port, service_name: (_ for _ in ()).throw(
            RuntimeError(f"{service_name} 启动失败：端口 {host}:{port} 已被占用或无法绑定")
        ),
    )

    with pytest.raises(RuntimeError, match="HTTP服务器 启动失败|HTTP服务器启动失败"):
        await server.run()


@pytest.mark.asyncio
async def test_server_run_converts_uvicorn_system_exit(monkeypatch):
    class FakeUvicornServer:
        def __init__(self, config):
            del config

        async def serve(self):
            raise SystemExit(1)

        async def shutdown(self):
            return None

    server = Server(host="127.0.0.1", port=8000)

    monkeypatch.setattr("src.common.server.ensure_tcp_port_available", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.common.server.UvicornServer", FakeUvicornServer)

    with pytest.raises(RuntimeError, match="HTTP服务器启动失败"):
        await server.run()


@pytest.mark.asyncio
async def test_webui_server_start_fails_fast_when_port_check_fails(monkeypatch):
    monkeypatch.setattr(WebUIServer, "_show_access_token", lambda self: None)
    monkeypatch.setattr(WebUIServer, "_register_api_routes", lambda self: None)
    monkeypatch.setattr(WebUIServer, "_setup_static_files", lambda self: None)
    webui_server = WebUIServer(host="0.0.0.0", port=8001)

    monkeypatch.setattr(
        "src.webui.webui_server.ensure_tcp_port_available",
        lambda host, port, service_name: (_ for _ in ()).throw(
            RuntimeError(f"{service_name} 启动失败：端口 {host}:{port} 已被占用或无法绑定")
        ),
    )

    with pytest.raises(RuntimeError, match="WebUI服务器 启动失败|WebUI服务器启动失败"):
        await webui_server.start()


@pytest.mark.asyncio
async def test_webui_server_start_converts_uvicorn_system_exit(monkeypatch):
    class FakeUvicornServer:
        def __init__(self, config):
            del config

        async def serve(self):
            raise SystemExit(1)

        async def shutdown(self):
            return None

    monkeypatch.setattr(WebUIServer, "_show_access_token", lambda self: None)
    monkeypatch.setattr(WebUIServer, "_register_api_routes", lambda self: None)
    monkeypatch.setattr(WebUIServer, "_setup_static_files", lambda self: None)
    webui_server = WebUIServer(host="0.0.0.0", port=8001)

    monkeypatch.setattr("src.webui.webui_server.ensure_tcp_port_available", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.webui.webui_server.UvicornServer", FakeUvicornServer)

    with pytest.raises(RuntimeError, match="WebUI服务器启动失败"):
        await webui_server.start()
