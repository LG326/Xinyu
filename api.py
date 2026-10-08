"""兼容入口：生产环境请将 FastAPI 应用视为纯 JSON 后端。"""

from backend.app import app, create_app

__all__ = ["app", "create_app"]
