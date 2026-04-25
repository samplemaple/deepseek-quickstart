"""FastAPI 接口层"""

from .app import app
from .routes import router

__all__ = ["app", "router"]
