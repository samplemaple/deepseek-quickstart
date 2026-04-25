"""FastAPI 应用入口

创建 FastAPI 实例，注册路由，添加全局异常处理。
模块实例在此处创建并注入到路由层。

部署方式：单进程 uvicorn（适配 2核4G 云服务器）
启动命令：uvicorn geo_tool.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from geo_tool.api.routes import init_modules, router
from geo_tool.models.api import ErrorResponse
from geo_tool.modules.content_extractor import ContentExtractor
from geo_tool.modules.content_recreator import ContentRecreator
from geo_tool.modules.keyword_analyzer import KeywordAnalyzer
from geo_tool.modules.llm_client import LLMClient
from geo_tool.modules.pipeline import Pipeline
from geo_tool.modules.platform_adapter import PlatformAdapter
from geo_tool.modules.quality_scorer import QualityScorer
from geo_tool.modules.rank_monitor import RankMonitor

logger = logging.getLogger(__name__)

# ===== 创建 FastAPI 应用 =====

app = FastAPI(
    title="小红书GEO内容二创工具",
    description="基于GEO方法论的内容提取、关键词分析、内容二创、质量评分、平台适配和排名监测一站式工具",
    version="1.0.0",
)

# ===== CORS 中间件（允许所有来源） =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== 全局异常处理 =====


@app.exception_handler(ValidationError)
async def validation_error_handler(_request: Request, exc: ValidationError):
    """Pydantic 数据验证错误处理"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message=f"请求数据验证失败：{exc.error_count()} 个字段错误",
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError):
    """业务逻辑值错误处理"""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error_code="INVALID_INPUT",
            error_message=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(_request: Request, exc: Exception):
    """兜底异常处理，捕获所有未处理的异常"""
    logger.exception("未处理的异常: %s", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="服务器内部错误，请稍后重试",
            retry_after=5,
        ).model_dump(),
    )


# ===== 创建模块实例并注入路由 =====

def _init_app():
    """初始化所有核心模块并注入到路由层"""
    # 创建 LLM 客户端（共享实例）
    llm_client = LLMClient()

    # 创建各核心模块
    extractor = ContentExtractor()
    analyzer = KeywordAnalyzer(llm_client=llm_client)
    recreator = ContentRecreator(llm_client=llm_client)
    scorer = QualityScorer(llm_client=llm_client)
    adapter = PlatformAdapter()
    monitor = RankMonitor()

    # 创建管道编排器
    pipeline = Pipeline(
        extractor=extractor,
        analyzer=analyzer,
        recreator=recreator,
        scorer=scorer,
        adapter=adapter,
    )

    # 注入模块到路由层
    init_modules(
        extractor=extractor,
        analyzer=analyzer,
        recreator=recreator,
        scorer=scorer,
        adapter=adapter,
        monitor=monitor,
        pipeline=pipeline,
    )

    # 注册路由
    app.include_router(router)

    # 挂载静态文件目录
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # 首页路由 — 返回前端界面
    @app.get("/")
    async def index():
        html_path = os.path.join(static_dir, "index.html")
        return FileResponse(html_path)

    logger.info("应用初始化完成，所有模块已就绪")


# 执行初始化
_init_app()
