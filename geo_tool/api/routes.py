"""API 路由定义

使用 FastAPI APIRouter 定义7个 API 端点，
每个端点对应一个核心业务模块的功能。
"""

import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from geo_tool.models.api import (
    AdaptRequest,
    AdaptResponse,
    ErrorResponse,
    ExtractRequest,
    ExtractResponse,
    KeywordRequest,
    KeywordResponse,
    MonitorRequest,
    MonitorResponse,
    PipelineRequest,
    PipelineResponse,
    RecreateRequest,
    RecreateResponse,
    ScoreRequest,
    ScoreResponse,
)

logger = logging.getLogger(__name__)

# 创建路由器，统一前缀 /api/v1
router = APIRouter(prefix="/api/v1")

# ===== 模块实例引用（由 app.py 注入） =====
# 这些变量在 app.py 中创建模块实例后赋值
_extractor = None
_analyzer = None
_recreator = None
_scorer = None
_adapter = None
_monitor = None
_pipeline = None


def init_modules(*, extractor, analyzer, recreator, scorer, adapter, monitor, pipeline):
    """初始化路由所需的模块实例

    由 app.py 在创建应用时调用，注入所有核心模块。

    Args:
        extractor: ContentExtractor 实例
        analyzer: KeywordAnalyzer 实例
        recreator: ContentRecreator 实例
        scorer: QualityScorer 实例
        adapter: PlatformAdapter 实例
        monitor: RankMonitor 实例
        pipeline: Pipeline 实例
    """
    global _extractor, _analyzer, _recreator, _scorer, _adapter, _monitor, _pipeline
    _extractor = extractor
    _analyzer = analyzer
    _recreator = recreator
    _scorer = scorer
    _adapter = adapter
    _monitor = monitor
    _pipeline = pipeline


# ===== API 端点 =====


@router.post("/extract", response_model=ExtractResponse, responses={500: {"model": ErrorResponse}})
async def extract_content(req: ExtractRequest):
    """提取小红书笔记内容"""
    try:
        result = await _extractor.extract(req.url)
        if result.get("success"):
            return ExtractResponse(success=True, data=result["data"])
        return ExtractResponse(success=False, error=result.get("error", "提取失败"))
    except Exception as e:
        logger.exception("内容提取异常")
        raise _build_api_error("EXTRACTION_FAILED", f"内容提取异常：{e}") from e


@router.post("/analyze-keywords", response_model=KeywordResponse, responses={500: {"model": ErrorResponse}})
async def analyze_keywords(req: KeywordRequest):
    """分析 GEO 关键词"""
    try:
        keywords = await _analyzer.analyze(req.content)
        return KeywordResponse(success=True, data=keywords)
    except Exception as e:
        logger.exception("关键词分析异常")
        raise _build_api_error("KEYWORD_ANALYSIS_FAILED", f"关键词分析异常：{e}") from e


@router.post("/recreate", response_model=RecreateResponse, responses={500: {"model": ErrorResponse}})
async def recreate_content(req: RecreateRequest):
    """内容二创"""
    try:
        recreated = await _recreator.recreate(
            content=req.content,
            keywords=req.keywords,
            template_type=req.template_type,
            business_type=req.business_type,
        )
        return RecreateResponse(success=True, data=recreated)
    except Exception as e:
        logger.exception("内容二创异常")
        raise _build_api_error("RECREATE_FAILED", f"内容二创异常：{e}") from e


@router.post("/score", response_model=ScoreResponse, responses={500: {"model": ErrorResponse}})
async def score_content(req: ScoreRequest):
    """内容质量评分"""
    try:
        report = await _scorer.score(req.content)
        return ScoreResponse(success=True, data=report)
    except Exception as e:
        logger.exception("质量评分异常")
        raise _build_api_error("SCORE_FAILED", f"质量评分异常：{e}") from e


@router.post("/adapt", response_model=AdaptResponse, responses={500: {"model": ErrorResponse}})
async def adapt_content(req: AdaptRequest):
    """多平台适配"""
    try:
        contents = await _adapter.adapt(req.content, req.platforms)
        return AdaptResponse(success=True, data=contents)
    except Exception as e:
        logger.exception("平台适配异常")
        raise _build_api_error("ADAPT_FAILED", f"平台适配异常：{e}") from e


@router.post("/monitor", response_model=MonitorResponse, responses={500: {"model": ErrorResponse}})
async def monitor_ranking(req: MonitorRequest):
    """排名监测"""
    try:
        result = await _monitor.check_ranking(
            keyword=req.keyword,
            platform=req.platform,
            content_title=req.content_title,
        )
        return MonitorResponse(success=True, data=result)
    except Exception as e:
        logger.exception("排名监测异常")
        raise _build_api_error("MONITOR_FAILED", f"排名监测异常：{e}") from e


@router.post("/pipeline", response_model=PipelineResponse, responses={500: {"model": ErrorResponse}})
async def run_pipeline(req: PipelineRequest):
    """一键执行完整管道：提取→分析→二创→评分→适配"""
    try:
        result = await _pipeline.run(
            url=req.url,
            template_type=req.template_type,
            platforms=req.platforms,
            business_type=req.business_type,
        )
        return result
    except Exception as e:
        logger.exception("管道执行异常")
        raise _build_api_error("PIPELINE_FAILED", f"管道执行异常：{e}") from e


# ===== 客户端提取数据中转 =====
# 临时存储客户端提取的数据（内存中，带过期清理）
_client_extracts: dict[str, dict[str, Any]] = {}


class ClientExtractRequest(BaseModel):
    """客户端提取数据提交请求"""
    title: str
    text: str
    image_urls: list[str] = []
    source_url: str = ""


class ClientExtractResponse(BaseModel):
    """客户端提取数据提交响应，返回一次性 token"""
    success: bool
    token: str


@router.post("/client-extract", response_model=ClientExtractResponse)
async def receive_client_extract(req: ClientExtractRequest):
    """接收客户端（Bookmarklet）提取的笔记数据，返回一次性 token

    客户端在小红书页面提取内容后，POST 到此接口，
    服务端生成一次性 token 存储数据，客户端用 token 跳转到前端页面。
    """
    # 清理过期数据（超过5分钟的）
    now = time.time()
    expired = [k for k, v in _client_extracts.items() if now - v["ts"] > 300]
    for k in expired:
        del _client_extracts[k]

    # 生成一次性 token
    token = secrets.token_urlsafe(16)
    _client_extracts[token] = {
        "title": req.title,
        "text": req.text,
        "image_urls": req.image_urls,
        "source_url": req.source_url,
        "ts": now,
    }
    logger.info("客户端提取数据已存储，token=%s，标题=%s", token, req.title[:30])
    return ClientExtractResponse(success=True, token=token)


@router.get("/client-extract/{token}")
async def get_client_extract(token: str):
    """根据 token 获取客户端提取的数据（5分钟内可重复获取）"""
    data = _client_extracts.get(token)
    if data is None:
        return {"success": False, "error": "token 无效或已过期"}
    return {"success": True, "data": data}


# ===== 辅助函数 =====


def _build_api_error(error_code: str, error_message: str, retry_after: int | None = None):
    """构建统一的 HTTP 异常，响应体为 ErrorResponse 格式"""
    from fastapi import HTTPException

    return HTTPException(
        status_code=500,
        detail=ErrorResponse(
            error_code=error_code,
            error_message=error_message,
            retry_after=retry_after,
        ).model_dump(),
    )
