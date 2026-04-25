"""API请求/响应数据模型"""

from pydantic import BaseModel

from .content import ExtractedContent
from .enums import BusinessType, Platform, TemplateType
from .keyword import KeywordMatrix
from .platform import PlatformContent
from .quality import QualityReport
from .ranking import RankingResult
from .recreated import RecreatedContent


# ===== 请求模型 =====

class ExtractRequest(BaseModel):
    """内容提取请求"""
    url: str


class KeywordRequest(BaseModel):
    """关键词分析请求"""
    content: ExtractedContent


class RecreateRequest(BaseModel):
    """内容二创请求"""
    content: ExtractedContent
    keywords: KeywordMatrix
    template_type: TemplateType | None = None
    business_type: BusinessType | None = None


class ScoreRequest(BaseModel):
    """质量评分请求"""
    content: RecreatedContent


class AdaptRequest(BaseModel):
    """平台适配请求"""
    content: RecreatedContent
    platforms: list[Platform]


class MonitorRequest(BaseModel):
    """排名监测请求"""
    keyword: str
    platform: Platform
    content_title: str


class PipelineRequest(BaseModel):
    """一键执行管道请求"""
    url: str
    template_type: TemplateType | None = None
    platforms: list[Platform] = [Platform.DOUBAO, Platform.DEEPSEEK]
    business_type: BusinessType | None = None


# ===== 响应模型 =====

class ExtractResponse(BaseModel):
    """内容提取响应"""
    success: bool
    data: ExtractedContent | None = None
    error: str | None = None


class KeywordResponse(BaseModel):
    """关键词分析响应"""
    success: bool
    data: KeywordMatrix | None = None


class RecreateResponse(BaseModel):
    """内容二创响应"""
    success: bool
    data: RecreatedContent | None = None


class ScoreResponse(BaseModel):
    """质量评分响应"""
    success: bool
    data: QualityReport | None = None


class AdaptResponse(BaseModel):
    """平台适配响应"""
    success: bool
    data: list[PlatformContent] = []


class MonitorResponse(BaseModel):
    """排名监测响应"""
    success: bool
    data: RankingResult | None = None


class PipelineResponse(BaseModel):
    """管道执行响应"""
    success: bool
    extracted: ExtractedContent | None = None
    keywords: KeywordMatrix | None = None
    recreated: RecreatedContent | None = None
    quality: QualityReport | None = None
    platform_contents: list[PlatformContent] = []
    error: str | None = None


# ===== 错误响应 =====

class ErrorResponse(BaseModel):
    """统一错误响应"""
    success: bool = False
    error_code: str        # 如 "INVALID_URL", "EXTRACTION_FAILED"
    error_message: str     # 人类可读的错误描述
    retry_after: int | None = None  # 建议重试等待秒数
