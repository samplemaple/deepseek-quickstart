"""数据模型定义"""

from .enums import (
    BusinessType,
    DecisionStage,
    KeywordType,
    Platform,
    QualityLevel,
    TemplateType,
)
from .content import ExtractedContent, NoteMetadata
from .keyword import GEOKeyword, KeywordMatrix
from .title import Title, TitleDiagnosis
from .recreated import ConversionStrategy, RecreatedContent
from .quality import DimensionScore, QualityReport
from .platform import PlatformConfig, PlatformContent
from .ranking import CitationInfo, DiagnosisReport, RankingResult
from .api import (
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

__all__ = [
    # Enums
    "BusinessType",
    "DecisionStage",
    "KeywordType",
    "Platform",
    "QualityLevel",
    "TemplateType",
    # Content
    "ExtractedContent",
    "NoteMetadata",
    # Keyword
    "GEOKeyword",
    "KeywordMatrix",
    # Title
    "Title",
    "TitleDiagnosis",
    # Recreated
    "ConversionStrategy",
    "RecreatedContent",
    # Quality
    "DimensionScore",
    "QualityReport",
    # Platform
    "PlatformConfig",
    "PlatformContent",
    # Ranking
    "CitationInfo",
    "DiagnosisReport",
    "RankingResult",
    # API
    "AdaptRequest",
    "AdaptResponse",
    "ErrorResponse",
    "ExtractRequest",
    "ExtractResponse",
    "KeywordRequest",
    "KeywordResponse",
    "MonitorRequest",
    "MonitorResponse",
    "PipelineRequest",
    "PipelineResponse",
    "RecreateRequest",
    "RecreateResponse",
    "ScoreRequest",
    "ScoreResponse",
]
