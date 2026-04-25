"""排名监测数据模型"""

from datetime import datetime

from pydantic import BaseModel, Field

from .enums import Platform


class CitationInfo(BaseModel):
    """引用信息"""
    is_cited: bool
    position: int | None = None  # 第几条推荐
    cited_snippet: str | None = None


class DiagnosisReport(BaseModel):
    """诊断报告"""
    is_indexed: bool | None = None
    title_match: bool | None = None
    quality_score: float | None = None
    diagnosis_summary: str
    optimization_strategy: str  # 小修小补/大改重发/推倒重来
    action_items: list[str]


class RankingResult(BaseModel):
    """排名监测结果"""
    keyword: str
    platform: Platform
    citation: CitationInfo
    diagnosis: DiagnosisReport | None = None
    checked_at: datetime = Field(default_factory=datetime.now)
