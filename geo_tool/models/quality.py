"""质量评分数据模型"""

from pydantic import BaseModel, Field

from .enums import QualityLevel


class DimensionScore(BaseModel):
    """维度评分"""
    dimension: str
    weight: float
    score: float = Field(ge=0, le=100)
    deduction_reasons: list[str] = []
    suggestions: list[str] = []


class QualityReport(BaseModel):
    """质量评分报告"""
    dimension_scores: list[DimensionScore]
    weighted_total: float = Field(ge=0, le=100)
    quality_level: QualityLevel
    priority_improvements: list[str] = []
