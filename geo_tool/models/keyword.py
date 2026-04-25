"""关键词数据模型"""

from pydantic import BaseModel, Field

from .enums import DecisionStage, KeywordType


class GEOKeyword(BaseModel):
    """GEO关键词"""
    word: str
    keyword_type: KeywordType
    decision_stage: DecisionStage
    business_value: float = Field(ge=0, le=1000)  # 需求成熟度×商业潜力×匹配精准度
    longtail_variants: list[str] = []


class KeywordMatrix(BaseModel):
    """关键词矩阵"""
    core_topics: list[str]
    keywords: list[GEOKeyword]

    def sorted_by_value(self) -> list[GEOKeyword]:
        """按商业价值降序排列关键词"""
        return sorted(self.keywords, key=lambda k: k.business_value, reverse=True)
