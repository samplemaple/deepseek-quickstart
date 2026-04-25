"""标题数据模型"""

from pydantic import BaseModel


class TitleDiagnosis(BaseModel):
    """标题诊断结果"""
    has_core_keyword: bool
    has_intent_word: bool
    has_timeliness_word: bool
    has_attribute_word: bool
    length_valid: bool  # 15-30字符
    match_score: float  # 0-100
    suggestions: list[str] = []


class Title(BaseModel):
    """候选标题"""
    text: str
    diagnosis: TitleDiagnosis
