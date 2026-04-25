"""二创内容数据模型"""

from datetime import datetime

from pydantic import BaseModel, Field

from .enums import TemplateType
from .title import Title


class ConversionStrategy(BaseModel):
    """转化策略"""
    contact_placements: list[str]  # 联系方式植入位置建议
    trust_elements: list[str]      # 信任建立元素
    call_to_action: list[str]      # 转化话术
    platform_warnings: list[str]   # 平台限流提醒


class RecreatedContent(BaseModel):
    """二创内容"""
    template_type: TemplateType
    titles: list[Title]
    body: str
    keywords_used: list[str]
    conversion: ConversionStrategy | None = None
    created_at: datetime = Field(default_factory=datetime.now)
