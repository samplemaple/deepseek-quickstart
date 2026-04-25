"""平台适配数据模型"""

from pydantic import BaseModel

from .enums import Platform


class PlatformConfig(BaseModel):
    """平台偏好配置"""
    platform: Platform
    recommended_channels: list[str]
    content_format_rules: list[str]
    notes: list[str]


class PlatformContent(BaseModel):
    """平台适配内容"""
    platform: Platform
    adapted_body: str
    publish_suggestions: PlatformConfig
