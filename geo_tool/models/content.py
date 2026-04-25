"""内容提取数据模型"""

from datetime import datetime

from pydantic import BaseModel, Field


class NoteMetadata(BaseModel):
    """小红书笔记元数据"""
    title: str
    author: str
    publish_time: str
    likes: int = 0
    collects: int = 0
    comments: int = 0


class ExtractedContent(BaseModel):
    """提取的笔记内容"""
    url: str
    text: str
    image_urls: list[str] = []
    metadata: NoteMetadata
    extracted_at: datetime = Field(default_factory=datetime.now)
