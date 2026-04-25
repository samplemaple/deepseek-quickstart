"""ContentExtractor 单元测试

测试 URL 验证、解析逻辑和 extract() 方法的核心行为。
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from geo_tool.modules.content_extractor import (
    ContentExtractor,
    LONG_LINK_PATTERN,
    SHORT_LINK_PATTERN,
    USER_AGENTS,
)
from geo_tool.models.content import ExtractedContent, NoteMetadata


# ===== URL 验证测试 =====


class TestValidateUrl:
    """测试 _validate_url 方法"""

    def setup_method(self):
        self.extractor = ContentExtractor()

    # --- 有效链接 ---

    def test_valid_short_link(self):
        """有效的短链应通过验证"""
        assert self.extractor._validate_url("https://xhslink.com/abc123") is True

    def test_valid_short_link_with_trailing_slash(self):
        """带尾部斜杠的短链应通过验证"""
        assert self.extractor._validate_url("https://xhslink.com/abc123/") is True

    def test_valid_short_link_http(self):
        """HTTP 协议的短链应通过验证"""
        assert self.extractor._validate_url("http://xhslink.com/abc123") is True

    def test_valid_long_link_explore(self):
        """有效的 explore 长链应通过验证"""
        url = "https://www.xiaohongshu.com/explore/6501234567890abcdef12345"
        assert self.extractor._validate_url(url) is True

    def test_valid_long_link_without_www(self):
        """不带 www 的长链应通过验证"""
        url = "https://xiaohongshu.com/explore/6501234567890abcdef12345"
        assert self.extractor._validate_url(url) is True

    def test_valid_long_link_discovery(self):
        """有效的 discovery/item 长链应通过验证"""
        url = "https://www.xiaohongshu.com/discovery/item/6501234567890abcdef12345"
        assert self.extractor._validate_url(url) is True

    def test_valid_long_link_with_query_params(self):
        """带查询参数的长链应通过验证"""
        url = "https://www.xiaohongshu.com/explore/6501234567890abcdef12345?source=share"
        assert self.extractor._validate_url(url) is True

    # --- 无效链接 ---

    def test_empty_string(self):
        """空字符串应不通过验证"""
        assert self.extractor._validate_url("") is False

    def test_none_input(self):
        """None 输入应不通过验证"""
        assert self.extractor._validate_url(None) is False

    def test_random_string(self):
        """随机字符串应不通过验证"""
        assert self.extractor._validate_url("hello world") is False

    def test_other_website_url(self):
        """其他网站 URL 应不通过验证"""
        assert self.extractor._validate_url("https://www.baidu.com") is False

    def test_invalid_note_id_format(self):
        """笔记 ID 格式不正确的长链应不通过验证"""
        url = "https://www.xiaohongshu.com/explore/invalid-id"
        assert self.extractor._validate_url(url) is False

    def test_short_link_without_path(self):
        """没有路径的短链应不通过验证"""
        assert self.extractor._validate_url("https://xhslink.com/") is False


# ===== User-Agent 列表测试 =====


class TestUserAgents:
    """测试 User-Agent 配置"""

    def test_at_least_five_user_agents(self):
        """User-Agent 列表至少包含 5 个"""
        assert len(USER_AGENTS) >= 5

    def test_user_agents_are_strings(self):
        """所有 User-Agent 都是非空字符串"""
        for ua in USER_AGENTS:
            assert isinstance(ua, str)
            assert len(ua) > 0


# ===== 解析逻辑测试 =====


class TestParseNote:
    """测试 _parse_note 方法"""

    def setup_method(self):
        self.extractor = ContentExtractor()
        self.test_url = "https://www.xiaohongshu.com/explore/6501234567890abcdef12345"

    def _build_initial_state_html(self, note_data: dict) -> str:
        """构建包含 __INITIAL_STATE__ 的 HTML"""
        state = {
            "note": {
                "noteDetailMap": {
                    "6501234567890abcdef12345": {
                        "note": note_data
                    }
                }
            }
        }
        json_str = json.dumps(state, ensure_ascii=False)
        return f"<html><head><script>window.__INITIAL_STATE__={json_str}</script></head><body></body></html>"

    def test_parse_from_initial_state(self):
        """从 __INITIAL_STATE__ 中成功提取笔记内容"""
        note_data = {
            "title": "测试标题",
            "desc": "这是一段测试描述内容",
            "imageList": [
                {"urlDefault": "https://img.example.com/1.jpg"},
                {"url": "https://img.example.com/2.jpg"},
            ],
            "user": {"nickname": "测试用户"},
            "time": "2025-01-01 12:00:00",
            "interactInfo": {
                "likedCount": "100",
                "collectedCount": "50",
                "commentCount": "20",
            },
        }
        html = self._build_initial_state_html(note_data)
        result = self.extractor._parse_note(html, self.test_url)

        assert result is not None
        assert "测试标题" in result.text
        assert "测试描述内容" in result.text
        assert len(result.image_urls) == 2
        assert result.metadata.title == "测试标题"
        assert result.metadata.author == "测试用户"
        assert result.metadata.likes == 100
        assert result.metadata.collects == 50
        assert result.metadata.comments == 20

    def test_parse_with_wan_count(self):
        """正确解析 '万' 格式的数字"""
        note_data = {
            "title": "热门笔记",
            "desc": "内容",
            "imageList": [],
            "user": {"nickname": "作者"},
            "time": "2025-01-01",
            "interactInfo": {
                "likedCount": "1.2万",
                "collectedCount": "5000",
                "commentCount": "300",
            },
        }
        html = self._build_initial_state_html(note_data)
        result = self.extractor._parse_note(html, self.test_url)

        assert result is not None
        assert result.metadata.likes == 12000

    def test_parse_from_og_meta(self):
        """从 og:meta 标签中提取笔记内容（备选方案）"""
        html = """
        <html>
        <head>
            <meta property="og:title" content="OG标题测试" />
            <meta property="og:description" content="OG描述内容" />
            <meta name="author" content="OG作者" />
            <meta property="og:image" content="https://img.example.com/og.jpg" />
        </head>
        <body></body>
        </html>
        """
        result = self.extractor._parse_note(html, self.test_url)

        assert result is not None
        assert "OG标题测试" in result.text
        assert result.metadata.author == "OG作者"
        assert len(result.image_urls) >= 1

    def test_parse_empty_html(self):
        """空 HTML 应返回 None"""
        result = self.extractor._parse_note("<html><body></body></html>", self.test_url)
        assert result is None

    def test_parse_image_url_protocol_fix(self):
        """以 // 开头的图片 URL 应自动补全 https:"""
        note_data = {
            "title": "图片测试",
            "desc": "内容",
            "imageList": [
                {"urlDefault": "//img.example.com/no-protocol.jpg"},
            ],
            "user": {"nickname": "作者"},
            "time": "2025-01-01",
            "interactInfo": {},
        }
        html = self._build_initial_state_html(note_data)
        result = self.extractor._parse_note(html, self.test_url)

        assert result is not None
        assert result.image_urls[0] == "https://img.example.com/no-protocol.jpg"


# ===== 笔记不可用检测测试 =====


class TestNoteUnavailable:
    """测试 _is_note_unavailable 方法"""

    def setup_method(self):
        self.extractor = ContentExtractor()

    def test_deleted_note(self):
        """包含'笔记已删除'的页面应返回提示"""
        html = "<html><body><div>笔记已删除</div></body></html>"
        result = self.extractor._is_note_unavailable(html)
        assert result is not None
        assert "笔记已删除" in result

    def test_private_note(self):
        """包含'仅自己可见'的页面应返回提示"""
        html = "<html><body><div>仅自己可见</div></body></html>"
        result = self.extractor._is_note_unavailable(html)
        assert result is not None
        assert "仅自己可见" in result

    def test_normal_note(self):
        """正常笔记页面应返回 None"""
        html = "<html><body><div>正常内容</div></body></html>"
        result = self.extractor._is_note_unavailable(html)
        assert result is None


# ===== extract() 异步方法测试 =====


class TestExtract:
    """测试 extract() 主入口方法"""

    def setup_method(self):
        self.extractor = ContentExtractor()

    @pytest.mark.asyncio
    async def test_invalid_url_returns_error(self):
        """无效 URL 应返回 success=False 和错误信息"""
        result = await self.extractor.extract("https://www.baidu.com")
        assert result["success"] is False
        assert "链接格式不正确" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self):
        """空 URL 应返回 success=False 和错误信息"""
        result = await self.extractor.extract("")
        assert result["success"] is False
        assert "链接格式不正确" in result["error"]

    @pytest.mark.asyncio
    async def test_random_string_returns_error(self):
        """随机字符串应返回 success=False"""
        result = await self.extractor.extract("just some random text")
        assert result["success"] is False
        assert result["error"] is not None
        assert len(result["error"]) > 0

    @pytest.mark.asyncio
    async def test_network_failure_returns_error(self):
        """网络请求全部失败时应返回访问受限提示"""
        with patch.object(
            self.extractor, "_handle_anti_crawl", return_value=None
        ):
            url = "https://xhslink.com/abc123"
            result = await self.extractor.extract(url)
            assert result["success"] is False
            assert "访问受限" in result["error"]

    @pytest.mark.asyncio
    async def test_404_returns_not_found(self):
        """404 响应应返回笔记不存在提示"""
        mock_response = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", "https://xhslink.com/abc123"),
        )
        with patch.object(
            self.extractor, "_handle_anti_crawl", return_value=mock_response
        ):
            url = "https://xhslink.com/abc123"
            result = await self.extractor.extract(url)
            assert result["success"] is False
            assert "不存在" in result["error"]

    @pytest.mark.asyncio
    async def test_deleted_note_returns_error(self):
        """已删除笔记应返回无法访问提示"""
        html = "<html><body><div>笔记已删除</div></body></html>"
        mock_response = httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://xhslink.com/abc123"),
            text=html,
        )
        with patch.object(
            self.extractor, "_handle_anti_crawl", return_value=mock_response
        ):
            url = "https://xhslink.com/abc123"
            result = await self.extractor.extract(url)
            assert result["success"] is False
            assert "笔记已删除" in result["error"]

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        """成功提取笔记内容"""
        state = {
            "note": {
                "noteDetailMap": {
                    "abc": {
                        "note": {
                            "title": "成功提取",
                            "desc": "测试内容",
                            "imageList": [{"urlDefault": "https://img.test.com/1.jpg"}],
                            "user": {"nickname": "测试作者"},
                            "time": "2025-06-01",
                            "interactInfo": {
                                "likedCount": "10",
                                "collectedCount": "5",
                                "commentCount": "2",
                            },
                        }
                    }
                }
            }
        }
        html = f"<html><head><script>window.__INITIAL_STATE__={json.dumps(state)}</script></head><body></body></html>"
        mock_response = httpx.Response(
            status_code=200,
            request=httpx.Request("GET", "https://xhslink.com/abc123"),
            text=html,
        )
        with patch.object(
            self.extractor, "_handle_anti_crawl", return_value=mock_response
        ):
            url = "https://xhslink.com/abc123"
            result = await self.extractor.extract(url)
            assert result["success"] is True
            assert isinstance(result["data"], ExtractedContent)
            assert "成功提取" in result["data"].text
            assert result["data"].metadata.author == "测试作者"
            assert len(result["data"].image_urls) == 1


# ===== safe_int 辅助方法测试 =====


class TestSafeInt:
    """测试 _safe_int 静态方法"""

    def test_int_input(self):
        assert ContentExtractor._safe_int(42) == 42

    def test_string_number(self):
        assert ContentExtractor._safe_int("100") == 100

    def test_wan_format(self):
        assert ContentExtractor._safe_int("1.5万") == 15000

    def test_invalid_string(self):
        assert ContentExtractor._safe_int("abc") == 0

    def test_none_input(self):
        assert ContentExtractor._safe_int(None) == 0
