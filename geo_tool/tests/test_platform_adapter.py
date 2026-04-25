"""PlatformAdapter 单元测试

测试平台适配器的核心功能：
- 四大平台偏好配置的完整性
- 内容适配格式调整
- adapt() 返回数量与输入平台数量一致
- 每个 PlatformConfig 包含非空的推荐渠道、格式要求和注意事项
"""

from datetime import datetime

import pytest

from geo_tool.models.enums import Platform, TemplateType
from geo_tool.models.platform import PlatformConfig, PlatformContent
from geo_tool.models.recreated import RecreatedContent
from geo_tool.models.title import Title, TitleDiagnosis
from geo_tool.modules.platform_adapter import PlatformAdapter


# ===== 测试辅助工具 =====


def _make_content(body: str = "这是一篇测试正文内容") -> RecreatedContent:
    """创建测试用的 RecreatedContent"""
    return RecreatedContent(
        template_type=TemplateType.GUIDE,
        titles=[
            Title(
                text="2025年投影仪推荐攻略指南大全",
                diagnosis=TitleDiagnosis(
                    has_core_keyword=True,
                    has_intent_word=True,
                    has_timeliness_word=True,
                    has_attribute_word=False,
                    length_valid=True,
                    match_score=75.0,
                ),
            )
        ],
        body=body,
        keywords_used=["投影仪", "推荐"],
        created_at=datetime.now(),
    )


# ===== 平台配置完整性测试 =====


class TestGetPlatformConfig:
    """测试 _get_platform_config 方法"""

    def setup_method(self):
        self.adapter = PlatformAdapter()

    @pytest.mark.parametrize("platform", list(Platform))
    def test_all_platforms_have_config(self, platform: Platform):
        """每个平台都应有对应的偏好配置"""
        config = self.adapter._get_platform_config(platform)
        assert isinstance(config, PlatformConfig)
        assert config.platform == platform

    @pytest.mark.parametrize("platform", list(Platform))
    def test_config_has_non_empty_channels(self, platform: Platform):
        """每个平台的推荐渠道列表不为空"""
        config = self.adapter._get_platform_config(platform)
        assert len(config.recommended_channels) > 0
        for ch in config.recommended_channels:
            assert isinstance(ch, str)
            assert len(ch.strip()) > 0

    @pytest.mark.parametrize("platform", list(Platform))
    def test_config_has_non_empty_format_rules(self, platform: Platform):
        """每个平台的内容格式要求列表不为空"""
        config = self.adapter._get_platform_config(platform)
        assert len(config.content_format_rules) > 0
        for rule in config.content_format_rules:
            assert isinstance(rule, str)
            assert len(rule.strip()) > 0

    @pytest.mark.parametrize("platform", list(Platform))
    def test_config_has_non_empty_notes(self, platform: Platform):
        """每个平台的注意事项列表不为空"""
        config = self.adapter._get_platform_config(platform)
        assert len(config.notes) > 0
        for note in config.notes:
            assert isinstance(note, str)
            assert len(note.strip()) > 0

    def test_baidu_recommends_baidu_channels(self):
        """百度AI搜索应推荐百度生态渠道"""
        config = self.adapter._get_platform_config(Platform.BAIDU_AI)
        channels = config.recommended_channels
        assert any("百家号" in ch for ch in channels)
        assert any("百度百科" in ch for ch in channels)
        assert any("百度知道" in ch for ch in channels)

    def test_doubao_recommends_bytedance_channels(self):
        """豆包应推荐字节生态渠道"""
        config = self.adapter._get_platform_config(Platform.DOUBAO)
        channels = config.recommended_channels
        assert any("今日头条" in ch for ch in channels)
        assert any("抖音" in ch for ch in channels)

    def test_deepseek_recommends_tech_channels(self):
        """DeepSeek 应推荐技术社区渠道"""
        config = self.adapter._get_platform_config(Platform.DEEPSEEK)
        channels = config.recommended_channels
        assert any("CSDN" in ch for ch in channels)
        assert any("博客园" in ch for ch in channels)

    def test_yuanbao_recommends_wechat_channels(self):
        """腾讯元宝应推荐微信生态渠道"""
        config = self.adapter._get_platform_config(Platform.TENCENT_YUANBAO)
        channels = config.recommended_channels
        assert any("微信公众号" in ch for ch in channels)


# ===== 内容适配测试 =====


class TestAdaptContent:
    """测试 _adapt_content 方法"""

    def setup_method(self):
        self.adapter = PlatformAdapter()
        self.content = _make_content("# 测试标题\n\n这是正文内容。")

    def test_adapted_body_contains_original(self):
        """适配后的正文应包含原始正文"""
        config = self.adapter._get_platform_config(Platform.BAIDU_AI)
        result = self.adapter._adapt_content(self.content, config)
        assert "# 测试标题" in result.adapted_body
        assert "这是正文内容。" in result.adapted_body

    def test_adapted_body_has_platform_header(self):
        """适配后的正文应包含平台特定头部"""
        config = self.adapter._get_platform_config(Platform.DOUBAO)
        result = self.adapter._adapt_content(self.content, config)
        assert "豆包" in result.adapted_body
        assert "推荐发布渠道" in result.adapted_body

    def test_adapted_body_has_platform_footer(self):
        """适配后的正文应包含平台注意事项尾部"""
        config = self.adapter._get_platform_config(Platform.DEEPSEEK)
        result = self.adapter._adapt_content(self.content, config)
        assert "发布注意事项" in result.adapted_body

    def test_publish_suggestions_matches_config(self):
        """发布建议应与平台配置一致"""
        config = self.adapter._get_platform_config(Platform.TENCENT_YUANBAO)
        result = self.adapter._adapt_content(self.content, config)
        assert result.publish_suggestions == config
        assert result.platform == Platform.TENCENT_YUANBAO


# ===== adapt() 异步方法测试 =====


class TestAdapt:
    """测试 adapt() 异步方法"""

    def setup_method(self):
        self.adapter = PlatformAdapter()
        self.content = _make_content()

    @pytest.mark.asyncio
    async def test_adapt_returns_correct_count(self):
        """返回的 PlatformContent 数量应等于输入的 Platform 数量"""
        platforms = [Platform.BAIDU_AI, Platform.DOUBAO]
        results = await self.adapter.adapt(self.content, platforms)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_adapt_all_four_platforms(self):
        """四个平台全部适配时应返回4个结果"""
        platforms = list(Platform)
        results = await self.adapter.adapt(self.content, platforms)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_adapt_single_platform(self):
        """单个平台适配应返回1个结果"""
        results = await self.adapter.adapt(self.content, [Platform.DEEPSEEK])
        assert len(results) == 1
        assert results[0].platform == Platform.DEEPSEEK

    @pytest.mark.asyncio
    async def test_adapt_empty_platforms(self):
        """空平台列表应返回空结果"""
        results = await self.adapter.adapt(self.content, [])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_adapt_each_result_has_correct_platform(self):
        """每个结果的 platform 应与输入对应"""
        platforms = [Platform.BAIDU_AI, Platform.TENCENT_YUANBAO, Platform.DOUBAO]
        results = await self.adapter.adapt(self.content, platforms)
        result_platforms = [r.platform for r in results]
        assert result_platforms == platforms

    @pytest.mark.asyncio
    async def test_adapt_results_are_platform_content(self):
        """每个结果应为 PlatformContent 实例"""
        platforms = [Platform.BAIDU_AI, Platform.DEEPSEEK]
        results = await self.adapter.adapt(self.content, platforms)
        for result in results:
            assert isinstance(result, PlatformContent)
            assert isinstance(result.publish_suggestions, PlatformConfig)
            assert len(result.adapted_body) > 0
