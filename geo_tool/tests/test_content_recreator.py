"""ContentRecreator 单元测试

测试内容二创引擎的核心功能：
- 模板自动选择
- 标题诊断
- 转化策略生成
- recreate 完整流程（使用 mock LLM）
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from geo_tool.models.content import ExtractedContent, NoteMetadata
from geo_tool.models.enums import BusinessType, DecisionStage, KeywordType, TemplateType
from geo_tool.models.keyword import GEOKeyword, KeywordMatrix
from geo_tool.models.title import TitleDiagnosis
from geo_tool.modules.content_recreator import ContentRecreator


# ===== 测试辅助工具 =====


def _make_content(title: str = "投影仪选购指南", text: str = "这是一篇关于投影仪的详细评测") -> ExtractedContent:
    """创建测试用的 ExtractedContent"""
    return ExtractedContent(
        url="https://www.xiaohongshu.com/explore/test123",
        text=text,
        image_urls=["https://example.com/img1.jpg"],
        metadata=NoteMetadata(
            title=title,
            author="测试用户",
            publish_time="2025-01-01",
            likes=100,
            collects=50,
            comments=20,
        ),
    )


def _make_keywords(
    core_topics: list[str] | None = None,
    words: list[str] | None = None,
) -> KeywordMatrix:
    """创建测试用的 KeywordMatrix"""
    if core_topics is None:
        core_topics = ["投影仪", "家用电器"]
    if words is None:
        words = ["投影仪", "家用投影仪", "投影仪推荐"]

    keywords = [
        GEOKeyword(
            word=w,
            keyword_type=KeywordType.COGNITION,
            decision_stage=DecisionStage.EVALUATION,
            business_value=500.0,
        )
        for w in words
    ]
    return KeywordMatrix(core_topics=core_topics, keywords=keywords)


def _make_mock_llm() -> MagicMock:
    """创建 mock LLM 客户端"""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="这是 LLM 生成的二创正文内容。")
    llm.chat_json = AsyncMock(return_value={
        "titles": [
            "2025年投影仪推荐榜单｜深度评测Top10",
            "2025年家用投影仪怎么选？专业攻略指南",
            "2025年投影仪对比评测｜真实体验分享必看",
            "2025年高端投影仪推荐｜专业选购指南",
            "2025年平价投影仪盘点｜性价比之选",
        ]
    })
    return llm


# ===== _select_template 测试 =====


class TestSelectTemplate:
    """测试模板自动选择功能"""

    def test_ranking_keywords(self):
        """包含榜单关键词时应推荐 RANKING 模板"""
        content = _make_content(title="2025年投影仪排行榜Top10推荐")
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert result == TemplateType.RANKING

    def test_review_keywords(self):
        """包含评测关键词时应推荐 REVIEW 模板"""
        content = _make_content(title="投影仪深度评测体验", text="这款投影仪实测效果非常好")
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert result == TemplateType.REVIEW

    def test_guide_keywords(self):
        """包含攻略关键词时应推荐 GUIDE 模板"""
        content = _make_content(title="投影仪选购攻略教程")
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert result == TemplateType.GUIDE

    def test_comparison_keywords(self):
        """包含对比关键词时应推荐 COMPARISON 模板"""
        content = _make_content(title="投影仪A vs 投影仪B对比")
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert result == TemplateType.COMPARISON

    def test_case_study_keywords(self):
        """包含案例关键词时应推荐 CASE_STUDY 模板"""
        content = _make_content(title="投影仪使用案例复盘经验分享")
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert result == TemplateType.CASE_STUDY

    def test_no_match_defaults_to_guide(self):
        """没有匹配关键词时应默认推荐 GUIDE 模板"""
        content = _make_content(title="随便写点什么", text="没有任何特征关键词的内容")
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert result == TemplateType.GUIDE

    def test_returns_valid_template_type(self):
        """返回值应为有效的 TemplateType"""
        content = _make_content()
        recreator = ContentRecreator(MagicMock())
        result = recreator._select_template(content)
        assert isinstance(result, TemplateType)
        assert result in list(TemplateType)


# ===== _diagnose_title 测试 =====


class TestDiagnoseTitle:
    """测试标题诊断功能"""

    def setup_method(self):
        self.recreator = ContentRecreator(MagicMock())
        self.keywords = _make_keywords()

    def test_full_match_title(self):
        """包含所有要素的标题应获得满分"""
        title = "2025年高端投影仪推荐榜单深度解析指南"  # 18字符，满足15-30
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert diag.has_core_keyword is True
        assert diag.has_intent_word is True
        assert diag.has_timeliness_word is True
        assert diag.has_attribute_word is True
        assert diag.length_valid is True
        assert diag.match_score == 100.0

    def test_missing_core_keyword(self):
        """缺少核心关键词时应扣分"""
        keywords = _make_keywords(core_topics=["量子计算"], words=["量子芯片"])
        title = "2025年高端产品推荐"
        diag = self.recreator._diagnose_title(title, keywords)
        assert diag.has_core_keyword is False
        assert diag.match_score < 100.0

    def test_missing_intent_word(self):
        """缺少意图词时应扣分"""
        title = "2025年高端投影仪产品"
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert diag.has_intent_word is False

    def test_missing_timeliness_word(self):
        """缺少时效词时应扣分"""
        title = "高端投影仪推荐榜单"
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert diag.has_timeliness_word is False

    def test_length_valid_range(self):
        """15-30字符的标题长度应合规"""
        title = "2025年投影仪推荐榜单指南"  # 13字符
        diag = self.recreator._diagnose_title(title, self.keywords)
        # 检查长度判断逻辑
        assert diag.length_valid == (15 <= len(title) <= 30)

    def test_too_short_title(self):
        """过短标题应标记长度不合规"""
        title = "投影仪推荐"  # 5字符
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert diag.length_valid is False
        assert any("过短" in s for s in diag.suggestions)

    def test_too_long_title(self):
        """过长标题应标记长度不合规"""
        title = "2025年最新高端家用投影仪推荐排行榜Top10深度评测对比分析完整版指南"
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert diag.length_valid is False
        assert any("过长" in s for s in diag.suggestions)

    def test_match_score_range(self):
        """匹配分数应在 0-100 范围内"""
        title = "随便一个标题"
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert 0.0 <= diag.match_score <= 100.0

    def test_diagnosis_has_all_fields(self):
        """诊断结果应包含所有必需字段"""
        title = "测试标题"
        diag = self.recreator._diagnose_title(title, self.keywords)
        assert isinstance(diag, TitleDiagnosis)
        assert isinstance(diag.has_core_keyword, bool)
        assert isinstance(diag.has_intent_word, bool)
        assert isinstance(diag.has_timeliness_word, bool)
        assert isinstance(diag.has_attribute_word, bool)
        assert isinstance(diag.length_valid, bool)
        assert isinstance(diag.match_score, float)
        assert isinstance(diag.suggestions, list)

    def test_suggestions_for_missing_elements(self):
        """缺少要素时应生成对应的优化建议"""
        keywords = _make_keywords(core_topics=["量子计算"], words=["量子芯片"])
        title = "一个普通标题"
        diag = self.recreator._diagnose_title(title, keywords)
        # 缺少所有要素，应有多条建议
        assert len(diag.suggestions) > 0


# ===== _generate_conversion_elements 测试 =====


class TestGenerateConversionElements:
    """测试转化策略生成功能"""

    @pytest.mark.asyncio
    async def test_online_business_contacts(self):
        """线上业务应推荐电话、官网、公众号、微信号"""
        recreator = ContentRecreator(MagicMock())
        strategy = await recreator._generate_conversion_elements(
            TemplateType.GUIDE, BusinessType.ONLINE
        )
        placements_text = " ".join(strategy.contact_placements)
        assert "电话" in placements_text
        assert "官网" in placements_text
        assert "公众号" in placements_text
        assert "微信" in placements_text

    @pytest.mark.asyncio
    async def test_local_business_contacts(self):
        """本地业务应推荐门店地址、营业时间、电话、地图导航"""
        recreator = ContentRecreator(MagicMock())
        strategy = await recreator._generate_conversion_elements(
            TemplateType.GUIDE, BusinessType.LOCAL
        )
        placements_text = " ".join(strategy.contact_placements)
        assert "门店地址" in placements_text or "门店" in placements_text
        assert "营业时间" in placements_text
        assert "电话" in placements_text
        assert "地图导航" in placements_text or "导航" in placements_text

    @pytest.mark.asyncio
    async def test_strategy_all_fields_non_empty(self):
        """转化策略的所有列表字段应非空"""
        recreator = ContentRecreator(MagicMock())
        for btype in BusinessType:
            strategy = await recreator._generate_conversion_elements(
                TemplateType.RANKING, btype
            )
            assert len(strategy.contact_placements) > 0
            assert len(strategy.trust_elements) > 0
            assert len(strategy.call_to_action) > 0
            assert len(strategy.platform_warnings) > 0


# ===== recreate 完整流程测试 =====


class TestRecreate:
    """测试 recreate 完整流程"""

    @pytest.mark.asyncio
    async def test_recreate_with_specified_template(self):
        """指定模板类型时应使用该模板"""
        llm = _make_mock_llm()
        recreator = ContentRecreator(llm)
        content = _make_content()
        keywords = _make_keywords()

        result = await recreator.recreate(
            content, keywords, template_type=TemplateType.RANKING
        )

        assert result.template_type == TemplateType.RANKING
        assert len(result.titles) >= 3
        assert len(result.body) > 0
        assert len(result.keywords_used) > 0
        assert result.conversion is None  # 未指定 business_type

    @pytest.mark.asyncio
    async def test_recreate_auto_template(self):
        """未指定模板类型时应自动推荐"""
        llm = _make_mock_llm()
        recreator = ContentRecreator(llm)
        content = _make_content(title="投影仪排行榜Top10推荐")
        keywords = _make_keywords()

        result = await recreator.recreate(content, keywords)

        assert result.template_type in list(TemplateType)
        assert len(result.titles) >= 3

    @pytest.mark.asyncio
    async def test_recreate_with_business_type(self):
        """指定业务类型时应生成转化策略"""
        llm = _make_mock_llm()
        recreator = ContentRecreator(llm)
        content = _make_content()
        keywords = _make_keywords()

        result = await recreator.recreate(
            content, keywords,
            template_type=TemplateType.GUIDE,
            business_type=BusinessType.ONLINE,
        )

        assert result.conversion is not None
        assert len(result.conversion.contact_placements) > 0

    @pytest.mark.asyncio
    async def test_recreate_titles_have_diagnosis(self):
        """每个候选标题应附带诊断结果"""
        llm = _make_mock_llm()
        recreator = ContentRecreator(llm)
        content = _make_content()
        keywords = _make_keywords()

        result = await recreator.recreate(
            content, keywords, template_type=TemplateType.REVIEW
        )

        for title in result.titles:
            assert title.diagnosis is not None
            assert isinstance(title.diagnosis.match_score, float)
            assert 0.0 <= title.diagnosis.match_score <= 100.0
