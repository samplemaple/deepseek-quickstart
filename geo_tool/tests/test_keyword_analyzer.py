"""KeywordAnalyzer 单元测试

测试 GEO 关键词分析器的核心行为：
- 核心主题词识别
- 关键词矩阵生成（五种类型 + 四种决策阶段）
- 商业价值评分计算
- 长尾关键词生成
- analyze() 完整流程
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from geo_tool.models.content import ExtractedContent, NoteMetadata
from geo_tool.models.enums import DecisionStage, KeywordType
from geo_tool.models.keyword import GEOKeyword, KeywordMatrix
from geo_tool.modules.keyword_analyzer import KeywordAnalyzer
from geo_tool.modules.llm_client import LLMClient


# ===== 测试辅助工具 =====


def _make_content(text: str = "这是一篇关于投影仪选购的测试笔记", title: str = "2025年家用投影仪推荐") -> ExtractedContent:
    """创建测试用 ExtractedContent"""
    return ExtractedContent(
        url="https://www.xiaohongshu.com/explore/test123",
        text=text,
        image_urls=[],
        metadata=NoteMetadata(
            title=title,
            author="测试用户",
            publish_time="2025-01-01",
            likes=100,
            collects=50,
            comments=20,
        ),
    )


def _make_analyzer(llm_mock: AsyncMock | None = None) -> tuple[KeywordAnalyzer, LLMClient]:
    """创建测试用 KeywordAnalyzer 和 mock LLMClient"""
    client = LLMClient(api_key="test-key")
    if llm_mock is not None:
        client.chat_json = llm_mock
    analyzer = KeywordAnalyzer(client)
    return analyzer, client


# ===== 初始化测试 =====


class TestKeywordAnalyzerInit:
    """测试 KeywordAnalyzer 初始化"""

    def test_accepts_llm_client(self):
        """应接受 LLMClient 实例"""
        client = LLMClient(api_key="test-key")
        analyzer = KeywordAnalyzer(client)
        assert analyzer._llm is client


# ===== _identify_core_topics 测试 =====


class TestIdentifyCoreTopics:
    """测试核心主题词识别"""

    @pytest.mark.asyncio
    async def test_returns_topic_list(self):
        """应返回核心主题词列表"""
        mock_json = AsyncMock(return_value={"core_topics": ["投影仪", "家用", "性价比"]})
        analyzer, _ = _make_analyzer(mock_json)

        topics = await analyzer._identify_core_topics("测试内容")
        assert topics == ["投影仪", "家用", "性价比"]

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """LLM 返回空列表时应返回空列表"""
        mock_json = AsyncMock(return_value={"core_topics": []})
        analyzer, _ = _make_analyzer(mock_json)

        topics = await analyzer._identify_core_topics("测试内容")
        assert topics == []

    @pytest.mark.asyncio
    async def test_filters_none_values(self):
        """应过滤掉 None 值"""
        mock_json = AsyncMock(return_value={"core_topics": ["投影仪", None, "家用"]})
        analyzer, _ = _make_analyzer(mock_json)

        topics = await analyzer._identify_core_topics("测试内容")
        assert topics == ["投影仪", "家用"]


# ===== _generate_keyword_matrix 测试 =====


class TestGenerateKeywordMatrix:
    """测试关键词矩阵生成"""

    @pytest.mark.asyncio
    async def test_generates_keywords_with_types(self):
        """应生成带类型和决策阶段的关键词"""
        mock_json = AsyncMock(return_value={
            "keywords": [
                {"word": "高端投影仪", "keyword_type": "location", "decision_stage": "awareness"},
                {"word": "投影仪怎么选", "keyword_type": "cognition", "decision_stage": "exploration"},
                {"word": "办公投影仪", "keyword_type": "scenario", "decision_stage": "evaluation"},
                {"word": "实测投影仪", "keyword_type": "trust", "decision_stage": "decision"},
                {"word": "极米投影仪", "keyword_type": "brand", "decision_stage": "decision"},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = await analyzer._generate_keyword_matrix(["投影仪"], "测试内容")

        assert len(matrix.keywords) == 5
        types = {kw.keyword_type for kw in matrix.keywords}
        assert types == {
            KeywordType.LOCATION,
            KeywordType.COGNITION,
            KeywordType.SCENARIO,
            KeywordType.TRUST,
            KeywordType.BRAND,
        }

    @pytest.mark.asyncio
    async def test_preserves_core_topics(self):
        """应保留核心主题词"""
        mock_json = AsyncMock(return_value={"keywords": []})
        analyzer, _ = _make_analyzer(mock_json)

        matrix = await analyzer._generate_keyword_matrix(["投影仪", "家用"], "测试内容")
        assert matrix.core_topics == ["投影仪", "家用"]

    @pytest.mark.asyncio
    async def test_skips_invalid_keywords(self):
        """应跳过无效关键词（空 word）"""
        mock_json = AsyncMock(return_value={
            "keywords": [
                {"word": "", "keyword_type": "location", "decision_stage": "awareness"},
                {"word": "有效词", "keyword_type": "cognition", "decision_stage": "exploration"},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = await analyzer._generate_keyword_matrix(["测试"], "测试内容")
        assert len(matrix.keywords) == 1
        assert matrix.keywords[0].word == "有效词"

    @pytest.mark.asyncio
    async def test_defaults_invalid_type_to_cognition(self):
        """无效的关键词类型应默认为 cognition"""
        mock_json = AsyncMock(return_value={
            "keywords": [
                {"word": "测试词", "keyword_type": "invalid_type", "decision_stage": "awareness"},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = await analyzer._generate_keyword_matrix(["测试"], "测试内容")
        assert matrix.keywords[0].keyword_type == KeywordType.COGNITION

    @pytest.mark.asyncio
    async def test_defaults_invalid_stage_to_awareness(self):
        """无效的决策阶段应默认为 awareness"""
        mock_json = AsyncMock(return_value={
            "keywords": [
                {"word": "测试词", "keyword_type": "location", "decision_stage": "invalid_stage"},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = await analyzer._generate_keyword_matrix(["测试"], "测试内容")
        assert matrix.keywords[0].decision_stage == DecisionStage.AWARENESS


# ===== _calculate_business_value 测试 =====


class TestCalculateBusinessValue:
    """测试商业价值评分计算"""

    @pytest.mark.asyncio
    async def test_calculates_value_correctly(self):
        """应正确计算 需求成熟度 × 商业潜力 × 匹配精准度"""
        mock_json = AsyncMock(return_value={
            "scores": [
                {"word": "投影仪", "maturity": 8, "potential": 7, "precision": 9},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(
            core_topics=["投影仪"],
            keywords=[
                GEOKeyword(
                    word="投影仪",
                    keyword_type=KeywordType.COGNITION,
                    decision_stage=DecisionStage.AWARENESS,
                    business_value=0.0,
                ),
            ],
        )

        result = await analyzer._calculate_business_value(matrix, "测试内容")
        assert result.keywords[0].business_value == 8 * 7 * 9  # 504

    @pytest.mark.asyncio
    async def test_clamps_values_to_valid_range(self):
        """评分因子应被限制在 1-10 范围内"""
        mock_json = AsyncMock(return_value={
            "scores": [
                {"word": "测试词", "maturity": 15, "potential": 0, "precision": -1},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(
            core_topics=["测试"],
            keywords=[
                GEOKeyword(
                    word="测试词",
                    keyword_type=KeywordType.COGNITION,
                    decision_stage=DecisionStage.AWARENESS,
                    business_value=0.0,
                ),
            ],
        )

        result = await analyzer._calculate_business_value(matrix, "测试内容")
        # maturity=10(clamped), potential=1(clamped), precision=1(clamped) → 10
        assert result.keywords[0].business_value == 10.0

    @pytest.mark.asyncio
    async def test_default_value_for_missing_keyword(self):
        """LLM 未返回评分的关键词应使用默认值 125"""
        mock_json = AsyncMock(return_value={"scores": []})
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(
            core_topics=["测试"],
            keywords=[
                GEOKeyword(
                    word="未评分词",
                    keyword_type=KeywordType.COGNITION,
                    decision_stage=DecisionStage.AWARENESS,
                    business_value=0.0,
                ),
            ],
        )

        result = await analyzer._calculate_business_value(matrix, "测试内容")
        assert result.keywords[0].business_value == 125.0

    @pytest.mark.asyncio
    async def test_empty_keywords_returns_unchanged(self):
        """空关键词列表应直接返回"""
        mock_json = AsyncMock()
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(core_topics=["测试"], keywords=[])
        result = await analyzer._calculate_business_value(matrix, "测试内容")
        assert result.keywords == []
        mock_json.assert_not_called()


# ===== _generate_longtail_keywords 测试 =====


class TestGenerateLongtailKeywords:
    """测试长尾关键词生成"""

    @pytest.mark.asyncio
    async def test_generates_longtail_variants(self):
        """应为关键词生成长尾变体"""
        variants = [f"长尾变体{i}" for i in range(12)]
        mock_json = AsyncMock(return_value={
            "longtail": [
                {"word": "投影仪", "variants": variants},
            ]
        })
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(
            core_topics=["投影仪"],
            keywords=[
                GEOKeyword(
                    word="投影仪",
                    keyword_type=KeywordType.COGNITION,
                    decision_stage=DecisionStage.AWARENESS,
                    business_value=100.0,
                ),
            ],
        )

        result = await analyzer._generate_longtail_keywords(matrix, "测试内容")
        assert len(result.keywords[0].longtail_variants) == 12

    @pytest.mark.asyncio
    async def test_empty_keywords_returns_unchanged(self):
        """空关键词列表应直接返回"""
        mock_json = AsyncMock()
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(core_topics=["测试"], keywords=[])
        result = await analyzer._generate_longtail_keywords(matrix, "测试内容")
        assert result.keywords == []
        mock_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_keyword_gets_empty_variants(self):
        """LLM 未返回长尾变体的关键词应得到空列表"""
        mock_json = AsyncMock(return_value={"longtail": []})
        analyzer, _ = _make_analyzer(mock_json)

        matrix = KeywordMatrix(
            core_topics=["测试"],
            keywords=[
                GEOKeyword(
                    word="未覆盖词",
                    keyword_type=KeywordType.COGNITION,
                    decision_stage=DecisionStage.AWARENESS,
                    business_value=100.0,
                ),
            ],
        )

        result = await analyzer._generate_longtail_keywords(matrix, "测试内容")
        assert result.keywords[0].longtail_variants == []


# ===== _parse_keyword 测试 =====


class TestParseKeyword:
    """测试关键词解析"""

    def test_valid_keyword(self):
        """有效关键词应正确解析"""
        analyzer, _ = _make_analyzer()
        result = analyzer._parse_keyword({
            "word": "投影仪",
            "keyword_type": "location",
            "decision_stage": "evaluation",
        })
        assert result is not None
        assert result.word == "投影仪"
        assert result.keyword_type == KeywordType.LOCATION
        assert result.decision_stage == DecisionStage.EVALUATION

    def test_empty_word_returns_none(self):
        """空 word 应返回 None"""
        analyzer, _ = _make_analyzer()
        result = analyzer._parse_keyword({"word": "", "keyword_type": "location", "decision_stage": "awareness"})
        assert result is None

    def test_missing_word_returns_none(self):
        """缺少 word 字段应返回 None"""
        analyzer, _ = _make_analyzer()
        result = analyzer._parse_keyword({"keyword_type": "location", "decision_stage": "awareness"})
        assert result is None


# ===== _clamp 测试 =====


class TestClamp:
    """测试数值限制"""

    def test_value_within_range(self):
        """范围内的值应不变"""
        assert KeywordAnalyzer._clamp(5, 1, 10) == 5.0

    def test_value_below_min(self):
        """低于最小值应返回最小值"""
        assert KeywordAnalyzer._clamp(-1, 1, 10) == 1.0

    def test_value_above_max(self):
        """高于最大值应返回最大值"""
        assert KeywordAnalyzer._clamp(15, 1, 10) == 10.0

    def test_invalid_value_returns_min(self):
        """无效值应返回最小值"""
        assert KeywordAnalyzer._clamp("abc", 1, 10) == 1.0


# ===== analyze() 完整流程测试 =====


class TestAnalyze:
    """测试 analyze() 完整流程"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """完整流程应返回按商业价值排序的关键词矩阵"""
        call_count = 0

        async def mock_chat_json(messages, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # 第一步：识别核心主题词
                return {"core_topics": ["投影仪", "家用"]}
            elif call_count == 2:
                # 第二步：生成关键词矩阵
                return {
                    "keywords": [
                        {"word": "高端投影仪", "keyword_type": "location", "decision_stage": "awareness"},
                        {"word": "投影仪怎么选", "keyword_type": "cognition", "decision_stage": "exploration"},
                        {"word": "办公投影仪", "keyword_type": "scenario", "decision_stage": "evaluation"},
                    ]
                }
            elif call_count == 3:
                # 第三步：计算商业价值
                return {
                    "scores": [
                        {"word": "高端投影仪", "maturity": 6, "potential": 8, "precision": 7},
                        {"word": "投影仪怎么选", "maturity": 9, "potential": 7, "precision": 8},
                        {"word": "办公投影仪", "maturity": 5, "potential": 6, "precision": 5},
                    ]
                }
            elif call_count == 4:
                # 第四步：生成长尾关键词
                return {
                    "longtail": [
                        {"word": "高端投影仪", "variants": [f"高端投影仪变体{i}" for i in range(10)]},
                        {"word": "投影仪怎么选", "variants": [f"投影仪怎么选变体{i}" for i in range(10)]},
                        {"word": "办公投影仪", "variants": [f"办公投影仪变体{i}" for i in range(10)]},
                    ]
                }
            return {}

        client = LLMClient(api_key="test-key")
        client.chat_json = AsyncMock(side_effect=mock_chat_json)
        analyzer = KeywordAnalyzer(client)

        content = _make_content()
        result = await analyzer.analyze(content)

        # 验证核心主题词
        assert result.core_topics == ["投影仪", "家用"]

        # 验证关键词数量
        assert len(result.keywords) == 3

        # 验证按商业价值降序排序
        values = [kw.business_value for kw in result.keywords]
        assert values == sorted(values, reverse=True)

        # 验证具体商业价值
        # 投影仪怎么选: 9*7*8=504, 高端投影仪: 6*8*7=336, 办公投影仪: 5*6*5=150
        assert result.keywords[0].word == "投影仪怎么选"
        assert result.keywords[0].business_value == 504.0
        assert result.keywords[1].word == "高端投影仪"
        assert result.keywords[1].business_value == 336.0
        assert result.keywords[2].word == "办公投影仪"
        assert result.keywords[2].business_value == 150.0

        # 验证长尾变体
        for kw in result.keywords:
            assert len(kw.longtail_variants) >= 10

        # 验证调用了4次 LLM
        assert call_count == 4
