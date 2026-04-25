"""QualityScorer 单元测试

测试质量评分器的核心功能：
- 结构化程度评估（规则引擎）
- 时效性评估（规则引擎）
- LLM 辅助维度评估（使用 mock）
- 加权总分计算
- 质量等级判定
- 降级模式（LLM 不可用）
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from geo_tool.models.enums import QualityLevel, TemplateType
from geo_tool.models.quality import DimensionScore, QualityReport
from geo_tool.models.recreated import RecreatedContent
from geo_tool.models.title import Title, TitleDiagnosis
from geo_tool.modules.quality_scorer import QualityScorer


# ===== 测试辅助工具 =====


def _make_content(body: str = "默认正文内容") -> RecreatedContent:
    """创建测试用的 RecreatedContent"""
    return RecreatedContent(
        template_type=TemplateType.GUIDE,
        titles=[
            Title(
                text="2025年投影仪推荐攻略",
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


def _make_well_structured_body() -> str:
    """创建结构化程度高的正文"""
    return (
        "# 2025年家用投影仪选购指南\n\n"
        "## 一、选购前的准备工作\n\n"
        "在选购投影仪之前，需要考虑以下几个关键因素：\n\n"
        "- 使用场景：卧室、客厅还是办公室\n"
        "- 预算范围：3000元以下、3000-5000元、5000元以上\n"
        "- 核心需求：画质、亮度、便携性\n"
        "- 安装方式：吊装、桌面、便携\n"
        "- 投射距离：短焦、中焦、长焦\n\n"
        "## 二、核心参数对比\n\n"
        "| 参数 | 入门级 | 中端 | 高端 |\n"
        "| --- | --- | --- | --- |\n"
        "| 亮度 | 500ANSI | 1000ANSI | 2000ANSI |\n"
        "| 分辨率 | 720P | 1080P | 4K |\n\n"
        "### 2.1 亮度选择建议\n\n"
        "**亮度是投影仪最重要的参数之一**。\n\n"
        "### 2.2 分辨率选择建议\n\n"
        "**1080P 是目前的主流选择**，4K 适合追求极致画质的用户。\n\n"
        "### 2.3 对比度说明\n\n"
        "**高对比度**能带来更好的画面层次感。\n"
    )


def _make_timely_body() -> str:
    """创建时效性高的正文"""
    return (
        "# 2025年最新投影仪推荐\n\n"
        "据2025年艾瑞咨询报告显示，家用投影仪市场在2025年Q1增长了35%。\n\n"
        "## 最新市场趋势\n\n"
        "2025年3月的最新数据表明，激光投影仪的市场份额持续上升。\n"
        "目前市场上最受欢迎的品牌包括极米、当贝和坚果。\n\n"
        "近期发布的新品中，极米H6在2025年上半年表现最为亮眼。\n"
    )


def _make_mock_llm(
    density_score: int = 70,
    authority_score: int = 60,
    practicality_score: int = 65,
) -> MagicMock:
    """创建 mock LLM 客户端，可自定义各维度返回分数"""
    llm = MagicMock()

    async def mock_chat_json(messages, **kwargs):
        content = messages[-1]["content"]
        if "信息密度" in messages[0]["content"]:
            return {
                "data_richness": min(40, int(density_score * 0.4)),
                "case_richness": min(30, int(density_score * 0.3)),
                "conciseness": min(30, int(density_score * 0.3)),
                "deductions": ["数据量偏少"],
                "suggestions": ["建议增加更多具体数据"],
            }
        elif "权威性" in messages[0]["content"]:
            return {
                "authority_reports": min(30, int(authority_score * 0.3)),
                "official_data": min(25, int(authority_score * 0.25)),
                "expert_identity": min(25, int(authority_score * 0.25)),
                "verifiable_info": min(20, int(authority_score * 0.2)),
                "deductions": ["缺少权威报告引用"],
                "suggestions": ["建议引用行业报告"],
            }
        elif "实用性" in messages[0]["content"]:
            return {
                "clear_conclusion": min(35, int(practicality_score * 0.35)),
                "actionable_advice": min(35, int(practicality_score * 0.35)),
                "decision_support": min(30, int(practicality_score * 0.3)),
                "deductions": ["结论不够明确"],
                "suggestions": ["建议提供更明确的结论"],
            }
        return {}

    llm.chat_json = AsyncMock(side_effect=mock_chat_json)
    return llm


# ===== _score_structure 测试 =====


class TestScoreStructure:
    """测试结构化程度评估"""

    def test_well_structured_content(self):
        """结构化程度高的内容应获得高分"""
        scorer = QualityScorer()
        body = _make_well_structured_body()
        result = scorer._score_structure(body)
        assert result.dimension == "结构化程度"
        assert result.weight == 0.30
        assert result.score >= 70.0

    def test_empty_content(self):
        """空内容应获得 0 分"""
        scorer = QualityScorer()
        result = scorer._score_structure("")
        assert result.score == 0.0
        assert len(result.deduction_reasons) > 0

    def test_only_headings(self):
        """仅有标题的内容应获得部分分数"""
        body = "# 主标题\n\n## 子标题1\n\n## 子标题2\n\n### 细分标题\n"
        scorer = QualityScorer()
        result = scorer._score_structure(body)
        assert result.score > 0.0
        assert result.score < 100.0

    def test_score_range(self):
        """评分应在 0-100 范围内"""
        scorer = QualityScorer()
        for body in ["", "简单文本", _make_well_structured_body()]:
            result = scorer._score_structure(body)
            assert 0.0 <= result.score <= 100.0

    def test_table_detection(self):
        """包含表格的内容应获得表格分数"""
        body = "| 列1 | 列2 |\n| --- | --- |\n| 值1 | 值2 |\n"
        scorer = QualityScorer()
        result = scorer._score_structure(body)
        assert result.score > 0.0

    def test_list_detection(self):
        """包含列表的内容应获得列表分数"""
        body = "- 项目1\n- 项目2\n- 项目3\n- 项目4\n- 项目5\n"
        scorer = QualityScorer()
        result = scorer._score_structure(body)
        assert result.score > 0.0


# ===== _score_timeliness 测试 =====


class TestScoreTimeliness:
    """测试时效性评估"""

    def test_timely_content(self):
        """时效性高的内容应获得高分"""
        scorer = QualityScorer()
        body = _make_timely_body()
        result = scorer._score_timeliness(body)
        assert result.dimension == "时效性"
        assert result.weight == 0.20
        assert result.score >= 60.0

    def test_no_time_markers(self):
        """没有时间标注的内容应获得低分"""
        scorer = QualityScorer()
        result = scorer._score_timeliness("这是一段没有任何时间标注的普通文本内容。")
        assert result.score < 30.0
        assert len(result.deduction_reasons) > 0

    def test_year_only(self):
        """仅有年份标注应获得部分分数"""
        scorer = QualityScorer()
        result = scorer._score_timeliness("2025年的投影仪市场发展迅速。")
        assert result.score > 0.0

    def test_score_range(self):
        """评分应在 0-100 范围内"""
        scorer = QualityScorer()
        for body in ["", "无时间标注", _make_timely_body()]:
            result = scorer._score_timeliness(body)
            assert 0.0 <= result.score <= 100.0


# ===== _calculate_weighted_total 测试 =====


class TestCalculateWeightedTotal:
    """测试加权总分计算"""

    def test_all_perfect_scores(self):
        """所有维度满分时加权总分应为 100"""
        scorer = QualityScorer()
        scores = [
            DimensionScore(dimension="结构化程度", weight=0.30, score=100.0),
            DimensionScore(dimension="信息密度", weight=0.25, score=100.0),
            DimensionScore(dimension="时效性", weight=0.20, score=100.0),
            DimensionScore(dimension="权威性与可信度", weight=0.15, score=100.0),
            DimensionScore(dimension="实用性", weight=0.10, score=100.0),
        ]
        total = scorer._calculate_weighted_total(scores)
        assert total == 100.0

    def test_all_zero_scores(self):
        """所有维度 0 分时加权总分应为 0"""
        scorer = QualityScorer()
        scores = [
            DimensionScore(dimension="结构化程度", weight=0.30, score=0.0),
            DimensionScore(dimension="信息密度", weight=0.25, score=0.0),
            DimensionScore(dimension="时效性", weight=0.20, score=0.0),
            DimensionScore(dimension="权威性与可信度", weight=0.15, score=0.0),
            DimensionScore(dimension="实用性", weight=0.10, score=0.0),
        ]
        total = scorer._calculate_weighted_total(scores)
        assert total == 0.0

    def test_weighted_calculation(self):
        """加权总分应等于各维度 score × weight 之和"""
        scorer = QualityScorer()
        scores = [
            DimensionScore(dimension="结构化程度", weight=0.30, score=80.0),
            DimensionScore(dimension="信息密度", weight=0.25, score=70.0),
            DimensionScore(dimension="时效性", weight=0.20, score=60.0),
            DimensionScore(dimension="权威性与可信度", weight=0.15, score=50.0),
            DimensionScore(dimension="实用性", weight=0.10, score=90.0),
        ]
        total = scorer._calculate_weighted_total(scores)
        expected = 80 * 0.30 + 70 * 0.25 + 60 * 0.20 + 50 * 0.15 + 90 * 0.10
        assert total == round(expected, 2)


# ===== 质量等级判定测试 =====


class TestDetermineQualityLevel:
    """测试质量等级判定"""

    def test_needs_work(self):
        """加权总分 < 70 应为 needs_work"""
        assert QualityScorer._determine_quality_level(0.0) == QualityLevel.NEEDS_WORK
        assert QualityScorer._determine_quality_level(50.0) == QualityLevel.NEEDS_WORK
        assert QualityScorer._determine_quality_level(69.99) == QualityLevel.NEEDS_WORK

    def test_good(self):
        """70 <= 加权总分 < 85 应为 good"""
        assert QualityScorer._determine_quality_level(70.0) == QualityLevel.GOOD
        assert QualityScorer._determine_quality_level(75.0) == QualityLevel.GOOD
        assert QualityScorer._determine_quality_level(84.99) == QualityLevel.GOOD

    def test_excellent(self):
        """加权总分 >= 85 应为 excellent"""
        assert QualityScorer._determine_quality_level(85.0) == QualityLevel.EXCELLENT
        assert QualityScorer._determine_quality_level(90.0) == QualityLevel.EXCELLENT
        assert QualityScorer._determine_quality_level(100.0) == QualityLevel.EXCELLENT


# ===== score 完整流程测试 =====


class TestScore:
    """测试 score 完整流程"""

    @pytest.mark.asyncio
    async def test_score_with_llm(self):
        """有 LLM 时应返回完整的五维度评分"""
        llm = _make_mock_llm()
        scorer = QualityScorer(llm_client=llm)
        content = _make_content(body=_make_well_structured_body())

        report = await scorer.score(content)

        assert isinstance(report, QualityReport)
        assert len(report.dimension_scores) == 5
        assert 0.0 <= report.weighted_total <= 100.0
        assert report.quality_level in list(QualityLevel)

        # 验证五个维度名称
        dim_names = [d.dimension for d in report.dimension_scores]
        assert "结构化程度" in dim_names
        assert "信息密度" in dim_names
        assert "时效性" in dim_names
        assert "权威性与可信度" in dim_names
        assert "实用性" in dim_names

    @pytest.mark.asyncio
    async def test_score_without_llm_degraded(self):
        """无 LLM 时应降级，LLM 维度标记为无法评估"""
        scorer = QualityScorer(llm_client=None)
        content = _make_content(body=_make_well_structured_body())

        report = await scorer.score(content)

        assert isinstance(report, QualityReport)
        assert len(report.dimension_scores) == 5

        # 结构化和时效性应正常评分
        structure = next(
            d for d in report.dimension_scores if d.dimension == "结构化程度"
        )
        assert structure.score > 0.0

        # LLM 维度应为 0 分且有降级提示
        density = next(
            d for d in report.dimension_scores if d.dimension == "信息密度"
        )
        assert density.score == 0.0
        assert any("LLM 不可用" in r for r in density.deduction_reasons)

    @pytest.mark.asyncio
    async def test_score_weights_correct(self):
        """五个维度权重应分别为 0.30/0.25/0.20/0.15/0.10"""
        llm = _make_mock_llm()
        scorer = QualityScorer(llm_client=llm)
        content = _make_content(body=_make_well_structured_body())

        report = await scorer.score(content)

        expected_weights = {
            "结构化程度": 0.30,
            "信息密度": 0.25,
            "时效性": 0.20,
            "权威性与可信度": 0.15,
            "实用性": 0.10,
        }
        for dim_score in report.dimension_scores:
            assert dim_score.weight == expected_weights[dim_score.dimension]

    @pytest.mark.asyncio
    async def test_needs_work_has_improvements(self):
        """needs_work 等级应包含优先改进项"""
        # 使用空内容确保低分
        scorer = QualityScorer(llm_client=None)
        content = _make_content(body="简单文本")

        report = await scorer.score(content)

        assert report.quality_level == QualityLevel.NEEDS_WORK
        assert len(report.priority_improvements) > 0

    @pytest.mark.asyncio
    async def test_good_or_excellent_no_improvements(self):
        """good 或 excellent 等级不应包含优先改进项"""
        # 使用高分 mock
        llm = _make_mock_llm(
            density_score=90, authority_score=85, practicality_score=90
        )
        scorer = QualityScorer(llm_client=llm)
        # 使用结构化和时效性都高的内容
        body = _make_well_structured_body() + "\n" + _make_timely_body()
        content = _make_content(body=body)

        report = await scorer.score(content)

        if report.quality_level in (QualityLevel.GOOD, QualityLevel.EXCELLENT):
            assert len(report.priority_improvements) == 0
