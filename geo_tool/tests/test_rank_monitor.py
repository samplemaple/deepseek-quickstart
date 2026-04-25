"""RankMonitor 单元测试

测试排名监测器的核心功能：
- _detect_citation() 文本匹配检测
- _diagnose_absence() 四步诊断法
- check_ranking() 组合流程
- 平台不可达时的降级处理
"""

from unittest.mock import AsyncMock, patch

import pytest

from geo_tool.models.enums import Platform
from geo_tool.models.ranking import CitationInfo, DiagnosisReport, RankingResult
from geo_tool.modules.rank_monitor import RankMonitor


# ===== 测试辅助工具 =====


def _make_ai_response_with_title(title: str) -> str:
    """创建包含指定标题的模拟 AI 回答"""
    return (
        f"根据搜索结果，关于该话题的相关信息如下：\n\n"
        f"1. {title}\n"
        f"2. 其他相关内容推荐\n"
        f"3. 更多参考资料\n"
    )


def _make_ai_response_without_title() -> str:
    """创建不包含特定标题的模拟 AI 回答"""
    return (
        "根据搜索结果，关于该话题的相关信息如下：\n\n"
        "1. 这是一篇关于技术趋势的文章\n"
        "2. 另一篇关于行业分析的内容\n"
        "3. 更多参考资料\n"
    )


# ===== _detect_citation 测试 =====


class TestDetectCitation:
    """测试引用检测方法"""

    def setup_method(self):
        self.monitor = RankMonitor()

    def test_detect_full_title_match(self):
        """完整标题匹配时应返回引用信息"""
        title = "2025年最值得买的投影仪推荐"
        response = _make_ai_response_with_title(title)
        result = self.monitor._detect_citation(response, title)
        assert result is not None
        assert result.is_cited is True
        assert result.position is not None
        assert result.cited_snippet is not None

    def test_detect_no_match(self):
        """标题未出现在回答中时应返回 None"""
        title = "2025年最值得买的投影仪推荐"
        response = _make_ai_response_without_title()
        result = self.monitor._detect_citation(response, title)
        assert result is None

    def test_detect_case_insensitive(self):
        """匹配应不区分大小写"""
        title = "Best Python Framework 2025"
        response = "推荐内容：\n1. best python framework 2025 是热门话题\n"
        result = self.monitor._detect_citation(response, title)
        assert result is not None
        assert result.is_cited is True

    def test_detect_empty_response(self):
        """空回答应返回 None"""
        result = self.monitor._detect_citation("", "测试标题")
        assert result is None

    def test_detect_empty_title(self):
        """空标题应返回 None"""
        result = self.monitor._detect_citation("一些回答内容", "")
        assert result is None

    def test_detect_none_inputs(self):
        """None 输入应返回 None"""
        result = self.monitor._detect_citation(None, "标题")
        assert result is None
        result = self.monitor._detect_citation("回答", None)
        assert result is None

    def test_detect_position_from_numbered_list(self):
        """应正确识别编号列表中的引用位置"""
        title = "投影仪选购指南"
        response = (
            "推荐内容：\n"
            "1. 第一篇文章\n"
            "2. 投影仪选购指南\n"
            "3. 第三篇文章\n"
        )
        result = self.monitor._detect_citation(response, title)
        assert result is not None
        assert result.position == 2



# ===== _diagnose_absence 测试 =====


class TestDiagnoseAbsence:
    """测试四步诊断法"""

    def setup_method(self):
        self.monitor = RankMonitor()

    @pytest.mark.asyncio
    async def test_diagnosis_returns_report(self):
        """诊断应返回完整的 DiagnosisReport"""
        report = await self.monitor._diagnose_absence(
            content_title="2025年投影仪推荐",
            keyword="投影仪推荐",
            platform=Platform.BAIDU_AI,
        )
        assert isinstance(report, DiagnosisReport)
        assert report.diagnosis_summary != ""
        assert report.optimization_strategy in ("小修小补", "大改重发", "推倒重来")
        assert len(report.action_items) > 0

    @pytest.mark.asyncio
    async def test_diagnosis_title_match_true(self):
        """标题包含关键词时 title_match 应为 True"""
        report = await self.monitor._diagnose_absence(
            content_title="2025年投影仪推荐攻略",
            keyword="投影仪推荐",
            platform=Platform.DOUBAO,
        )
        assert report.title_match is True

    @pytest.mark.asyncio
    async def test_diagnosis_title_match_false(self):
        """标题不包含关键词时 title_match 应为 False"""
        report = await self.monitor._diagnose_absence(
            content_title="家用设备选购指南",
            keyword="投影仪推荐",
            platform=Platform.DEEPSEEK,
        )
        assert report.title_match is False
        # 应包含标题优化建议
        has_title_suggestion = any(
            "关键词" in item for item in report.action_items
        )
        assert has_title_suggestion

    @pytest.mark.asyncio
    async def test_diagnosis_strategy_for_title_match(self):
        """标题匹配时优化策略应为小修小补"""
        report = await self.monitor._diagnose_absence(
            content_title="投影仪推荐2025",
            keyword="投影仪推荐",
            platform=Platform.BAIDU_AI,
        )
        assert report.optimization_strategy == "小修小补"

    @pytest.mark.asyncio
    async def test_diagnosis_strategy_for_title_mismatch(self):
        """标题不匹配时优化策略应为大改重发"""
        report = await self.monitor._diagnose_absence(
            content_title="家用设备选购",
            keyword="投影仪推荐",
            platform=Platform.TENCENT_YUANBAO,
        )
        assert report.optimization_strategy == "大改重发"

    @pytest.mark.asyncio
    async def test_diagnosis_summary_contains_keyword(self):
        """诊断摘要应包含关键词信息"""
        report = await self.monitor._diagnose_absence(
            content_title="测试标题",
            keyword="测试关键词",
            platform=Platform.DOUBAO,
        )
        assert "测试关键词" in report.diagnosis_summary

    @pytest.mark.asyncio
    async def test_diagnosis_summary_contains_platform(self):
        """诊断摘要应包含平台名称"""
        report = await self.monitor._diagnose_absence(
            content_title="测试标题",
            keyword="测试关键词",
            platform=Platform.BAIDU_AI,
        )
        assert "百度AI搜索" in report.diagnosis_summary


# ===== check_ranking 测试 =====


class TestCheckRanking:
    """测试 check_ranking() 异步方法"""

    def setup_method(self):
        self.monitor = RankMonitor()

    @pytest.mark.asyncio
    async def test_check_ranking_cited(self):
        """内容被引用时应返回 is_cited=True 且 position 和 snippet 非空"""
        title = "2025年投影仪推荐攻略"
        mock_response = _make_ai_response_with_title(title)

        with patch.object(
            self.monitor, "_query_ai_platform", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_response
            result = await self.monitor.check_ranking(
                keyword="投影仪推荐",
                platform=Platform.BAIDU_AI,
                content_title=title,
            )

        assert isinstance(result, RankingResult)
        assert result.citation.is_cited is True
        assert result.citation.position is not None
        assert result.citation.cited_snippet is not None
        assert result.diagnosis is None

    @pytest.mark.asyncio
    async def test_check_ranking_not_cited(self):
        """内容未被引用时应返回 is_cited=False 且 diagnosis 非空"""
        mock_response = _make_ai_response_without_title()

        with patch.object(
            self.monitor, "_query_ai_platform", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_response
            result = await self.monitor.check_ranking(
                keyword="投影仪推荐",
                platform=Platform.DOUBAO,
                content_title="2025年投影仪推荐攻略",
            )

        assert isinstance(result, RankingResult)
        assert result.citation.is_cited is False
        assert result.diagnosis is not None
        assert result.diagnosis.diagnosis_summary != ""
        assert result.diagnosis.optimization_strategy in (
            "小修小补", "大改重发", "推倒重来"
        )
        assert len(result.diagnosis.action_items) > 0

    @pytest.mark.asyncio
    async def test_check_ranking_platform_unreachable(self):
        """平台不可达时应返回 is_cited=False 并在诊断中说明"""
        with patch.object(
            self.monitor, "_query_ai_platform", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = None  # 模拟平台不可达
            result = await self.monitor.check_ranking(
                keyword="投影仪推荐",
                platform=Platform.DEEPSEEK,
                content_title="2025年投影仪推荐攻略",
            )

        assert isinstance(result, RankingResult)
        assert result.citation.is_cited is False
        assert result.diagnosis is not None
        assert "不可用" in result.diagnosis.diagnosis_summary or "暂不可用" in result.diagnosis.diagnosis_summary
        assert len(result.diagnosis.action_items) > 0

    @pytest.mark.asyncio
    async def test_check_ranking_result_fields(self):
        """返回结果应包含正确的 keyword 和 platform 字段"""
        with patch.object(
            self.monitor, "_query_ai_platform", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = _make_ai_response_without_title()
            result = await self.monitor.check_ranking(
                keyword="测试关键词",
                platform=Platform.TENCENT_YUANBAO,
                content_title="测试标题",
            )

        assert result.keyword == "测试关键词"
        assert result.platform == Platform.TENCENT_YUANBAO
        assert result.checked_at is not None

    @pytest.mark.asyncio
    async def test_check_ranking_all_platforms(self):
        """所有平台都应能正常返回结果"""
        for platform in Platform:
            with patch.object(
                self.monitor, "_query_ai_platform", new_callable=AsyncMock
            ) as mock_query:
                mock_query.return_value = _make_ai_response_without_title()
                result = await self.monitor.check_ranking(
                    keyword="测试",
                    platform=platform,
                    content_title="测试标题",
                )
            assert isinstance(result, RankingResult)
            assert result.platform == platform
