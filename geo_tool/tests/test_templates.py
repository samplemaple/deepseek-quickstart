"""GEO内容模板模块测试

验证五种模板的定义完整性和 get_template 函数的正确性。
"""

import pytest

from geo_tool.models.enums import TemplateType
from geo_tool.modules.templates import (
    RANKING_TEMPLATE,
    REVIEW_TEMPLATE,
    GUIDE_TEMPLATE,
    COMPARISON_TEMPLATE,
    CASE_STUDY_TEMPLATE,
    get_template,
)


class TestGetTemplate:
    """测试 get_template 函数"""

    def test_get_ranking_template(self):
        """验证能正确获取榜单式模板"""
        template = get_template(TemplateType.RANKING)
        assert template == RANKING_TEMPLATE
        assert isinstance(template, str)
        assert len(template) > 0

    def test_get_review_template(self):
        """验证能正确获取评测类模板"""
        template = get_template(TemplateType.REVIEW)
        assert template == REVIEW_TEMPLATE

    def test_get_guide_template(self):
        """验证能正确获取攻略类模板"""
        template = get_template(TemplateType.GUIDE)
        assert template == GUIDE_TEMPLATE

    def test_get_comparison_template(self):
        """验证能正确获取对比类模板"""
        template = get_template(TemplateType.COMPARISON)
        assert template == COMPARISON_TEMPLATE

    def test_get_case_study_template(self):
        """验证能正确获取案例类模板"""
        template = get_template(TemplateType.CASE_STUDY)
        assert template == CASE_STUDY_TEMPLATE

    def test_invalid_template_type_raises_error(self):
        """验证传入无效模板类型时抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的模板类型"):
            get_template("invalid_type")

    def test_all_template_types_covered(self):
        """验证所有 TemplateType 枚举值都有对应模板"""
        for template_type in TemplateType:
            template = get_template(template_type)
            assert isinstance(template, str)
            assert len(template) > 0


class TestTemplateStructure:
    """测试模板结构完整性"""

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_content_placeholder(self, template_type):
        """验证每个模板都包含 {content} 占位符"""
        template = get_template(template_type)
        assert "{content}" in template

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_keywords_placeholder(self, template_type):
        """验证每个模板都包含 {keywords} 占位符"""
        template = get_template(template_type)
        assert "{keywords}" in template

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_year_placeholder(self, template_type):
        """验证每个模板都包含 {year} 占位符"""
        template = get_template(template_type)
        assert "{year}" in template

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_heading_structure(self, template_type):
        """验证每个模板要求使用层级标题（H1/H2/H3）"""
        template = get_template(template_type)
        assert "# " in template  # H1
        assert "## " in template  # H2
        assert "### " in template  # H3

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_table_structure(self, template_type):
        """验证每个模板包含表格结构"""
        template = get_template(template_type)
        # Markdown 表格使用 | 分隔符
        assert "|" in template

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_authority_reference(self, template_type):
        """验证每个模板要求植入权威数据引用（EEAT信号）"""
        template = get_template(template_type)
        assert "{authority_source}" in template

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_mentions_eeat(self, template_type):
        """验证每个模板的GEO优化要求中提及EEAT"""
        template = get_template(template_type)
        assert "EEAT" in template

    @pytest.mark.parametrize("template_type", list(TemplateType))
    def test_template_contains_industry_placeholder(self, template_type):
        """验证每个模板都包含 {industry} 占位符"""
        template = get_template(template_type)
        assert "{industry}" in template


class TestRankingTemplate:
    """测试榜单式模板的特定结构"""

    def test_contains_list_count_placeholder(self):
        """验证榜单模板包含 {list_count} 占位符"""
        assert "{list_count}" in RANKING_TEMPLATE

    def test_contains_benefit_placeholder(self):
        """验证榜单模板包含 {benefit} 占位符"""
        assert "{benefit}" in RANKING_TEMPLATE

    def test_contains_ranking_structure(self):
        """验证榜单模板包含评测标准+详细榜单+选购建议结构"""
        assert "评测标准" in RANKING_TEMPLATE
        assert "详细榜单" in RANKING_TEMPLATE
        assert "选购建议" in RANKING_TEMPLATE


class TestReviewTemplate:
    """测试评测类模板的特定结构"""

    def test_contains_product_name_placeholder(self):
        """验证评测模板包含 {product_name} 占位符"""
        assert "{product_name}" in REVIEW_TEMPLATE

    def test_contains_review_structure(self):
        """验证评测模板包含结论速览+核心功能实测+竞品对比+购买建议结构"""
        assert "结论速览" in REVIEW_TEMPLATE
        assert "核心功能实测" in REVIEW_TEMPLATE
        assert "竞品对比" in REVIEW_TEMPLATE
        assert "购买建议" in REVIEW_TEMPLATE


class TestGuideTemplate:
    """测试攻略类模板的特定结构"""

    def test_contains_guide_topic_placeholder(self):
        """验证攻略模板包含 {guide_topic} 占位符"""
        assert "{guide_topic}" in GUIDE_TEMPLATE

    def test_contains_guide_structure(self):
        """验证攻略模板包含准备工作+详细操作步骤+常见问题解答结构"""
        assert "准备工作" in GUIDE_TEMPLATE
        assert "详细操作步骤" in GUIDE_TEMPLATE
        assert "常见问题" in GUIDE_TEMPLATE


class TestComparisonTemplate:
    """测试对比类模板的特定结构"""

    def test_contains_comparison_items_placeholder(self):
        """验证对比模板包含 {comparison_items} 占位符"""
        assert "{comparison_items}" in COMPARISON_TEMPLATE

    def test_contains_comparison_structure(self):
        """验证对比模板包含速览结论+核心定位+多维度对比+选购建议+FAQ结构"""
        assert "速览结论" in COMPARISON_TEMPLATE
        assert "核心定位" in COMPARISON_TEMPLATE
        assert "多维度对比" in COMPARISON_TEMPLATE
        assert "选购建议" in COMPARISON_TEMPLATE
        assert "FAQ" in COMPARISON_TEMPLATE

    def test_contains_five_comparison_dimensions(self):
        """验证对比模板覆盖五个对比维度"""
        assert "价格" in COMPARISON_TEMPLATE
        assert "功能" in COMPARISON_TEMPLATE
        assert "易用性" in COMPARISON_TEMPLATE
        assert "适用场景" in COMPARISON_TEMPLATE
        assert "用户评价" in COMPARISON_TEMPLATE


class TestCaseStudyTemplate:
    """测试案例类模板的特定结构"""

    def test_contains_case_topic_placeholder(self):
        """验证案例模板包含 {case_topic} 占位符"""
        assert "{case_topic}" in CASE_STUDY_TEMPLATE

    def test_contains_star_structure(self):
        """验证案例模板包含STAR结构"""
        assert "背景" in CASE_STUDY_TEMPLATE
        assert "Situation" in CASE_STUDY_TEMPLATE
        assert "方案" in CASE_STUDY_TEMPLATE
        assert "Solution" in CASE_STUDY_TEMPLATE
        assert "结果" in CASE_STUDY_TEMPLATE
        assert "Result" in CASE_STUDY_TEMPLATE
        assert "总结" in CASE_STUDY_TEMPLATE

    def test_contains_data_comparison(self):
        """验证案例模板要求包含数据对比（实施前vs实施后）"""
        assert "实施前" in CASE_STUDY_TEMPLATE
        assert "实施后" in CASE_STUDY_TEMPLATE
