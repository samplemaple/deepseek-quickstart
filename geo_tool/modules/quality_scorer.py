"""GEO 质量评分器

基于 GEO 五大标准对内容进行量化评分。
核心功能：
- 结构化程度评估（规则引擎）
- 信息密度评估（LLM 辅助）
- 时效性评估（规则引擎）
- 权威性与可信度评估（LLM 辅助）
- 实用性评估（LLM 辅助）
- 加权总分计算和质量等级判定
- LLM 不可用时降级为纯规则引擎评估
"""

import logging
import re

from geo_tool.models.enums import QualityLevel
from geo_tool.models.quality import DimensionScore, QualityReport
from geo_tool.models.recreated import RecreatedContent
from geo_tool.modules.llm_client import LLMClient

logger = logging.getLogger(__name__)


class QualityScorer:
    """GEO 质量评分器

    结合规则引擎和 LLM 对二创内容进行五维度评分，
    生成质量报告并给出优化建议。
    """

    # 五个维度名称
    _DIM_STRUCTURE = "结构化程度"
    _DIM_DENSITY = "信息密度"
    _DIM_TIMELINESS = "时效性"
    _DIM_AUTHORITY = "权威性与可信度"
    _DIM_PRACTICALITY = "实用性"

    # 五个维度权重（总和 = 1.0）
    _WEIGHT_STRUCTURE = 0.30
    _WEIGHT_DENSITY = 0.25
    _WEIGHT_TIMELINESS = 0.20
    _WEIGHT_AUTHORITY = 0.15
    _WEIGHT_PRACTICALITY = 0.10

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """初始化质量评分器

        Args:
            llm_client: DeepSeek API 客户端实例（依赖注入），
                        为 None 时进入降级模式，仅使用规则引擎评估
        """
        self._llm = llm_client

    async def score(self, content: RecreatedContent) -> QualityReport:
        """对内容进行 GEO 五大标准评分

        组合五个维度的评分结果，计算加权总分，
        判定质量等级并生成优化建议。

        Args:
            content: 二创内容

        Returns:
            质量评分报告
        """
        body = content.body

        # 规则引擎维度（始终可用）
        structure_score = self._score_structure(body)
        timeliness_score = self._score_timeliness(body)

        # LLM 辅助维度（降级时标记为"无法评估"）
        if self._llm is not None:
            density_score = await self._score_density(body)
            authority_score = await self._score_authority(body)
            practicality_score = await self._score_practicality(body)
        else:
            # 降级模式：LLM 不可用，标记为"无法评估"
            logger.warning("LLM 不可用，信息密度/权威性/实用性降级为无法评估")
            density_score = DimensionScore(
                dimension=self._DIM_DENSITY,
                weight=self._WEIGHT_DENSITY,
                score=0.0,
                deduction_reasons=["LLM 不可用，无法评估信息密度"],
                suggestions=["请配置 LLM 后重新评估"],
            )
            authority_score = DimensionScore(
                dimension=self._DIM_AUTHORITY,
                weight=self._WEIGHT_AUTHORITY,
                score=0.0,
                deduction_reasons=["LLM 不可用，无法评估权威性"],
                suggestions=["请配置 LLM 后重新评估"],
            )
            practicality_score = DimensionScore(
                dimension=self._DIM_PRACTICALITY,
                weight=self._WEIGHT_PRACTICALITY,
                score=0.0,
                deduction_reasons=["LLM 不可用，无法评估实用性"],
                suggestions=["请配置 LLM 后重新评估"],
            )

        # 组合所有维度评分
        dimension_scores = [
            structure_score,
            density_score,
            timeliness_score,
            authority_score,
            practicality_score,
        ]

        # 计算加权总分
        weighted_total = self._calculate_weighted_total(dimension_scores)

        # 判定质量等级
        quality_level = self._determine_quality_level(weighted_total)

        # 生成优先改进项（仅在需优化时）
        priority_improvements = self._generate_improvements(
            dimension_scores, quality_level
        )

        return QualityReport(
            dimension_scores=dimension_scores,
            weighted_total=weighted_total,
            quality_level=quality_level,
            priority_improvements=priority_improvements,
        )

    def _score_structure(self, content: str) -> DimensionScore:
        """评估结构化程度（规则引擎）

        检测内容中的 H1/H2/H3 层级标题、表格、列表等结构化元素，
        根据元素的丰富程度给出 0-100 分的评分。

        Args:
            content: 内容正文

        Returns:
            结构化程度维度评分
        """
        score = 0.0
        deductions: list[str] = []
        suggestions: list[str] = []

        # 检测 H1 标题（# 开头或 HTML <h1>）
        h1_pattern = re.compile(r"(?m)^#\s+.+|<h1[^>]*>.+</h1>")
        h1_count = len(h1_pattern.findall(content))
        if h1_count > 0:
            score += 20.0
        else:
            deductions.append("缺少 H1 主标题")
            suggestions.append("建议添加 H1 主标题，明确文章核心主题")

        # 检测 H2 标题（## 开头或 HTML <h2>）
        h2_pattern = re.compile(r"(?m)^##\s+.+|<h2[^>]*>.+</h2>")
        h2_count = len(h2_pattern.findall(content))
        if h2_count >= 2:
            score += 20.0
        elif h2_count == 1:
            score += 10.0
            deductions.append("H2 子标题数量不足（仅1个）")
            suggestions.append("建议增加 H2 子标题，丰富内容层级结构")
        else:
            deductions.append("缺少 H2 子标题")
            suggestions.append("建议添加 H2 子标题，划分内容段落")

        # 检测 H3 标题（### 开头或 HTML <h3>）
        h3_pattern = re.compile(r"(?m)^###\s+.+|<h3[^>]*>.+</h3>")
        h3_count = len(h3_pattern.findall(content))
        if h3_count >= 2:
            score += 15.0
        elif h3_count == 1:
            score += 8.0
        else:
            deductions.append("缺少 H3 细分标题")
            suggestions.append("建议添加 H3 细分标题，进一步细化内容结构")

        # 检测表格（Markdown 表格或 HTML 表格）
        table_md = re.compile(r"(?m)^\|.+\|$")
        table_html = re.compile(r"<table[^>]*>", re.IGNORECASE)
        has_table = bool(table_md.search(content)) or bool(
            table_html.search(content)
        )
        if has_table:
            score += 20.0
        else:
            deductions.append("缺少表格元素")
            suggestions.append("建议添加对比表格或数据表格，提升信息可读性")

        # 检测列表（有序列表或无序列表）
        list_pattern = re.compile(r"(?m)^[\s]*[-*+]\s+.+|^[\s]*\d+[.、]\s*.+")
        list_count = len(list_pattern.findall(content))
        if list_count >= 5:
            score += 15.0
        elif list_count >= 2:
            score += 10.0
        elif list_count >= 1:
            score += 5.0
        else:
            deductions.append("缺少列表元素")
            suggestions.append("建议使用有序或无序列表，提升内容条理性")

        # 检测加粗/强调元素
        bold_pattern = re.compile(r"\*\*.+?\*\*|<strong>.+?</strong>|<b>.+?</b>")
        bold_count = len(bold_pattern.findall(content))
        if bold_count >= 3:
            score += 10.0
        elif bold_count >= 1:
            score += 5.0

        # 确保分数在 0-100 范围内
        score = min(100.0, max(0.0, score))

        return DimensionScore(
            dimension=self._DIM_STRUCTURE,
            weight=self._WEIGHT_STRUCTURE,
            score=score,
            deduction_reasons=deductions,
            suggestions=suggestions,
        )

    async def _score_density(self, content: str) -> DimensionScore:
        """评估信息密度（LLM 辅助）

        通过 LLM 评估内容中具体数据、真实案例的丰富程度，
        以及冗余废话的占比，给出 0-100 分的评分。

        Args:
            content: 内容正文

        Returns:
            信息密度维度评分
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的内容质量评估专家。\n"
                    "请评估以下内容的信息密度，从三个方面打分：\n"
                    "1. 具体数据丰富度（0-40分）：是否包含具体数字、百分比、价格等量化数据\n"
                    "2. 真实案例丰富度（0-30分）：是否包含真实案例、实际体验、具体场景\n"
                    "3. 冗余废话占比（0-30分）：内容是否精炼，废话少得30分，废话多得0分\n\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请评估以下内容的信息密度：\n\n"
                    f"{content[:3000]}\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"data_richness": 25, "case_richness": 20, '
                    f'"conciseness": 20, "deductions": ["扣分原因1"], '
                    f'"suggestions": ["优化建议1"]}}'
                ),
            },
        ]

        try:
            result = await self._llm.chat_json(messages, temperature=0.3)
            data_richness = self._clamp(result.get("data_richness", 20), 0, 40)
            case_richness = self._clamp(result.get("case_richness", 15), 0, 30)
            conciseness = self._clamp(result.get("conciseness", 15), 0, 30)
            total = data_richness + case_richness + conciseness

            deductions = [str(d) for d in result.get("deductions", []) if d]
            suggestions = [str(s) for s in result.get("suggestions", []) if s]

            return DimensionScore(
                dimension=self._DIM_DENSITY,
                weight=self._WEIGHT_DENSITY,
                score=min(100.0, max(0.0, float(total))),
                deduction_reasons=deductions,
                suggestions=suggestions,
            )
        except Exception as e:
            logger.error("信息密度评估失败: %s", str(e))
            return DimensionScore(
                dimension=self._DIM_DENSITY,
                weight=self._WEIGHT_DENSITY,
                score=0.0,
                deduction_reasons=[f"评估失败: {str(e)}"],
                suggestions=["请检查 LLM 配置后重新评估"],
            )

    def _score_timeliness(self, content: str) -> DimensionScore:
        """评估时效性（规则引擎）

        检测内容标题和正文中的时间标注（如"2025年"）、
        数据时效性标记，给出 0-100 分的评分。

        Args:
            content: 内容正文

        Returns:
            时效性维度评分
        """
        score = 0.0
        deductions: list[str] = []
        suggestions: list[str] = []

        # 检测年份标注（如 2024年、2025年）
        year_pattern = re.compile(r"20[2-3]\d\s*年")
        year_matches = year_pattern.findall(content)
        if len(year_matches) >= 3:
            score += 35.0
        elif len(year_matches) >= 1:
            score += 20.0
        else:
            deductions.append("缺少年份时间标注")
            suggestions.append("建议在标题和正文中添加年份标注（如「2025年」），提升时效性信号")

        # 检测具体时间标注（月份、季度等）
        time_pattern = re.compile(
            r"\d{1,2}\s*月|[一二三四]季度|Q[1-4]|上半年|下半年|"
            r"春季|夏季|秋季|冬季|本年度|年度"
        )
        time_matches = time_pattern.findall(content)
        if len(time_matches) >= 2:
            score += 25.0
        elif len(time_matches) >= 1:
            score += 15.0
        else:
            deductions.append("缺少具体时间段标注（月份/季度）")
            suggestions.append("建议添加具体时间段（如「Q1」「3月」），增强数据时效性")

        # 检测"最新"、"更新"等时效性关键词
        freshness_pattern = re.compile(r"最新|更新|近期|最近|实时|当前|目前")
        freshness_matches = freshness_pattern.findall(content)
        if len(freshness_matches) >= 2:
            score += 20.0
        elif len(freshness_matches) >= 1:
            score += 10.0

        # 检测数据来源时间标注（如"据2025年报告"）
        data_time_pattern = re.compile(r"据?\s*20[2-3]\d\s*年.*?(?:报告|数据|统计|调查|研究)")
        data_time_matches = data_time_pattern.findall(content)
        if data_time_matches:
            score += 20.0
        else:
            deductions.append("缺少带时间标注的数据引用")
            suggestions.append("建议引用带时间标注的数据源（如「据2025年艾瑞咨询报告」）")

        # 确保分数在 0-100 范围内
        score = min(100.0, max(0.0, score))

        return DimensionScore(
            dimension=self._DIM_TIMELINESS,
            weight=self._WEIGHT_TIMELINESS,
            score=score,
            deduction_reasons=deductions,
            suggestions=suggestions,
        )

    async def _score_authority(self, content: str) -> DimensionScore:
        """评估权威性与可信度（LLM 辅助）

        通过 LLM 评估内容是否引用了权威报告、官方数据、
        专家身份说明和可验证信息，给出 0-100 分的评分。

        Args:
            content: 内容正文

        Returns:
            权威性与可信度维度评分
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的内容权威性评估专家。\n"
                    "请评估以下内容的权威性与可信度，从四个方面打分：\n"
                    "1. 权威报告引用（0-30分）：是否引用了行业报告、研究论文等权威来源\n"
                    "2. 官方数据引用（0-25分）：是否引用了官方统计数据、政府数据等\n"
                    "3. 专家身份声明（0-25分）：是否有专业身份声明、从业经验说明\n"
                    "4. 可验证信息（0-20分）：是否包含可验证的具体信息（品牌名、型号、价格等）\n\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请评估以下内容的权威性与可信度：\n\n"
                    f"{content[:3000]}\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"authority_reports": 15, "official_data": 10, '
                    f'"expert_identity": 10, "verifiable_info": 10, '
                    f'"deductions": ["扣分原因1"], "suggestions": ["优化建议1"]}}'
                ),
            },
        ]

        try:
            result = await self._llm.chat_json(messages, temperature=0.3)
            authority_reports = self._clamp(
                result.get("authority_reports", 10), 0, 30
            )
            official_data = self._clamp(result.get("official_data", 10), 0, 25)
            expert_identity = self._clamp(
                result.get("expert_identity", 10), 0, 25
            )
            verifiable_info = self._clamp(
                result.get("verifiable_info", 10), 0, 20
            )
            total = authority_reports + official_data + expert_identity + verifiable_info

            deductions = [str(d) for d in result.get("deductions", []) if d]
            suggestions = [str(s) for s in result.get("suggestions", []) if s]

            return DimensionScore(
                dimension=self._DIM_AUTHORITY,
                weight=self._WEIGHT_AUTHORITY,
                score=min(100.0, max(0.0, float(total))),
                deduction_reasons=deductions,
                suggestions=suggestions,
            )
        except Exception as e:
            logger.error("权威性评估失败: %s", str(e))
            return DimensionScore(
                dimension=self._DIM_AUTHORITY,
                weight=self._WEIGHT_AUTHORITY,
                score=0.0,
                deduction_reasons=[f"评估失败: {str(e)}"],
                suggestions=["请检查 LLM 配置后重新评估"],
            )

    async def _score_practicality(self, content: str) -> DimensionScore:
        """评估实用性（LLM 辅助）

        通过 LLM 评估内容是否提供了明确的结论、
        可执行的建议或操作步骤，给出 0-100 分的评分。

        Args:
            content: 内容正文

        Returns:
            实用性维度评分
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的内容实用性评估专家。\n"
                    "请评估以下内容的实用性，从三个方面打分：\n"
                    "1. 明确结论（0-35分）：是否提供了清晰、明确的结论或推荐\n"
                    "2. 可执行建议（0-35分）：是否包含具体的、可操作的建议或步骤\n"
                    "3. 决策辅助（0-30分）：是否能帮助读者做出决策（对比、推荐理由等）\n\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请评估以下内容的实用性：\n\n"
                    f"{content[:3000]}\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"clear_conclusion": 20, "actionable_advice": 20, '
                    f'"decision_support": 15, "deductions": ["扣分原因1"], '
                    f'"suggestions": ["优化建议1"]}}'
                ),
            },
        ]

        try:
            result = await self._llm.chat_json(messages, temperature=0.3)
            clear_conclusion = self._clamp(
                result.get("clear_conclusion", 15), 0, 35
            )
            actionable_advice = self._clamp(
                result.get("actionable_advice", 15), 0, 35
            )
            decision_support = self._clamp(
                result.get("decision_support", 10), 0, 30
            )
            total = clear_conclusion + actionable_advice + decision_support

            deductions = [str(d) for d in result.get("deductions", []) if d]
            suggestions = [str(s) for s in result.get("suggestions", []) if s]

            return DimensionScore(
                dimension=self._DIM_PRACTICALITY,
                weight=self._WEIGHT_PRACTICALITY,
                score=min(100.0, max(0.0, float(total))),
                deduction_reasons=deductions,
                suggestions=suggestions,
            )
        except Exception as e:
            logger.error("实用性评估失败: %s", str(e))
            return DimensionScore(
                dimension=self._DIM_PRACTICALITY,
                weight=self._WEIGHT_PRACTICALITY,
                score=0.0,
                deduction_reasons=[f"评估失败: {str(e)}"],
                suggestions=["请检查 LLM 配置后重新评估"],
            )

    def _calculate_weighted_total(self, scores: list[DimensionScore]) -> float:
        """计算加权总分

        按各维度权重计算加权总分，结果范围 0-100。

        Args:
            scores: 五个维度的评分列表

        Returns:
            加权总分（0-100）
        """
        total = sum(s.score * s.weight for s in scores)
        return min(100.0, max(0.0, round(total, 2)))

    @staticmethod
    def _determine_quality_level(weighted_total: float) -> QualityLevel:
        """根据加权总分判定质量等级

        - weighted_total < 70 → needs_work（需优化）
        - 70 <= weighted_total < 85 → good（良好）
        - weighted_total >= 85 → excellent（优秀）

        Args:
            weighted_total: 加权总分

        Returns:
            质量等级
        """
        if weighted_total < 70:
            return QualityLevel.NEEDS_WORK
        elif weighted_total < 85:
            return QualityLevel.GOOD
        else:
            return QualityLevel.EXCELLENT

    @staticmethod
    def _generate_improvements(
        scores: list[DimensionScore], quality_level: QualityLevel
    ) -> list[str]:
        """生成优先改进项

        当质量等级为 needs_work 时，按维度权重从高到低
        列出各维度的优化建议。

        Args:
            scores: 五个维度的评分列表
            quality_level: 质量等级

        Returns:
            按优先级排列的改进项列表
        """
        if quality_level != QualityLevel.NEEDS_WORK:
            return []

        # 按权重从高到低排序，优先改进权重高的维度
        sorted_scores = sorted(scores, key=lambda s: s.weight, reverse=True)

        improvements: list[str] = []
        for dim_score in sorted_scores:
            # 低于 70 分的维度需要改进
            if dim_score.score < 70:
                for suggestion in dim_score.suggestions:
                    improvements.append(
                        f"【{dim_score.dimension}】{suggestion}"
                    )

        return improvements

    @staticmethod
    def _clamp(value: float | int, min_val: float, max_val: float) -> float:
        """将数值限制在指定范围内

        Args:
            value: 待限制的数值
            min_val: 最小值
            max_val: 最大值

        Returns:
            限制后的数值
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return min_val
        return max(min_val, min(max_val, v))
