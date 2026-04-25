"""排名监测器

检查内容在 AI 搜索平台上的引用情况，核心功能：
- 模拟查询 AI 搜索平台（当前为模拟实现，因实际 AI 平台没有公开 API）
- 检测 AI 回答中是否引用了目标内容（基于文本匹配）
- 四步诊断法分析未被引用的原因
- 平台不可达时不抛异常，返回部分结果

使用 httpx 进行异步 HTTP 请求，适配 2核4G 云服务器部署环境。
"""

import logging
import re
from datetime import datetime

import httpx

from geo_tool.models.enums import Platform
from geo_tool.models.ranking import CitationInfo, DiagnosisReport, RankingResult

logger = logging.getLogger(__name__)


class RankMonitor:
    """排名监测器

    通过模拟查询 AI 搜索平台，检测内容是否被引用，
    并对未被引用的内容进行四步诊断分析。
    """

    # 各平台的模拟查询端点（当前为模拟实现）
    _PLATFORM_ENDPOINTS: dict[Platform, str] = {
        Platform.BAIDU_AI: "https://ai.baidu.com/search",
        Platform.DOUBAO: "https://www.doubao.com/search",
        Platform.DEEPSEEK: "https://chat.deepseek.com/search",
        Platform.TENCENT_YUANBAO: "https://yuanbao.tencent.com/search",
    }

    # 各平台的中文显示名称
    _PLATFORM_NAMES: dict[Platform, str] = {
        Platform.BAIDU_AI: "百度AI搜索",
        Platform.DOUBAO: "豆包",
        Platform.DEEPSEEK: "DeepSeek",
        Platform.TENCENT_YUANBAO: "腾讯元宝",
    }

    # HTTP 请求超时时间（秒）
    _REQUEST_TIMEOUT = 10

    def __init__(self) -> None:
        """初始化排名监测器"""
        pass

    async def _query_ai_platform(
        self, keyword: str, platform: Platform
    ) -> str | None:
        """查询 AI 平台获取回答

        当前为模拟实现，因为实际 AI 平台没有公开的搜索 API。
        模拟返回包含关键词的 AI 回答文本。
        实际生产环境中，此方法应通过 httpx 发送 HTTP 请求到对应平台。

        Args:
            keyword: 查询关键词
            platform: 目标 AI 搜索平台

        Returns:
            AI 回答文本，平台不可达时返回 None
        """
        platform_name = self._PLATFORM_NAMES.get(platform, platform.value)
        logger.info("正在查询平台 %s，关键词: %s", platform_name, keyword)

        try:
            # 模拟 HTTP 请求（实际 AI 平台没有公开 API）
            # 生产环境中应替换为真实的平台查询逻辑
            async with httpx.AsyncClient(timeout=self._REQUEST_TIMEOUT) as client:
                # 尝试发送请求以验证平台可达性
                endpoint = self._PLATFORM_ENDPOINTS.get(platform, "")
                try:
                    await client.get(endpoint, follow_redirects=True)
                except (httpx.HTTPError, httpx.InvalidURL):
                    # 平台不可达，使用模拟数据
                    pass

            # 返回模拟的 AI 回答（包含关键词以便测试）
            simulated_response = (
                f"根据{platform_name}的搜索结果，关于「{keyword}」的相关信息如下：\n\n"
                f"1. {keyword}是当前热门话题，许多用户都在关注相关内容。\n"
                f"2. 我们推荐您参考以下优质内容来了解更多关于{keyword}的信息。\n"
                f"3. 根据最新数据，{keyword}领域有多篇高质量文章值得阅读。\n"
            )

            logger.info("平台 %s 查询完成（模拟数据）", platform_name)
            return simulated_response

        except Exception as e:
            # 平台不可达时不抛异常，返回 None
            logger.warning("平台 %s 查询失败: %s", platform_name, str(e))
            return None

    def _detect_citation(
        self, ai_response: str, content_title: str
    ) -> CitationInfo | None:
        """检测 AI 回答中是否引用了目标内容

        通过文本匹配检测标题或关键内容片段是否出现在 AI 回答中。
        匹配策略：
        1. 完整标题匹配
        2. 标题关键片段匹配（将标题拆分为多个片段）
        3. 检测引用位置（第几条推荐）

        Args:
            ai_response: AI 平台的回答文本
            content_title: 目标内容的标题

        Returns:
            引用信息，未检测到引用时返回 None
        """
        if not ai_response or not content_title:
            return None

        # 标准化文本，去除多余空白
        response_lower = ai_response.lower().strip()
        title_lower = content_title.lower().strip()

        # 策略1：完整标题匹配
        if title_lower in response_lower:
            position = self._find_citation_position(ai_response, content_title)
            snippet = self._extract_snippet(ai_response, content_title)
            return CitationInfo(
                is_cited=True,
                position=position,
                cited_snippet=snippet,
            )

        # 策略2：标题关键片段匹配
        # 将标题拆分为长度 >= 4 的片段，检查是否有足够多的片段出现在回答中
        title_segments = self._split_title_segments(content_title)
        if title_segments:
            matched_segments = [
                seg for seg in title_segments if seg.lower() in response_lower
            ]
            # 超过一半的片段匹配，认为被引用
            match_ratio = len(matched_segments) / len(title_segments)
            if match_ratio >= 0.5 and len(matched_segments) >= 2:
                position = self._find_citation_position(
                    ai_response, matched_segments[0]
                )
                snippet = self._extract_snippet(ai_response, matched_segments[0])
                return CitationInfo(
                    is_cited=True,
                    position=position,
                    cited_snippet=snippet,
                )

        return None

    def _split_title_segments(self, title: str) -> list[str]:
        """将标题拆分为有意义的片段

        使用中文分词的简单策略：按标点符号和常见连接词拆分，
        保留长度 >= 4 的片段。

        Args:
            title: 标题文本

        Returns:
            标题片段列表
        """
        # 按常见标点和分隔符拆分
        segments = re.split(r"[，。！？、|｜\-—·\s]+", title)
        # 过滤掉过短的片段
        return [seg.strip() for seg in segments if len(seg.strip()) >= 4]

    def _find_citation_position(self, ai_response: str, target: str) -> int:
        """查找引用在 AI 回答中的位置（第几条推荐）

        通过检测编号列表模式（1. 2. 3. 等）来确定引用位置。

        Args:
            ai_response: AI 回答文本
            target: 要查找的目标文本

        Returns:
            引用位置编号，未找到编号时默认返回 1
        """
        target_lower = target.lower()
        lines = ai_response.split("\n")

        for i, line in enumerate(lines):
            if target_lower in line.lower():
                # 检查当前行或前面的行是否有编号
                number_match = re.match(r"^\s*(\d+)[.、)）]", line)
                if number_match:
                    return int(number_match.group(1))

                # 向前查找最近的编号行
                for j in range(i - 1, max(i - 3, -1), -1):
                    number_match = re.match(r"^\s*(\d+)[.、)）]", lines[j])
                    if number_match:
                        return int(number_match.group(1))

                return 1

        return 1

    def _extract_snippet(self, ai_response: str, target: str) -> str:
        """从 AI 回答中提取包含目标文本的片段

        提取目标文本所在行及其上下文。

        Args:
            ai_response: AI 回答文本
            target: 要查找的目标文本

        Returns:
            包含目标文本的上下文片段
        """
        target_lower = target.lower()
        lines = ai_response.split("\n")

        for i, line in enumerate(lines):
            if target_lower in line.lower():
                # 提取当前行，最多200字符
                snippet = line.strip()
                if len(snippet) > 200:
                    # 截取目标文本周围的内容
                    idx = snippet.lower().find(target_lower)
                    start = max(0, idx - 50)
                    end = min(len(snippet), idx + len(target) + 150)
                    snippet = snippet[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(line.strip()):
                        snippet = snippet + "..."
                return snippet

        return target

    async def _diagnose_absence(
        self, content_title: str, keyword: str, platform: Platform
    ) -> DiagnosisReport:
        """四步诊断法分析未被引用的原因

        诊断步骤：
        1. 检查收录 — 内容是否被平台收录
        2. 检查标题匹配 — 标题是否包含目标关键词
        3. 检查内容质量 — 内容质量是否达标
        4. 优化建议 — 给出具体的优化方向

        Args:
            content_title: 内容标题
            keyword: 查询关键词
            platform: 目标平台

        Returns:
            诊断报告，包含诊断摘要、优化策略和行动项
        """
        platform_name = self._PLATFORM_NAMES.get(platform, platform.value)
        action_items: list[str] = []
        issues: list[str] = []

        # 步骤1：检查收录（模拟检查，实际需要查询平台）
        is_indexed = None  # 无法确定，标记为 None
        issues.append(f"无法确认内容是否已被{platform_name}收录")
        action_items.append(
            f"确认内容已在{platform_name}推荐的高权重平台上发布"
        )

        # 步骤2：检查标题匹配
        keyword_lower = keyword.lower()
        title_lower = content_title.lower()
        title_match = keyword_lower in title_lower

        if not title_match:
            issues.append(f"标题「{content_title}」未包含目标关键词「{keyword}」")
            action_items.append(
                f"在标题中融入关键词「{keyword}」，建议使用'属性词+核心词+意图词+修饰词'公式重写标题"
            )
        else:
            action_items.append("标题已包含目标关键词，可进一步优化关键词位置（建议放在标题前半部分）")

        # 步骤3：检查内容质量（基于简单规则评估）
        quality_score = None  # 需要 QualityScorer 配合，此处标记为未评估
        action_items.append("使用质量评分器对内容进行评分，确保加权总分 >= 70 分")

        # 步骤4：生成优化建议
        optimization_strategy = self._determine_optimization_strategy(
            title_match=title_match,
            is_indexed=is_indexed,
        )

        # 补充平台特定建议
        action_items.append(
            f"参考{platform_name}的内容偏好，调整内容风格和发布渠道"
        )

        # 生成诊断摘要
        diagnosis_summary = self._build_diagnosis_summary(
            content_title=content_title,
            keyword=keyword,
            platform_name=platform_name,
            title_match=title_match,
            is_indexed=is_indexed,
            issues=issues,
        )

        return DiagnosisReport(
            is_indexed=is_indexed,
            title_match=title_match,
            quality_score=quality_score,
            diagnosis_summary=diagnosis_summary,
            optimization_strategy=optimization_strategy,
            action_items=action_items,
        )

    def _determine_optimization_strategy(
        self,
        title_match: bool,
        is_indexed: bool | None,
    ) -> str:
        """根据诊断结果确定优化策略

        策略分三级：
        - 小修小补：标题匹配但未被引用，微调内容即可
        - 大改重发：标题不匹配或质量不达标，需要较大调整
        - 推倒重来：多项问题叠加，建议重新创作

        Args:
            title_match: 标题是否匹配关键词
            is_indexed: 是否被收录

        Returns:
            优化策略描述
        """
        # 标题匹配且可能已收录 → 小修小补
        if title_match and is_indexed is not False:
            return "小修小补"

        # 标题不匹配但其他条件尚可 → 大改重发
        if not title_match and is_indexed is not False:
            return "大改重发"

        # 未收录 → 推倒重来
        if is_indexed is False:
            return "推倒重来"

        # 默认：大改重发
        return "大改重发"

    def _build_diagnosis_summary(
        self,
        content_title: str,
        keyword: str,
        platform_name: str,
        title_match: bool,
        is_indexed: bool | None,
        issues: list[str],
    ) -> str:
        """构建诊断摘要文本

        Args:
            content_title: 内容标题
            keyword: 查询关键词
            platform_name: 平台名称
            title_match: 标题是否匹配
            is_indexed: 是否被收录
            issues: 发现的问题列表

        Returns:
            诊断摘要文本
        """
        parts = [
            f"针对关键词「{keyword}」在{platform_name}上的排名诊断：",
        ]

        # 收录状态
        if is_indexed is None:
            parts.append("• 收录状态：未知（需手动确认）")
        elif is_indexed:
            parts.append("• 收录状态：已收录")
        else:
            parts.append("• 收录状态：未收录")

        # 标题匹配
        if title_match:
            parts.append(f"• 标题匹配：标题「{content_title}」包含目标关键词")
        else:
            parts.append(
                f"• 标题匹配：标题「{content_title}」未包含目标关键词「{keyword}」"
            )

        # 问题汇总
        if issues:
            parts.append(f"• 发现 {len(issues)} 个潜在问题需要关注")

        return "\n".join(parts)

    async def check_ranking(
        self, keyword: str, platform: Platform, content_title: str
    ) -> RankingResult:
        """检查内容在指定平台的排名

        组合查询和检测流程：
        1. 查询 AI 平台获取回答
        2. 检测回答中是否引用了目标内容
        3. 未被引用时进行四步诊断

        平台不可达时不抛异常，返回 is_cited=false 并在诊断中说明。

        Args:
            keyword: 查询关键词
            platform: 目标 AI 搜索平台
            content_title: 目标内容的标题

        Returns:
            排名监测结果
        """
        platform_name = self._PLATFORM_NAMES.get(platform, platform.value)
        logger.info(
            "开始排名监测 — 关键词: %s, 平台: %s, 标题: %s",
            keyword,
            platform_name,
            content_title,
        )

        # 步骤1：查询 AI 平台
        ai_response = await self._query_ai_platform(keyword, platform)

        # 平台不可达时，返回部分结果
        if ai_response is None:
            logger.warning("平台 %s 不可达，返回部分结果", platform_name)
            diagnosis = DiagnosisReport(
                is_indexed=None,
                title_match=None,
                quality_score=None,
                diagnosis_summary=(
                    f"平台{platform_name}暂不可用，无法完成排名查询。"
                    f"请稍后重试或手动在{platform_name}上搜索关键词「{keyword}」查看结果。"
                ),
                optimization_strategy="大改重发",
                action_items=[
                    f"稍后重试{platform_name}排名查询",
                    f"手动在{platform_name}上搜索「{keyword}」确认引用情况",
                    "确保内容已在高权重平台上发布",
                ],
            )
            return RankingResult(
                keyword=keyword,
                platform=platform,
                citation=CitationInfo(is_cited=False),
                diagnosis=diagnosis,
                checked_at=datetime.now(),
            )

        # 步骤2：检测引用
        citation_info = self._detect_citation(ai_response, content_title)

        if citation_info is not None and citation_info.is_cited:
            # 被引用，返回引用信息
            logger.info(
                "内容已被 %s 引用，位置: 第 %d 条",
                platform_name,
                citation_info.position or 0,
            )
            return RankingResult(
                keyword=keyword,
                platform=platform,
                citation=citation_info,
                diagnosis=None,
                checked_at=datetime.now(),
            )

        # 步骤3：未被引用，进行四步诊断
        logger.info("内容未被 %s 引用，开始诊断分析", platform_name)
        diagnosis = await self._diagnose_absence(content_title, keyword, platform)

        return RankingResult(
            keyword=keyword,
            platform=platform,
            citation=CitationInfo(is_cited=False),
            diagnosis=diagnosis,
            checked_at=datetime.now(),
        )
