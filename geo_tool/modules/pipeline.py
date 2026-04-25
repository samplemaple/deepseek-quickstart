"""管道编排模块

串联 提取→分析→二创→评分→适配 完整流程，实现一键执行。
支持部分成功处理：某步骤失败时返回已完成步骤结果和失败步骤错误信息。
"""

import logging

from geo_tool.models.api import PipelineResponse
from geo_tool.models.enums import BusinessType, Platform, TemplateType
from geo_tool.modules.content_extractor import ContentExtractor
from geo_tool.modules.content_recreator import ContentRecreator
from geo_tool.modules.keyword_analyzer import KeywordAnalyzer
from geo_tool.modules.platform_adapter import PlatformAdapter
from geo_tool.modules.quality_scorer import QualityScorer

logger = logging.getLogger(__name__)


class Pipeline:
    """管道编排器

    接收所有核心模块实例，通过 run() 方法串联完整的 GEO 内容二创流程。
    每个步骤失败时，保留已完成步骤的结果，在 error 字段中记录失败信息。
    """

    def __init__(
        self,
        extractor: ContentExtractor,
        analyzer: KeywordAnalyzer,
        recreator: ContentRecreator,
        scorer: QualityScorer,
        adapter: PlatformAdapter,
    ) -> None:
        """初始化管道编排器

        Args:
            extractor: 内容提取器
            analyzer: 关键词分析器
            recreator: 内容二创引擎
            scorer: 质量评分器
            adapter: 平台适配器
        """
        self._extractor = extractor
        self._analyzer = analyzer
        self._recreator = recreator
        self._scorer = scorer
        self._adapter = adapter

    async def run(
        self,
        url: str,
        template_type: TemplateType | None = None,
        platforms: list[Platform] | None = None,
        business_type: BusinessType | None = None,
    ) -> PipelineResponse:
        """执行完整的 GEO 内容二创管道

        流程：提取 → 分析 → 二创 → 评分 → 适配
        任一步骤失败时，保留已完成步骤的结果并记录错误信息。

        Args:
            url: 小红书笔记链接
            template_type: 模板类型，为 None 时自动推荐
            platforms: 目标平台列表，默认为 [豆包, DeepSeek]
            business_type: 业务类型，用于生成转化策略

        Returns:
            PipelineResponse，包含各步骤结果和可能的错误信息
        """
        if platforms is None:
            platforms = [Platform.DOUBAO, Platform.DEEPSEEK]

        response = PipelineResponse(success=False)

        # ===== 步骤1: 内容提取 =====
        logger.info("管道步骤 1/5：开始提取内容，URL=%s", url)
        try:
            extract_result = await self._extractor.extract(url)
            if not extract_result.get("success"):
                error_msg = extract_result.get("error", "内容提取失败：未知错误")
                response.error = f"提取步骤失败：{error_msg}"
                logger.error("管道在提取步骤失败：%s", error_msg)
                return response
            extracted = extract_result["data"]
            response.extracted = extracted
            logger.info("管道步骤 1/5 完成：内容提取成功")
        except Exception as e:
            response.error = f"提取步骤异常：{e}"
            logger.exception("管道在提取步骤发生异常")
            return response

        # ===== 步骤2: 关键词分析 =====
        logger.info("管道步骤 2/5：开始关键词分析")
        try:
            keywords = await self._analyzer.analyze(extracted)
            response.keywords = keywords
            logger.info("管道步骤 2/5 完成：关键词分析成功")
        except Exception as e:
            response.error = f"关键词分析步骤异常：{e}"
            logger.exception("管道在关键词分析步骤发生异常")
            return response

        # ===== 步骤3: 内容二创 =====
        logger.info("管道步骤 3/5：开始内容二创")
        try:
            recreated = await self._recreator.recreate(
                content=extracted,
                keywords=keywords,
                template_type=template_type,
                business_type=business_type,
            )
            response.recreated = recreated
            logger.info("管道步骤 3/5 完成：内容二创成功")
        except Exception as e:
            response.error = f"内容二创步骤异常：{e}"
            logger.exception("管道在内容二创步骤发生异常")
            return response

        # ===== 步骤4: 质量评分 =====
        logger.info("管道步骤 4/5：开始质量评分")
        try:
            quality = await self._scorer.score(recreated)
            response.quality = quality
            logger.info("管道步骤 4/5 完成：质量评分成功")
        except Exception as e:
            response.error = f"质量评分步骤异常：{e}"
            logger.exception("管道在质量评分步骤发生异常")
            return response

        # ===== 步骤5: 平台适配 =====
        logger.info("管道步骤 5/5：开始平台适配，目标平台=%s", [p.value for p in platforms])
        try:
            platform_contents = await self._adapter.adapt(recreated, platforms)
            response.platform_contents = platform_contents
            logger.info("管道步骤 5/5 完成：平台适配成功")
        except Exception as e:
            response.error = f"平台适配步骤异常：{e}"
            logger.exception("管道在平台适配步骤发生异常")
            return response

        # 全部步骤成功
        response.success = True
        logger.info("管道执行完成：所有步骤成功")
        return response
