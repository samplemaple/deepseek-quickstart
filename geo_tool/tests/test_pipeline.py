"""Pipeline 管道编排模块测试

测试完整管道流程、部分成功处理和错误传播。
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from geo_tool.models.content import ExtractedContent, NoteMetadata
from geo_tool.models.enums import (
    BusinessType,
    DecisionStage,
    KeywordType,
    Platform,
    QualityLevel,
    TemplateType,
)
from geo_tool.models.keyword import GEOKeyword, KeywordMatrix
from geo_tool.models.platform import PlatformConfig, PlatformContent
from geo_tool.models.quality import DimensionScore, QualityReport
from geo_tool.models.recreated import RecreatedContent
from geo_tool.models.title import Title, TitleDiagnosis
from geo_tool.modules.pipeline import Pipeline


# ===== 测试用固定数据 =====

def _make_metadata() -> NoteMetadata:
    return NoteMetadata(
        title="测试笔记标题",
        author="测试作者",
        publish_time="2024-01-01",
        likes=100,
        collects=50,
        comments=20,
    )


def _make_extracted() -> ExtractedContent:
    return ExtractedContent(
        url="https://www.xiaohongshu.com/explore/abc123",
        text="这是一篇测试笔记的正文内容",
        image_urls=["https://example.com/img1.jpg"],
        metadata=_make_metadata(),
    )


def _make_keywords() -> KeywordMatrix:
    return KeywordMatrix(
        core_topics=["测试主题"],
        keywords=[
            GEOKeyword(
                word="测试关键词",
                keyword_type=KeywordType.COGNITION,
                decision_stage=DecisionStage.AWARENESS,
                business_value=500.0,
                longtail_variants=["长尾词1", "长尾词2"],
            )
        ],
    )


def _make_recreated() -> RecreatedContent:
    return RecreatedContent(
        template_type=TemplateType.GUIDE,
        titles=[
            Title(
                text="2024年最全测试攻略指南推荐",
                diagnosis=TitleDiagnosis(
                    has_core_keyword=True,
                    has_intent_word=True,
                    has_timeliness_word=True,
                    has_attribute_word=True,
                    length_valid=True,
                    match_score=85.0,
                ),
            )
        ],
        body="# 测试攻略\n\n## 第一步\n\n详细内容...",
        keywords_used=["测试关键词"],
    )


def _make_quality_report() -> QualityReport:
    return QualityReport(
        dimension_scores=[
            DimensionScore(dimension="结构化", weight=0.30, score=80.0),
            DimensionScore(dimension="信息密度", weight=0.25, score=75.0),
            DimensionScore(dimension="时效性", weight=0.20, score=70.0),
            DimensionScore(dimension="权威性", weight=0.15, score=65.0),
            DimensionScore(dimension="实用性", weight=0.10, score=60.0),
        ],
        weighted_total=73.25,
        quality_level=QualityLevel.GOOD,
    )


def _make_platform_contents() -> list[PlatformContent]:
    return [
        PlatformContent(
            platform=Platform.DOUBAO,
            adapted_body="豆包适配版本内容",
            publish_suggestions=PlatformConfig(
                platform=Platform.DOUBAO,
                recommended_channels=["今日头条"],
                content_format_rules=["规则1"],
                notes=["注意事项1"],
            ),
        ),
        PlatformContent(
            platform=Platform.DEEPSEEK,
            adapted_body="DeepSeek适配版本内容",
            publish_suggestions=PlatformConfig(
                platform=Platform.DEEPSEEK,
                recommended_channels=["CSDN"],
                content_format_rules=["规则1"],
                notes=["注意事项1"],
            ),
        ),
    ]


def _build_pipeline(
    extract_result=None,
    keywords=None,
    recreated=None,
    quality=None,
    platform_contents=None,
    extract_side_effect=None,
    analyze_side_effect=None,
    recreate_side_effect=None,
    score_side_effect=None,
    adapt_side_effect=None,
) -> Pipeline:
    """构建带有 mock 模块的 Pipeline 实例"""
    extractor = MagicMock()
    analyzer = MagicMock()
    recreator = MagicMock()
    scorer = MagicMock()
    adapter = MagicMock()

    # 设置默认返回值
    if extract_side_effect:
        extractor.extract = AsyncMock(side_effect=extract_side_effect)
    else:
        extractor.extract = AsyncMock(
            return_value=extract_result or {"success": True, "data": _make_extracted()}
        )

    if analyze_side_effect:
        analyzer.analyze = AsyncMock(side_effect=analyze_side_effect)
    else:
        analyzer.analyze = AsyncMock(return_value=keywords or _make_keywords())

    if recreate_side_effect:
        recreator.recreate = AsyncMock(side_effect=recreate_side_effect)
    else:
        recreator.recreate = AsyncMock(return_value=recreated or _make_recreated())

    if score_side_effect:
        scorer.score = AsyncMock(side_effect=score_side_effect)
    else:
        scorer.score = AsyncMock(return_value=quality or _make_quality_report())

    if adapt_side_effect:
        adapter.adapt = AsyncMock(side_effect=adapt_side_effect)
    else:
        adapter.adapt = AsyncMock(
            return_value=platform_contents or _make_platform_contents()
        )

    return Pipeline(
        extractor=extractor,
        analyzer=analyzer,
        recreator=recreator,
        scorer=scorer,
        adapter=adapter,
    )


# ===== 测试用例 =====


@pytest.mark.asyncio
async def test_pipeline_full_success():
    """测试完整管道成功执行"""
    pipeline = _build_pipeline()
    result = await pipeline.run(
        url="https://www.xiaohongshu.com/explore/abc123",
    )

    assert result.success is True
    assert result.extracted is not None
    assert result.keywords is not None
    assert result.recreated is not None
    assert result.quality is not None
    assert len(result.platform_contents) == 2
    assert result.error is None


@pytest.mark.asyncio
async def test_pipeline_default_platforms():
    """测试默认平台为豆包和DeepSeek"""
    pipeline = _build_pipeline()
    result = await pipeline.run(url="https://www.xiaohongshu.com/explore/abc123")

    assert result.success is True
    # 验证 adapt 被调用时传入了默认平台
    adapter_call = pipeline._adapter.adapt
    call_args = adapter_call.call_args
    platforms_arg = call_args[1].get("platforms") or call_args[0][1]
    assert platforms_arg == [Platform.DOUBAO, Platform.DEEPSEEK]


@pytest.mark.asyncio
async def test_pipeline_custom_platforms():
    """测试自定义平台列表"""
    pipeline = _build_pipeline()
    result = await pipeline.run(
        url="https://www.xiaohongshu.com/explore/abc123",
        platforms=[Platform.BAIDU_AI, Platform.TENCENT_YUANBAO],
    )

    assert result.success is True
    adapter_call = pipeline._adapter.adapt
    call_args = adapter_call.call_args
    platforms_arg = call_args[1].get("platforms") or call_args[0][1]
    assert platforms_arg == [Platform.BAIDU_AI, Platform.TENCENT_YUANBAO]


@pytest.mark.asyncio
async def test_pipeline_extract_failure():
    """测试提取步骤失败时返回错误并保留空结果"""
    pipeline = _build_pipeline(
        extract_result={"success": False, "error": "链接格式不正确"}
    )
    result = await pipeline.run(url="https://invalid-url.com")

    assert result.success is False
    assert "提取步骤失败" in result.error
    assert "链接格式不正确" in result.error
    assert result.extracted is None
    assert result.keywords is None
    assert result.recreated is None
    assert result.quality is None
    assert result.platform_contents == []


@pytest.mark.asyncio
async def test_pipeline_extract_exception():
    """测试提取步骤抛出异常时的处理"""
    pipeline = _build_pipeline(
        extract_side_effect=RuntimeError("网络连接超时")
    )
    result = await pipeline.run(url="https://www.xiaohongshu.com/explore/abc123")

    assert result.success is False
    assert "提取步骤异常" in result.error
    assert result.extracted is None


@pytest.mark.asyncio
async def test_pipeline_analyze_failure():
    """测试关键词分析步骤失败时保留已完成的提取结果"""
    pipeline = _build_pipeline(
        analyze_side_effect=RuntimeError("LLM 调用失败")
    )
    result = await pipeline.run(url="https://www.xiaohongshu.com/explore/abc123")

    assert result.success is False
    assert "关键词分析步骤异常" in result.error
    # 提取结果应保留
    assert result.extracted is not None
    # 后续步骤结果应为空
    assert result.keywords is None
    assert result.recreated is None


@pytest.mark.asyncio
async def test_pipeline_recreate_failure():
    """测试二创步骤失败时保留提取和分析结果"""
    pipeline = _build_pipeline(
        recreate_side_effect=RuntimeError("模板生成失败")
    )
    result = await pipeline.run(url="https://www.xiaohongshu.com/explore/abc123")

    assert result.success is False
    assert "内容二创步骤异常" in result.error
    assert result.extracted is not None
    assert result.keywords is not None
    assert result.recreated is None
    assert result.quality is None


@pytest.mark.asyncio
async def test_pipeline_score_failure():
    """测试评分步骤失败时保留提取、分析和二创结果"""
    pipeline = _build_pipeline(
        score_side_effect=RuntimeError("评分引擎异常")
    )
    result = await pipeline.run(url="https://www.xiaohongshu.com/explore/abc123")

    assert result.success is False
    assert "质量评分步骤异常" in result.error
    assert result.extracted is not None
    assert result.keywords is not None
    assert result.recreated is not None
    assert result.quality is None


@pytest.mark.asyncio
async def test_pipeline_adapt_failure():
    """测试适配步骤失败时保留前四步结果"""
    pipeline = _build_pipeline(
        adapt_side_effect=RuntimeError("平台适配异常")
    )
    result = await pipeline.run(url="https://www.xiaohongshu.com/explore/abc123")

    assert result.success is False
    assert "平台适配步骤异常" in result.error
    assert result.extracted is not None
    assert result.keywords is not None
    assert result.recreated is not None
    assert result.quality is not None
    assert result.platform_contents == []


@pytest.mark.asyncio
async def test_pipeline_passes_template_and_business_type():
    """测试 template_type 和 business_type 正确传递给二创模块"""
    pipeline = _build_pipeline()
    await pipeline.run(
        url="https://www.xiaohongshu.com/explore/abc123",
        template_type=TemplateType.RANKING,
        business_type=BusinessType.LOCAL,
    )

    recreator_call = pipeline._recreator.recreate
    call_kwargs = recreator_call.call_args[1]
    assert call_kwargs["template_type"] == TemplateType.RANKING
    assert call_kwargs["business_type"] == BusinessType.LOCAL
