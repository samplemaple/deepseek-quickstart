"""平台适配器

针对不同 AI 搜索平台特性优化内容，纯配置和规则引擎实现。
核心功能：
- 维护四大平台（百度AI搜索、豆包、DeepSeek、腾讯元宝）的偏好配置
- 根据平台配置调整内容格式（添加平台特定的头部/尾部建议）
- 为每个目标平台生成适配版本
"""

import logging

from geo_tool.models.enums import Platform
from geo_tool.models.platform import PlatformConfig, PlatformContent
from geo_tool.models.recreated import RecreatedContent

logger = logging.getLogger(__name__)


class PlatformAdapter:
    """平台适配器

    基于预定义的平台偏好配置，对二创内容进行格式调整和发布建议生成。
    不依赖 LLM，纯配置和规则引擎驱动。
    """

    # 四大平台偏好配置
    _PLATFORM_CONFIGS: dict[Platform, dict] = {
        Platform.BAIDU_AI: {
            "recommended_channels": [
                "百家号",
                "百度百科",
                "百度知道",
                "百度经验",
            ],
            "content_format_rules": [
                "标题包含核心关键词，便于百度搜索召回",
                "正文使用清晰的 H2/H3 层级标题结构",
                "段落间适当留白，提升可读性",
                "引用权威数据源时标注出处，增强可信度",
                "适当使用表格和列表呈现对比信息",
            ],
            "notes": [
                "百度AI搜索偏好百度生态内容，优先在百家号发布",
                "适合 B2B 决策和专业知识查询场景",
                "内容应体现专业性和权威性，引用行业报告和官方数据",
                "避免过度营销话术，百度对广告内容有严格审核",
                "百家号文章建议 1500-3000 字，信息密度要高",
            ],
        },
        Platform.DOUBAO: {
            "recommended_channels": [
                "今日头条",
                "抖音",
                "头条号",
                "西瓜视频",
            ],
            "content_format_rules": [
                "标题突出实用性和吸引力，适当使用数字和疑问句式",
                "正文开头直接给出核心结论，抓住读者注意力",
                "多使用短段落和列表，适配移动端阅读习惯",
                "融入生活化场景描述，增强代入感",
                "适当添加互动引导语，提升用户参与度",
            ],
            "notes": [
                "豆包偏好字节生态内容，优先在今日头条和抖音发布",
                "适合消费类和生活服务类场景",
                "内容风格偏轻松实用，贴近日常生活",
                "头条号文章注意配图质量，图文并茂效果更好",
                "抖音相关内容可考虑图文笔记形式发布",
            ],
        },
        Platform.DEEPSEEK: {
            "recommended_channels": [
                "CSDN",
                "博客园",
                "知乎专栏",
                "掘金",
            ],
            "content_format_rules": [
                "标题精准描述内容主题，避免标题党",
                "正文逻辑严谨，论据充分，数据详实",
                "使用 Markdown 格式，代码块和技术术语规范",
                "提供可操作的方法论和具体步骤",
                "引用学术论文、行业白皮书等高质量信息源",
            ],
            "notes": [
                "DeepSeek 偏好技术社区和专业平台内容",
                "适合高知人群和专业服务场景",
                "内容应体现深度思考和专业见解",
                "CSDN/博客园文章建议结构清晰、干货密度高",
                "知乎专栏适合发布深度分析和行业洞察类内容",
            ],
        },
        Platform.TENCENT_YUANBAO: {
            "recommended_channels": [
                "微信公众号",
                "腾讯新闻",
                "腾讯内容开放平台",
            ],
            "content_format_rules": [
                "标题简洁有力，适配微信分享场景",
                "正文排版精美，善用引用块和分割线",
                "内容深度优先，适合长文阅读场景",
                "融入个人经验和真实案例，增强种草效果",
                "结尾设置互动话题或引导关注，提升传播力",
            ],
            "notes": [
                "腾讯元宝偏好微信生态内容，优先在公众号发布",
                "适合深度种草和专业咨询场景",
                "公众号文章注重排版美观和阅读体验",
                "内容应有独特观点和深度分析，避免同质化",
                "注意微信平台对外链和联系方式的限制规则",
            ],
        },
    }

    def _get_platform_config(self, platform: Platform) -> PlatformConfig:
        """获取平台偏好配置

        Args:
            platform: 目标 AI 搜索平台

        Returns:
            对应平台的偏好配置，包含推荐渠道、格式要求和注意事项
        """
        config_data = self._PLATFORM_CONFIGS[platform]
        return PlatformConfig(
            platform=platform,
            recommended_channels=config_data["recommended_channels"],
            content_format_rules=config_data["content_format_rules"],
            notes=config_data["notes"],
        )

    def _adapt_content(
        self, content: RecreatedContent, config: PlatformConfig
    ) -> PlatformContent:
        """根据平台配置调整内容格式

        对正文做简单的格式调整，添加平台特定的头部和尾部建议。

        Args:
            content: 二创内容
            config: 平台偏好配置

        Returns:
            适配后的平台内容
        """
        # 构建平台特定的头部提示
        platform_name = self._get_platform_display_name(config.platform)
        header = f"【{platform_name}适配版本】\n\n"

        # 构建推荐发布渠道提示
        channels_str = "、".join(config.recommended_channels)
        header += f"📢 推荐发布渠道：{channels_str}\n\n"

        # 原始正文
        body = content.body

        # 构建平台特定的尾部建议
        footer = "\n\n---\n"
        footer += f"📋 {platform_name}发布注意事项：\n"
        for i, note in enumerate(config.notes, 1):
            footer += f"  {i}. {note}\n"

        # 组合适配后的正文
        adapted_body = header + body + footer

        return PlatformContent(
            platform=config.platform,
            adapted_body=adapted_body,
            publish_suggestions=config,
        )

    async def adapt(
        self,
        content: RecreatedContent,
        platforms: list[Platform],
    ) -> list[PlatformContent]:
        """为每个目标平台生成适配版本

        Args:
            content: 二创内容
            platforms: 目标平台列表

        Returns:
            与输入平台数量相同的适配内容列表，每个对应一个不同的平台
        """
        results: list[PlatformContent] = []

        for platform in platforms:
            logger.info("正在为平台 %s 生成适配版本", platform.value)
            config = self._get_platform_config(platform)
            adapted = self._adapt_content(content, config)
            results.append(adapted)

        logger.info("平台适配完成，共生成 %d 个版本", len(results))
        return results

    @staticmethod
    def _get_platform_display_name(platform: Platform) -> str:
        """获取平台的中文显示名称"""
        names = {
            Platform.BAIDU_AI: "百度AI搜索",
            Platform.DOUBAO: "豆包",
            Platform.DEEPSEEK: "DeepSeek",
            Platform.TENCENT_YUANBAO: "腾讯元宝",
        }
        return names.get(platform, platform.value)
