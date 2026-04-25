"""GEO 内容二创引擎

基于 GEO 模板和 DeepSeek API 进行内容二创。
核心功能：
- 根据内容特征自动推荐模板类型
- 基于"属性词+核心词+意图词+修饰词"公式生成候选标题
- 诊断标题的关键词匹配度
- 根据业务类型生成转化策略
- 组合模板、关键词和 LLM 生成二创内容
"""

import logging
from datetime import datetime
from typing import Any

from geo_tool.models.content import ExtractedContent
from geo_tool.models.enums import BusinessType, TemplateType
from geo_tool.models.keyword import KeywordMatrix
from geo_tool.models.recreated import ConversionStrategy, RecreatedContent
from geo_tool.models.title import Title, TitleDiagnosis
from geo_tool.modules.llm_client import LLMClient
from geo_tool.modules.templates import get_template

logger = logging.getLogger(__name__)


# 常见意图词列表，用于标题诊断
_INTENT_WORDS = [
    "推荐", "对比", "怎么选", "哪个好", "测评", "评测",
    "攻略", "指南", "教程", "排行", "榜单", "避坑",
    "必看", "必备", "盘点", "合集", "选购", "入门",
    "实测", "体验", "分享", "总结", "分析", "解读",
]

# 常见时效词模式，用于标题诊断
_TIMELINESS_PATTERNS = [
    "2024", "2025", "2026", "最新", "今年", "本年度",
    "上半年", "下半年", "Q1", "Q2", "Q3", "Q4",
    "春季", "夏季", "秋季", "冬季", "年度",
]

# 常见属性词列表，用于标题诊断
_ATTRIBUTE_WORDS = [
    "高端", "平价", "专业", "入门级", "旗舰", "性价比",
    "便携", "家用", "商用", "学生", "新手", "进阶",
    "全面", "深度", "详细", "完整", "真实", "客观",
    "Top", "TOP", "top", "最佳", "优质", "热门",
]


class ContentRecreator:
    """GEO 内容二创引擎

    通过 LLM 和 GEO 模板对小红书笔记内容进行二次创作，
    生成符合 AI 搜索引擎引用标准的高质量内容。
    """

    # 模板类型与内容特征关键词的映射（用于自动推荐）
    _TEMPLATE_KEYWORDS: dict[TemplateType, list[str]] = {
        TemplateType.RANKING: [
            "排行", "榜单", "Top", "推荐", "排名", "最佳", "盘点", "合集",
        ],
        TemplateType.REVIEW: [
            "评测", "测评", "体验", "开箱", "使用感受", "优缺点", "实测",
        ],
        TemplateType.GUIDE: [
            "攻略", "教程", "指南", "怎么", "如何", "步骤", "方法", "技巧",
        ],
        TemplateType.COMPARISON: [
            "对比", "vs", "VS", "比较", "区别", "哪个好", "怎么选",
        ],
        TemplateType.CASE_STUDY: [
            "案例", "经验", "复盘", "实战", "故事", "心得", "成果",
        ],
    }

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化内容二创引擎

        Args:
            llm_client: DeepSeek API 客户端实例（依赖注入）
        """
        self._llm = llm_client

    async def recreate(
        self,
        content: ExtractedContent,
        keywords: KeywordMatrix,
        template_type: TemplateType | None = None,
        business_type: BusinessType | None = None,
    ) -> RecreatedContent:
        """基于模板进行内容二创

        完整流程：选择模板 → 生成标题 → 填充模板并调用 LLM 生成正文 → 生成转化策略。

        Args:
            content: 提取的小红书笔记内容
            keywords: GEO 关键词矩阵
            template_type: 指定的模板类型，为 None 时自动推荐
            business_type: 业务类型（线上/本地），用于生成转化策略

        Returns:
            二创内容，包含标题、正文和转化策略
        """
        # 第一步：确定模板类型
        if template_type is None:
            template_type = self._select_template(content)
            logger.info("自动推荐模板类型: %s", template_type.value)
        else:
            logger.info("使用指定模板类型: %s", template_type.value)

        # 第二步：生成候选标题
        logger.info("开始生成候选标题...")
        titles = await self._generate_titles(keywords, template_type)
        logger.info("生成了 %d 个候选标题", len(titles))

        # 第三步：获取模板并填充占位符，调用 LLM 生成正文
        logger.info("开始生成二创正文...")
        body = await self._generate_body(content, keywords, template_type)
        logger.info("正文生成完成，长度: %d 字符", len(body))

        # 第四步：生成转化策略（如果指定了业务类型）
        conversion: ConversionStrategy | None = None
        if business_type is not None:
            logger.info("开始生成转化策略，业务类型: %s", business_type.value)
            conversion = await self._generate_conversion_elements(
                template_type, business_type
            )

        # 收集使用的关键词
        keywords_used = [kw.word for kw in keywords.sorted_by_value()[:10]]

        return RecreatedContent(
            template_type=template_type,
            titles=titles,
            body=body,
            keywords_used=keywords_used,
            conversion=conversion,
            created_at=datetime.now(),
        )

    def _select_template(self, content: ExtractedContent) -> TemplateType:
        """根据内容特征自动推荐模板类型

        通过匹配内容文本中的特征关键词来判断最适合的模板类型。
        如果没有明显匹配，默认推荐攻略类模板（适用范围最广）。

        Args:
            content: 提取的小红书笔记内容

        Returns:
            推荐的模板类型
        """
        text = f"{content.metadata.title} {content.text}".lower()

        # 统计每种模板类型的关键词匹配数
        scores: dict[TemplateType, int] = {}
        for tpl_type, kw_list in self._TEMPLATE_KEYWORDS.items():
            score = sum(1 for kw in kw_list if kw.lower() in text)
            scores[tpl_type] = score

        # 选择匹配度最高的模板类型
        best_type = max(scores, key=lambda t: scores[t])

        # 如果最高分为0（没有任何匹配），默认使用攻略类
        if scores[best_type] == 0:
            return TemplateType.GUIDE

        return best_type

    async def _generate_titles(
        self, keywords: KeywordMatrix, template_type: TemplateType
    ) -> list[Title]:
        """基于"属性词+核心词+意图词+修饰词"公式生成候选标题

        通过 LLM 生成至少3个候选标题，每个标题附带诊断结果。

        Args:
            keywords: GEO 关键词矩阵
            template_type: 模板类型

        Returns:
            候选标题列表（至少3个），每个包含诊断信息
        """
        # 提取核心关键词（取商业价值最高的前5个）
        top_keywords = [kw.word for kw in keywords.sorted_by_value()[:5]]
        core_topics = keywords.core_topics[:3]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的 GEO 标题优化专家。\n"
                    "请基于「属性词+核心词+意图词+修饰词」公式生成候选标题。\n\n"
                    "标题公式说明：\n"
                    "- 属性词：描述特征的词（如「高端」「平价」「Top10」「深度」）\n"
                    "- 核心词：内容的核心关键词\n"
                    "- 意图词：用户搜索意图（如「推荐」「对比」「怎么选」「攻略」）\n"
                    "- 修饰词：增强吸引力的词（如「2025年」「必看」「亲测有效」）\n\n"
                    "要求：\n"
                    "1. 每个标题长度控制在15-30个字符之间\n"
                    "2. 每个标题必须包含核心关键词\n"
                    "3. 每个标题必须包含意图词（推荐/对比/怎么选/攻略/评测等）\n"
                    "4. 每个标题必须包含时效词（如2025年）\n"
                    "5. 生成至少5个候选标题\n\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"模板类型：{template_type.value}\n"
                    f"核心主题：{', '.join(core_topics)}\n"
                    f"高价值关键词：{', '.join(top_keywords)}\n\n"
                    f"请生成至少5个候选标题。\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"titles": ["标题1", "标题2", "标题3", "标题4", "标题5"]}}'
                ),
            },
        ]

        result = await self._llm.chat_json(messages, temperature=0.7)
        raw_titles = result.get("titles", [])

        # 确保至少有3个标题
        title_texts = [str(t).strip() for t in raw_titles if t]
        if len(title_texts) < 3:
            # 补充默认标题
            fallback_keyword = top_keywords[0] if top_keywords else "内容"
            fallback_titles = [
                f"2025年{fallback_keyword}推荐榜单｜深度解析",
                f"2025年{fallback_keyword}怎么选？专业攻略指南",
                f"2025年{fallback_keyword}对比评测｜真实体验分享",
            ]
            while len(title_texts) < 3:
                title_texts.append(fallback_titles[len(title_texts)])

        # 为每个标题生成诊断
        titles: list[Title] = []
        for text in title_texts:
            diagnosis = self._diagnose_title(text, keywords)
            titles.append(Title(text=text, diagnosis=diagnosis))

        return titles

    def _diagnose_title(self, title: str, keywords: KeywordMatrix) -> TitleDiagnosis:
        """诊断标题的关键词匹配度

        检查标题是否包含核心关键词、意图词、时效词、属性词，
        计算匹配分数（0-100），检查长度是否在15-30字符之间。

        Args:
            title: 待诊断的标题文本
            keywords: GEO 关键词矩阵（用于检查核心关键词匹配）

        Returns:
            标题诊断结果
        """
        # 检查是否包含核心关键词
        core_words = keywords.core_topics + [kw.word for kw in keywords.keywords[:5]]
        has_core_keyword = any(word in title for word in core_words if word)

        # 检查是否包含意图词
        has_intent_word = any(word in title for word in _INTENT_WORDS)

        # 检查是否包含时效词
        has_timeliness_word = any(
            pattern in title for pattern in _TIMELINESS_PATTERNS
        )

        # 检查是否包含属性词
        has_attribute_word = any(word in title for word in _ATTRIBUTE_WORDS)

        # 检查标题长度（15-30字符）
        title_len = len(title)
        length_valid = 15 <= title_len <= 30

        # 计算匹配分数（0-100）
        score = 0.0
        # 核心关键词匹配：30分
        if has_core_keyword:
            score += 30.0
        # 意图词匹配：25分
        if has_intent_word:
            score += 25.0
        # 时效词匹配：20分
        if has_timeliness_word:
            score += 20.0
        # 属性词匹配：15分
        if has_attribute_word:
            score += 15.0
        # 长度合规：10分
        if length_valid:
            score += 10.0

        # 生成优化建议
        suggestions: list[str] = []
        if not has_core_keyword:
            suggestions.append("建议在标题中加入核心关键词，提高搜索匹配度")
        if not has_intent_word:
            suggestions.append("建议加入意图词（如「推荐」「怎么选」「攻略」），明确用户搜索意图")
        if not has_timeliness_word:
            suggestions.append("建议加入时效词（如「2025年」），提升内容时效性信号")
        if not has_attribute_word:
            suggestions.append("建议加入属性词（如「深度」「专业」「Top」），增强标题吸引力")
        if not length_valid:
            if title_len < 15:
                suggestions.append(f"标题过短（{title_len}字符），建议扩展到15-30字符")
            else:
                suggestions.append(f"标题过长（{title_len}字符），建议精简到15-30字符")

        return TitleDiagnosis(
            has_core_keyword=has_core_keyword,
            has_intent_word=has_intent_word,
            has_timeliness_word=has_timeliness_word,
            has_attribute_word=has_attribute_word,
            length_valid=length_valid,
            match_score=score,
            suggestions=suggestions,
        )

    async def _generate_conversion_elements(
        self, template_type: TemplateType, business_type: BusinessType
    ) -> ConversionStrategy:
        """根据业务类型生成转化策略

        线上业务推荐：电话、官网、公众号、微信号
        本地业务推荐：门店地址、营业时间、电话、地图导航

        Args:
            template_type: 模板类型
            business_type: 业务类型（线上/本地）

        Returns:
            转化策略，包含联系方式植入建议、信任元素、转化话术和平台限流提醒
        """
        # 根据业务类型确定联系方式植入建议
        if business_type == BusinessType.ONLINE:
            contact_placements = [
                "在文章结尾植入电话号码，引导电话咨询",
                "在文章开头或结尾植入官网链接，引导访问官网",
                "在文章末尾引导关注公众号，获取更多资讯",
                "在评论区或文末植入微信号，引导添加好友咨询",
            ]
        else:
            # 本地业务
            contact_placements = [
                "在文章结尾植入门店地址，方便用户到店体验",
                "在文章中标注营业时间，方便用户规划到店时间",
                "在文章结尾植入电话号码，引导电话预约",
                "在文章末尾植入地图导航链接，方便用户导航到店",
            ]

        # 根据模板类型生成信任建立元素
        trust_elements = self._build_trust_elements(template_type)

        # 生成转化话术
        call_to_action = self._build_call_to_action(template_type, business_type)

        # 平台限流提醒
        platform_warnings = [
            "小红书：避免在正文中直接放置微信号和电话，可在评论区引导",
            "百家号：正文中可适当植入联系方式，但避免过度营销",
            "微信公众号：可自由植入联系方式，但注意内容质量",
            "今日头条：避免在标题中出现联系方式，正文末尾可适当植入",
        ]

        return ConversionStrategy(
            contact_placements=contact_placements,
            trust_elements=trust_elements,
            call_to_action=call_to_action,
            platform_warnings=platform_warnings,
        )

    async def _generate_body(
        self,
        content: ExtractedContent,
        keywords: KeywordMatrix,
        template_type: TemplateType,
    ) -> str:
        """使用模板和 LLM 生成二创正文

        获取对应模板，填充占位符后发送给 LLM 生成内容。

        Args:
            content: 提取的小红书笔记内容
            keywords: GEO 关键词矩阵
            template_type: 模板类型

        Returns:
            生成的二创正文
        """
        # 获取模板
        template = get_template(template_type)

        # 准备关键词字符串
        top_keywords = [kw.word for kw in keywords.sorted_by_value()[:8]]
        keywords_str = "、".join(top_keywords)

        # 准备通用占位符
        year = str(datetime.now().year)
        industry = keywords.core_topics[0] if keywords.core_topics else "行业"

        # 构建占位符映射
        placeholders: dict[str, str] = {
            "content": content.text[:2000],  # 限制内容长度，避免超出 token 限制
            "keywords": keywords_str,
            "year": year,
            "industry": industry,
            "authority_source": "艾瑞咨询",
        }

        # 根据模板类型添加特定占位符
        placeholders.update(
            self._get_template_specific_placeholders(
                template_type, content, keywords
            )
        )

        # 填充模板占位符（安全填充，忽略缺失的占位符）
        prompt = self._safe_format(template, placeholders)

        # 调用 LLM 生成正文
        messages = [
            {"role": "user", "content": prompt},
        ]

        body = await self._llm.chat(messages, temperature=0.7, max_tokens=4096)
        return body

    def _get_template_specific_placeholders(
        self,
        template_type: TemplateType,
        content: ExtractedContent,
        keywords: KeywordMatrix,
    ) -> dict[str, str]:
        """根据模板类型获取特定的占位符

        不同模板需要不同的占位符参数，此方法根据模板类型
        从内容和关键词中提取合适的值。

        Args:
            template_type: 模板类型
            content: 提取的笔记内容
            keywords: 关键词矩阵

        Returns:
            模板特定的占位符字典
        """
        title = content.metadata.title
        topic = keywords.core_topics[0] if keywords.core_topics else title

        if template_type == TemplateType.RANKING:
            return {
                "list_count": "10",
                "benefit": "专业选购指南",
            }
        elif template_type == TemplateType.REVIEW:
            return {
                "product_name": topic,
            }
        elif template_type == TemplateType.GUIDE:
            return {
                "guide_topic": topic,
            }
        elif template_type == TemplateType.COMPARISON:
            return {
                "comparison_items": topic,
            }
        elif template_type == TemplateType.CASE_STUDY:
            return {
                "case_topic": topic,
            }
        return {}

    @staticmethod
    def _build_trust_elements(template_type: TemplateType) -> list[str]:
        """根据模板类型构建信任建立元素

        Args:
            template_type: 模板类型

        Returns:
            信任建立元素列表
        """
        # 通用信任元素
        base_elements = [
            "声明专业身份（如「从业10年的行业专家」）",
            "引用权威数据源（如艾瑞咨询、官方统计数据）",
            "展示真实数据和可验证信息",
        ]

        # 模板特定信任元素
        type_specific: dict[TemplateType, list[str]] = {
            TemplateType.RANKING: ["展示评测方法论和评分标准", "引用第三方评测机构数据"],
            TemplateType.REVIEW: ["提供真实使用时长和测试数据", "展示实拍图片和测试截图"],
            TemplateType.GUIDE: ["分享个人实操经验和踩坑记录", "提供可验证的操作步骤"],
            TemplateType.COMPARISON: ["使用统一标准进行客观对比", "引用多个数据源交叉验证"],
            TemplateType.CASE_STUDY: ["提供实施前后的数据对比", "展示可量化的成果指标"],
        }

        return base_elements + type_specific.get(template_type, [])

    @staticmethod
    def _build_call_to_action(
        template_type: TemplateType, business_type: BusinessType
    ) -> list[str]:
        """根据模板类型和业务类型构建转化话术

        Args:
            template_type: 模板类型
            business_type: 业务类型

        Returns:
            转化话术列表
        """
        # 通用转化话术
        cta_list = [
            "「免费领取完整版资料/报告」— 低门槛入门方式",
            "「限时免费咨询名额，仅剩XX位」— 紧迫感营造",
        ]

        # 根据业务类型添加特定话术
        if business_type == BusinessType.ONLINE:
            cta_list.extend([
                "「点击官网了解更多详情」— 引导访问官网",
                "「关注公众号获取独家优惠」— 引导关注公众号",
            ])
        else:
            cta_list.extend([
                "「到店体验可享专属折扣」— 引导到店",
                "「电话预约免排队」— 引导电话预约",
            ])

        return cta_list

    @staticmethod
    def _safe_format(template: str, placeholders: dict[str, str]) -> str:
        """安全地填充模板占位符

        对于模板中存在但 placeholders 中缺失的占位符，
        保留原始占位符文本，避免 KeyError。

        Args:
            template: 模板字符串
            placeholders: 占位符映射

        Returns:
            填充后的模板字符串
        """
        result = template
        for key, value in placeholders.items():
            result = result.replace("{" + key + "}", value)
        return result
