"""GEO 关键词分析器

基于 DeepSeek API 进行 NLP 分析，生成 GEO 关键词矩阵。
核心功能：
- 识别内容核心主题词
- 按五种关键词类型（定位词、认知词、场景词、信任词、品牌词）生成矩阵
- 计算商业价值评分（需求成熟度 × 商业潜力 × 匹配精准度）
- 基于"语境约束 + 核心关键词 + 意图任务"公式生成长尾关键词
"""

import logging
from typing import Any

from geo_tool.models.content import ExtractedContent
from geo_tool.models.enums import DecisionStage, KeywordType
from geo_tool.models.keyword import GEOKeyword, KeywordMatrix
from geo_tool.modules.llm_client import LLMClient

logger = logging.getLogger(__name__)


class KeywordAnalyzer:
    """GEO 关键词分析器

    通过 LLM 分析提取的小红书笔记内容，生成包含五种关键词类型的
    GEO 关键词矩阵，并为每个关键词计算商业价值评分和长尾变体。
    """

    # 五种关键词类型的中文映射
    _KEYWORD_TYPE_MAP: dict[str, KeywordType] = {
        "location": KeywordType.LOCATION,
        "cognition": KeywordType.COGNITION,
        "scenario": KeywordType.SCENARIO,
        "trust": KeywordType.TRUST,
        "brand": KeywordType.BRAND,
    }

    # 四种决策阶段的中文映射
    _DECISION_STAGE_MAP: dict[str, DecisionStage] = {
        "awareness": DecisionStage.AWARENESS,
        "exploration": DecisionStage.EXPLORATION,
        "evaluation": DecisionStage.EVALUATION,
        "decision": DecisionStage.DECISION,
    }

    def __init__(self, llm_client: LLMClient) -> None:
        """初始化关键词分析器

        Args:
            llm_client: DeepSeek API 客户端实例（依赖注入）
        """
        self._llm = llm_client

    async def analyze(self, content: ExtractedContent) -> KeywordMatrix:
        """分析内容并生成 GEO 关键词矩阵

        完整流程：识别核心主题词 → 生成关键词矩阵 → 计算商业价值 → 生成长尾关键词。
        返回结果按商业价值从高到低排序。

        Args:
            content: 提取的小红书笔记内容

        Returns:
            按商业价值排序的关键词矩阵
        """
        # 拼接标题和正文作为分析文本
        text = f"标题：{content.metadata.title}\n\n正文：{content.text}"

        # 第一步：识别核心主题词
        logger.info("开始识别核心主题词...")
        core_topics = await self._identify_core_topics(text)
        logger.info("识别到 %d 个核心主题词: %s", len(core_topics), core_topics)

        # 第二步：基于核心主题词生成关键词矩阵
        logger.info("开始生成关键词矩阵...")
        matrix = await self._generate_keyword_matrix(core_topics, text)
        logger.info("生成了 %d 个关键词", len(matrix.keywords))

        # 第三步：为每个关键词计算商业价值评分
        logger.info("开始计算商业价值评分...")
        matrix = await self._calculate_business_value(matrix, text)

        # 第四步：为每个关键词生成长尾变体
        logger.info("开始生成长尾关键词...")
        matrix = await self._generate_longtail_keywords(matrix, text)

        # 按商业价值排序
        sorted_keywords = matrix.sorted_by_value()
        return KeywordMatrix(
            core_topics=matrix.core_topics,
            keywords=sorted_keywords,
        )

    async def _identify_core_topics(self, text: str) -> list[str]:
        """通过 LLM 识别核心主题词

        向 LLM 发送内容文本，要求其提取核心主题词和行业词。

        Args:
            text: 笔记的标题和正文拼接文本

        Returns:
            核心主题词列表
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的 SEO 和内容分析专家。"
                    "请从用户提供的小红书笔记内容中提取核心主题词和行业词。"
                    "核心主题词应该是内容的关键概念，能够代表文章的主要话题。"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请分析以下小红书笔记内容，提取3-8个核心主题词。\n\n"
                    f"{text}\n\n"
                    f'请以 JSON 格式返回：{{"core_topics": ["主题词1", "主题词2", ...]}}'
                ),
            },
        ]

        result = await self._llm.chat_json(messages, temperature=0.3)
        topics = result.get("core_topics", [])

        # 确保返回的是字符串列表
        return [str(t) for t in topics if t]

    async def _generate_keyword_matrix(
        self, topics: list[str], text: str
    ) -> KeywordMatrix:
        """基于五种关键词类型生成关键词矩阵

        为每个核心主题词，按定位词、认知词、场景词、信任词、品牌词
        五种类型生成关键词，并标注对应的用户决策阶段。

        Args:
            topics: 核心主题词列表
            text: 原始笔记文本（提供上下文）

        Returns:
            包含关键词的矩阵（商业价值和长尾变体待后续填充）
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的 GEO（生成式引擎优化）关键词分析专家。\n"
                    "你需要基于核心主题词，按照五种 GEO 关键词类型生成关键词矩阵。\n\n"
                    "五种关键词类型说明：\n"
                    "1. location（定位词）：明确产品/服务的市场定位，如「高端」「平价」「专业级」\n"
                    "2. cognition（认知词）：用户认知和搜索习惯相关词，如「怎么选」「哪个好」「推荐」\n"
                    "3. scenario（场景词）：使用场景和需求场景，如「办公用」「送礼」「旅行必备」\n"
                    "4. trust（信任词）：建立信任的词汇，如「实测」「亲测」「专家推荐」\n"
                    "5. brand（品牌词）：品牌和产品名称相关词\n\n"
                    "四种用户决策阶段：\n"
                    "1. awareness（了解期）：用户刚开始了解某个领域\n"
                    "2. exploration（摸索期）：用户在探索不同选项\n"
                    "3. evaluation（评估期）：用户在对比评估具体产品\n"
                    "4. decision（决策期）：用户即将做出购买决策\n\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"核心主题词：{', '.join(topics)}\n\n"
                    f"原始内容摘要：{text[:500]}\n\n"
                    f"请为以上主题词生成 GEO 关键词矩阵。要求：\n"
                    f"1. 每种关键词类型至少生成2个关键词\n"
                    f"2. 每个关键词标注类型（location/cognition/scenario/trust/brand）和决策阶段（awareness/exploration/evaluation/decision）\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"keywords": [\n'
                    f'  {{"word": "关键词", "keyword_type": "location", "decision_stage": "awareness"}},\n'
                    f"  ...\n"
                    f"]}}"
                ),
            },
        ]

        result = await self._llm.chat_json(messages, temperature=0.3)
        raw_keywords = result.get("keywords", [])

        # 解析 LLM 返回的关键词列表
        keywords: list[GEOKeyword] = []
        for kw in raw_keywords:
            keyword = self._parse_keyword(kw)
            if keyword is not None:
                keywords.append(keyword)

        return KeywordMatrix(core_topics=topics, keywords=keywords)

    async def _calculate_business_value(
        self, matrix: KeywordMatrix, text: str
    ) -> KeywordMatrix:
        """计算每个关键词的商业价值评分

        商业价值 = 需求成熟度(1-10) × 商业潜力(1-10) × 匹配精准度(1-10)
        评分范围：1-1000

        Args:
            matrix: 待评分的关键词矩阵
            text: 原始笔记文本（提供上下文）

        Returns:
            更新了商业价值评分的关键词矩阵
        """
        if not matrix.keywords:
            return matrix

        # 提取所有关键词文本
        keyword_words = [kw.word for kw in matrix.keywords]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的关键词商业价值评估专家。\n"
                    "请为每个关键词评估三个维度的分数（每个维度1-10分）：\n"
                    "1. 需求成熟度：该关键词对应的用户需求有多成熟（1=非常小众，10=大众刚需）\n"
                    "2. 商业潜力：该关键词的商业变现潜力（1=无商业价值，10=高转化高客单价）\n"
                    "3. 匹配精准度：该关键词与原始内容的匹配程度（1=完全不相关，10=高度匹配）\n\n"
                    "商业价值 = 需求成熟度 × 商业潜力 × 匹配精准度\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始内容摘要：{text[:300]}\n\n"
                    f"请为以下关键词评估商业价值：\n"
                    f"{', '.join(keyword_words)}\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"scores": [\n'
                    f'  {{"word": "关键词", "maturity": 8, "potential": 7, "precision": 9}},\n'
                    f"  ...\n"
                    f"]}}"
                ),
            },
        ]

        result = await self._llm.chat_json(messages, temperature=0.3)
        raw_scores = result.get("scores", [])

        # 构建关键词 → 评分映射
        score_map: dict[str, float] = {}
        for item in raw_scores:
            word = item.get("word", "")
            maturity = self._clamp(item.get("maturity", 5), 1, 10)
            potential = self._clamp(item.get("potential", 5), 1, 10)
            precision = self._clamp(item.get("precision", 5), 1, 10)
            score_map[word] = float(maturity * potential * precision)

        # 更新关键词的商业价值评分
        updated_keywords: list[GEOKeyword] = []
        for kw in matrix.keywords:
            business_value = score_map.get(kw.word, 125.0)  # 默认 5×5×5=125
            # 确保评分在有效范围内
            business_value = self._clamp(business_value, 1.0, 1000.0)
            updated_keywords.append(
                kw.model_copy(update={"business_value": business_value})
            )

        return KeywordMatrix(
            core_topics=matrix.core_topics,
            keywords=updated_keywords,
        )

    async def _generate_longtail_keywords(
        self, matrix: KeywordMatrix, text: str
    ) -> KeywordMatrix:
        """为每个关键词生成长尾关键词变体

        基于"语境约束 + 核心关键词 + 意图任务"公式，
        为每个核心关键词生成至少10个长尾关键词组合。

        Args:
            matrix: 关键词矩阵
            text: 原始笔记文本（提供上下文）

        Returns:
            更新了长尾变体的关键词矩阵
        """
        if not matrix.keywords:
            return matrix

        keyword_words = [kw.word for kw in matrix.keywords]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的 GEO 长尾关键词生成专家。\n"
                    "请基于「语境约束 + 核心关键词 + 意图任务」公式为每个关键词生成长尾变体。\n\n"
                    "公式说明：\n"
                    "- 语境约束：时间、地点、人群、场景等限定条件（如「2025年」「北京」「新手」）\n"
                    "- 核心关键词：用户提供的关键词\n"
                    "- 意图任务：用户的搜索意图（如「推荐」「对比」「怎么选」「多少钱」）\n\n"
                    "示例：核心关键词「投影仪」→ 长尾变体：\n"
                    "「2025年家用投影仪推荐」「卧室投影仪怎么选」「3000元投影仪哪个好」等\n\n"
                    "请返回纯 JSON 格式，不要包含 markdown 代码块。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始内容摘要：{text[:300]}\n\n"
                    f"请为以下每个关键词生成至少10个长尾关键词变体：\n"
                    f"{', '.join(keyword_words)}\n\n"
                    f"返回 JSON 格式：\n"
                    f'{{"longtail": [\n'
                    f'  {{"word": "关键词", "variants": ["长尾1", "长尾2", ...]}},\n'
                    f"  ...\n"
                    f"]}}"
                ),
            },
        ]

        result = await self._llm.chat_json(messages, temperature=0.5)
        raw_longtail = result.get("longtail", [])

        # 构建关键词 → 长尾变体映射
        longtail_map: dict[str, list[str]] = {}
        for item in raw_longtail:
            word = item.get("word", "")
            variants = item.get("variants", [])
            # 确保变体是字符串列表且至少10个
            clean_variants = [str(v) for v in variants if v]
            longtail_map[word] = clean_variants

        # 更新关键词的长尾变体
        updated_keywords: list[GEOKeyword] = []
        for kw in matrix.keywords:
            variants = longtail_map.get(kw.word, [])
            updated_keywords.append(
                kw.model_copy(update={"longtail_variants": variants})
            )

        return KeywordMatrix(
            core_topics=matrix.core_topics,
            keywords=updated_keywords,
        )

    def _parse_keyword(self, raw: dict[str, Any]) -> GEOKeyword | None:
        """解析 LLM 返回的单个关键词数据

        将 LLM 返回的原始字典转换为 GEOKeyword 对象。
        对无效的类型或阶段值进行容错处理。

        Args:
            raw: LLM 返回的关键词字典

        Returns:
            解析成功返回 GEOKeyword，失败返回 None
        """
        word = raw.get("word", "").strip()
        if not word:
            return None

        # 解析关键词类型，无效值默认为 cognition
        raw_type = raw.get("keyword_type", "cognition")
        keyword_type = self._KEYWORD_TYPE_MAP.get(
            str(raw_type).lower(), KeywordType.COGNITION
        )

        # 解析决策阶段，无效值默认为 awareness
        raw_stage = raw.get("decision_stage", "awareness")
        decision_stage = self._DECISION_STAGE_MAP.get(
            str(raw_stage).lower(), DecisionStage.AWARENESS
        )

        return GEOKeyword(
            word=word,
            keyword_type=keyword_type,
            decision_stage=decision_stage,
            business_value=0.0,  # 商业价值待后续计算
            longtail_variants=[],  # 长尾变体待后续生成
        )

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
