"""枚举类型定义"""

from enum import Enum


class TemplateType(str, Enum):
    """GEO内容模板类型"""
    RANKING = "ranking"       # 榜单式
    REVIEW = "review"         # 评测类
    GUIDE = "guide"           # 攻略类
    COMPARISON = "comparison" # 对比类
    CASE_STUDY = "case_study" # 案例类


class Platform(str, Enum):
    """目标AI搜索平台"""
    BAIDU_AI = "baidu_ai"       # 百度AI搜索
    DOUBAO = "doubao"           # 豆包
    DEEPSEEK = "deepseek"       # DeepSeek
    TENCENT_YUANBAO = "yuanbao" # 腾讯元宝


class KeywordType(str, Enum):
    """GEO关键词类型"""
    LOCATION = "location"     # 定位词
    COGNITION = "cognition"   # 认知词
    SCENARIO = "scenario"     # 场景词
    TRUST = "trust"           # 信任词
    BRAND = "brand"           # 品牌词


class DecisionStage(str, Enum):
    """用户决策阶段"""
    AWARENESS = "awareness"     # 了解期
    EXPLORATION = "exploration" # 摸索期
    EVALUATION = "evaluation"   # 评估期
    DECISION = "decision"       # 决策期


class BusinessType(str, Enum):
    """业务类型"""
    ONLINE = "online" # 线上业务
    LOCAL = "local"   # 本地业务


class QualityLevel(str, Enum):
    """内容质量等级"""
    EXCELLENT = "excellent"   # 优秀 (>=85)
    GOOD = "good"             # 良好 (70-84)
    NEEDS_WORK = "needs_work" # 需优化 (<70)
