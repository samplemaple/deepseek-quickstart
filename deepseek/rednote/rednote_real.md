# 🌟 小红书爆款文案生成 Agent — 企业级真实工具集成

> 本文档是 `rednote_real.ipynb` 的美化版本，完整展示了基于 DeepSeek LLM 的小红书文案生成 Agent 企业级实现。

---

## 📋 目录

1. [配置与环境管理](#1-配置与环境管理)
2. [客户端初始化](#2-客户端初始化)
3. [商品数据初始化流水线](#3-商品数据初始化流水线)
4. [真实工具函数实现](#4-真实工具函数实现)
5. [工具调度与 Agent 定义](#5-工具调度与-agent-定义)
6. [Agent 核心引擎](#6-agent-核心引擎)
7. [输出格式化](#7-输出格式化)
8. [端到端测试](#8-端到端测试)
9. [导出文案为 Markdown 文件](#9-导出文案为-markdown-文件)

---

## 🏗️ 架构概览

本 Notebook 将教学版 `rednote.ipynb` 中的三个模拟工具替换为真实实现：

| 模拟工具 | 真实工具 | 技术方案 |
|---------|---------|---------|
| `mock_search_web` | `real_search_web` | Tavily 搜索 API + LLM 总结 + Milvus 存储 |
| `mock_query_product_database` | `real_query_product_database` | Milvus 向量数据库语义检索 |
| `mock_generate_emoji` | `real_generate_emoji` | DeepSeek LLM 智能生成 |

```
用户输入产品名称
    │
    ▼
┌─────────────────────────┐
│  generate_rednote()     │
│  ReAct 循环引擎         │
│                         │
│  Thought → Action →     │
│  Observation → ...      │
└────────┬────────────────┘
         │ tool_calls
         ▼
┌─────────────────────────┐
│  Tool Dispatcher        │
│  available_tools 字典    │
├─────────┬───────┬───────┤
│ search  │product│ emoji │
│  _web   │  _db  │ _gen  │
├─────────┼───────┼───────┤
│ Tavily  │Milvus │DeepSeek│
│  API    │向量DB  │ LLM   │
└─────────┴───────┴───────┘
         │
         ▼
   JSON 格式文案输出
```

---

## 1. 配置与环境管理

> 集中管理所有配置、环境变量验证和日志初始化。

### 1.1 依赖导入与日志配置

```python
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="./rednote-agent.log",
    filemode="a",
    encoding="utf-8",
    force=True,
)
logger = logging.getLogger("rednote-agent")
```

### 1.2 全局常量

| 常量 | 值 | 说明 |
|-----|---|------|
| `MILVUS_URI` | `http://localhost:19530` | Milvus 向量数据库地址 |
| `PRODUCT_COLLECTION` | `product_collection` | 商品数据 Collection 名称 |
| `SIMILARITY_THRESHOLD` | `0.5` | 商品查询相似度阈值（COSINE） |
| `UPSERT_SIMILARITY_THRESHOLD` | `0.85` | 增量写入去重阈值（COSINE） |
| `DEFAULT_EMOJIS` | `["✨", "🔥", "💖", "💯", "🎉"]` | LLM 失败时的默认表情 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek 模型名称 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | 中文 Embedding 模型 |

```python
MILVUS_URI = "http://localhost:19530"
PRODUCT_COLLECTION = "product_collection"
SIMILARITY_THRESHOLD = 0.5
DEFAULT_EMOJIS = ["✨", "🔥", "💖", "💯", "🎉"]
DEEPSEEK_MODEL = "deepseek-chat"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
```

### 1.3 环境变量验证

启动时验证所有必需的 API Key，缺失则立即报错阻止后续执行。

```python
REQUIRED_ENV_VARS = {
    "DEEPSEEK_API_KEY": "DeepSeek API 密钥",
    "TAVILY_API_KEY": "Tavily 搜索 API 密钥",
}

def validate_env_vars():
    """验证所有必需的环境变量是否已设置。"""
    missing = []
    for var, desc in REQUIRED_ENV_VARS.items():
        if not os.getenv(var):
            missing.append(f"  - {var}: {desc}")
    if missing:
        raise ValueError(
            "缺少以下必需的环境变量:\n" + "\n".join(missing)
        )

validate_env_vars()
logger.info("环境变量验证通过")
```

---

## 2. 客户端初始化

> 初始化 DeepSeek LLM 客户端和 Embedding 模型。

```python
from openai import OpenAI
from pymilvus import model

# ========== DeepSeek LLM 客户端 ==========
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
logger.info("DeepSeek LLM 客户端初始化完成")

# ========== Embedding 模型 ==========
embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME,
    device='cpu'
)
logger.info("Embedding 模型初始化完成")
```

**技术选型说明：**
- DeepSeek 使用 OpenAI 兼容 SDK，切换模型只需改 `base_url`
- `BAAI/bge-small-zh-v1.5` 是专为中文优化的轻量级 Embedding 模型（512 维）

---

## 3. 商品数据初始化流水线

> 将商品数据向量化后增量写入 Milvus，支持去重和 metric_type 自动迁移。

### 3.1 增量写入逻辑

```
Collection 不存在？ ──Yes──→ 创建 Collection → 全量写入
        │
       No
        │
metric_type 不是 COSINE？ ──Yes──→ 删除旧表 → 重新创建 → 全量写入
        │
       No
        │
逐条检查相似度：
  >= 0.85 → 跳过（已存在）
  <  0.85 → 新增写入
```

```python
from pymilvus import MilvusClient

UPSERT_SIMILARITY_THRESHOLD = 0.85

def init_product_database(products: list[dict]):
    """将商品数据增量写入 Milvus 向量数据库。

    逻辑：
      - 如果 Collection 不存在，创建并全量写入。
      - 如果 Collection 已存在，逐条检查：用商品文本向量在库中搜索，
        相似度 >= UPSERT_SIMILARITY_THRESHOLD 视为已存在（跳过），
        相似度 < UPSERT_SIMILARITY_THRESHOLD 视为新商品（新增）。

    Args:
        products: 商品列表，每条包含 name, ingredients, effects, target_audience, specs 字段
    """
    # 验证 Schema
    required_fields = ["name", "ingredients", "effects", "target_audience", "specs"]
    for i, p in enumerate(products):
        for field in required_fields:
            if not p.get(field):
                raise ValueError(f"商品 #{i} 缺少必需字段 '{field}' 或字段为空")

    logger.info(f"[Product DB] 开始初始化，共 {len(products)} 条商品数据")
    milvus_client = MilvusClient(uri=MILVUS_URI)

    # 将商品信息拼接为完整文本用于 embedding
    texts = [
        f"{p['name']}：核心成分为{p['ingredients']}，功效为{p['effects']}，"
        f"适用人群为{p['target_audience']}，规格为{p['specs']}。"
        for p in products
    ]
    embeddings = embedding_model.encode_documents(texts)
    embedding_dim = len(embeddings[0])

    collection_exists = milvus_client.has_collection(PRODUCT_COLLECTION)

    # 如果已有 Collection 的 metric_type 不是 COSINE，删除重建
    if collection_exists:
        indexes = milvus_client.list_indexes(collection_name=PRODUCT_COLLECTION)
        need_rebuild = False
        if indexes:
            idx_info = milvus_client.describe_index(
                collection_name=PRODUCT_COLLECTION, index_name=indexes[0]
            )
            current_metric = idx_info.get('metric_type', '')
            if current_metric and current_metric != 'COSINE':
                need_rebuild = True
                logger.warning(f"[Product DB] metric_type={current_metric}，需要重建为 COSINE")
        if need_rebuild:
            milvus_client.drop_collection(PRODUCT_COLLECTION)
            collection_exists = False

    if not collection_exists:
        milvus_client.create_collection(
            collection_name=PRODUCT_COLLECTION,
            dimension=embedding_dim,
            metric_type="COSINE",
            consistency_level="Strong",
        )
        data = [
            {"id": i, "vector": embeddings[i], "text": texts[i]}
            for i in range(len(products))
        ]
        milvus_client.insert(collection_name=PRODUCT_COLLECTION, data=data)
        milvus_client.flush(collection_name=PRODUCT_COLLECTION)
        logger.info(f"[Product DB] 新建 Collection，写入 {len(products)} 条商品数据")
        return

    # Collection 已存在 → 增量写入：逐条检查相似度
    existing_count = milvus_client.query(
        collection_name=PRODUCT_COLLECTION,
        filter="", output_fields=["id"], limit=10000,
    )
    next_id = max((r["id"] for r in existing_count), default=-1) + 1

    new_data = []
    skipped = 0
    for idx, (text, emb) in enumerate(zip(texts, embeddings)):
        search_res = milvus_client.search(
            collection_name=PRODUCT_COLLECTION,
            data=[emb], limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text"],
        )
        if search_res and search_res[0]:
            top_score = search_res[0][0]["distance"]
            if top_score >= UPSERT_SIMILARITY_THRESHOLD:
                logger.info(f"[Product DB] 跳过已存在商品 '{products[idx]['name']}'")
                skipped += 1
                continue
        new_data.append({"id": next_id, "vector": emb, "text": text})
        next_id += 1

    if new_data:
        milvus_client.insert(collection_name=PRODUCT_COLLECTION, data=new_data)
        milvus_client.flush(collection_name=PRODUCT_COLLECTION)
        logger.info(f"[Product DB] 增量写入 {len(new_data)} 条，跳过 {skipped} 条")
    else:
        logger.info(f"[Product DB] 所有 {skipped} 条商品均已存在，无需写入")
```

### 3.2 示例商品数据

```python
products = [
    {
        "name": "深海蓝藻保湿面膜",
        "ingredients": "深海蓝藻提取物，富含多糖和氨基酸",
        "effects": "深层补水、修护肌肤屏障、舒缓敏感泛红",
        "target_audience": "所有肤质，尤其适合干燥、敏感肌",
        "specs": "25ml*5片"
    },
    {
        "name": "美白精华",
        "ingredients": "烟酰胺和VC衍生物",
        "effects": "提亮肤色、淡化痘印、改善暗沉",
        "target_audience": "需要均匀肤色的人群",
        "specs": "30ml"
    }
]

init_product_database(products)
```

**Milvus Collection Schema：**

| 字段 | 类型 | 说明 |
|-----|------|------|
| `id` | int64 (主键) | 自增 ID |
| `vector` | float_vector (512维) | 商品文本的 Embedding 向量 |
| `text` | varchar (动态字段) | 商品完整描述文本 |

---

## 4. 真实工具函数实现

> 三个核心工具函数，每个都包含完整的错误处理和降级策略。

### 4.1 🔍 `real_search_web` — 网页搜索工具

**流程：** Tavily 搜索 → LLM 总结为 JSON → 存入 Milvus → 返回格式化结果

```python
from tavily import TavilyClient

SEARCH_COLLECTION = "search_results_collection"


def _summarize_search_results_to_json(query: str, results: list[dict]) -> dict | None:
    """调用 LLM 将搜索结果总结为结构化 JSON。

    返回格式:
        {"query": str, "summary": str, "key_points": [str, ...],
         "sources": [{"title": str, "url": str}, ...]}
    失败时返回 None。
    """
    raw_text = "\n".join(
        f"- {r.get('title', '')}：{r.get('content', '')}" for r in results
    )
    prompt = (
        "你是一个信息提取专家。请将以下搜索结果总结为一个 JSON 对象，格式如下：\n"
        '{"query": "原始搜索词", "summary": "一段 100 字以内的综合摘要", '
        '"key_points": ["要点1", "要点2", ...], '
        '"sources": [{"title": "标题", "url": "链接"}, ...]}\n'
        "只返回 JSON，不要包含任何其他文字。\n\n"
        f"搜索词：{query}\n搜索结果：\n{raw_text}"
    )
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(content)
        logger.info(f"[Web Search] LLM 总结成功，key_points 数量: {len(data.get('key_points', []))}")
        return data
    except Exception as e:
        logger.error(f"[Web Search] LLM 总结失败: {e}")
        return None


def _store_search_summary_to_milvus(summary_json: dict) -> None:
    """将 LLM 总结后的搜索结果增量写入 Milvus 向量数据库。"""
    try:
        text = json.dumps(summary_json, ensure_ascii=False)
        emb = embedding_model.encode_documents([text])
        embedding_dim = len(emb[0])
        milvus_client = MilvusClient(uri=MILVUS_URI)

        # 检测 metric_type 不匹配则删除重建
        if milvus_client.has_collection(SEARCH_COLLECTION):
            indexes = milvus_client.list_indexes(collection_name=SEARCH_COLLECTION)
            if indexes:
                idx_info = milvus_client.describe_index(
                    collection_name=SEARCH_COLLECTION, index_name=indexes[0]
                )
                if idx_info.get('metric_type', '') not in ('', 'COSINE'):
                    milvus_client.drop_collection(SEARCH_COLLECTION)

        if not milvus_client.has_collection(SEARCH_COLLECTION):
            milvus_client.create_collection(
                collection_name=SEARCH_COLLECTION,
                dimension=embedding_dim,
                metric_type="COSINE",
                consistency_level="Strong",
            )

        # 去重检查
        search_res = milvus_client.search(
            collection_name=SEARCH_COLLECTION,
            data=emb, limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text"],
        )
        if search_res and search_res[0]:
            top_score = search_res[0][0]["distance"]
            if top_score >= UPSERT_SIMILARITY_THRESHOLD:
                matched_text = search_res[0][0].get("entity", {}).get("text", "未知")
                logger.info(f"[Web Search] 搜索结果已存在 (相似度 {top_score:.3f})，跳过写入。"
                            f"匹配到的已有记录: {matched_text[:200]}")
                return

        existing = milvus_client.query(
            collection_name=SEARCH_COLLECTION,
            filter="", output_fields=["id"], limit=10000
        )
        next_id = max((r["id"] for r in existing), default=-1) + 1
        milvus_client.insert(
            collection_name=SEARCH_COLLECTION,
            data=[{"id": next_id, "vector": emb[0], "text": text}],
        )
        milvus_client.flush(collection_name=SEARCH_COLLECTION)
        logger.info(f"[Web Search] 搜索结果已写入 Milvus (id={next_id})")
    except Exception as e:
        logger.error(f"[Web Search] 写入 Milvus 失败: {e}")


def real_search_web(query: str) -> str:
    """调用 Tavily API 搜索网页，LLM 总结后存入向量数据库，返回前 5 条结果的摘要。"""
    try:
        logger.info(f"[Web Search] 搜索关键词: {query}")
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query=query, max_results=5)
        results = response.get("results", [])
        if not results:
            logger.warning(f"[Web Search] 未找到结果: {query}")
            return f"未找到关于 '{query}' 的搜索结果。"

        # LLM 总结 + 存入向量数据库
        summary_json = _summarize_search_results_to_json(query, results)
        if summary_json:
            _store_search_summary_to_milvus(summary_json)

        # 格式化返回给 Agent
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            content = r.get("content", "无摘要")
            formatted.append(f"{i}. {title}\n   {content}")
        logger.info(f"[Web Search] 返回 {len(results)} 条结果")
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"[Web Search] 搜索失败: {e}")
        return f"网页搜索失败: {str(e)}"
```

### 4.2 📦 `real_query_product_database` — 商品数据查询工具

**流程：** 产品名称 → Embedding → Milvus 语义搜索 → 返回商品信息

```python
def real_query_product_database(product_name: str) -> str:
    """通过 Milvus 向量数据库语义检索商品信息。"""
    try:
        logger.info(f"[Product DB] 查询商品: {product_name}")
        milvus_client = MilvusClient(uri=MILVUS_URI)
        query_embedding = embedding_model.encode_queries([product_name])

        search_res = milvus_client.search(
            collection_name=PRODUCT_COLLECTION,
            data=query_embedding, limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text"],
        )

        if not search_res or not search_res[0]:
            return f"产品数据库中未找到与 '{product_name}' 匹配的商品。"

        top_result = search_res[0][0]
        if top_result["distance"] < SIMILARITY_THRESHOLD:
            return f"产品数据库中未找到与 '{product_name}' 匹配的商品。"

        logger.info(f"[Product DB] 匹配成功，相似度: {top_result['distance']:.3f}")
        return top_result["entity"]["text"]
    except Exception as e:
        logger.error(f"[Product DB] 查询失败: {e}")
        return f"商品数据查询失败: {str(e)}"
```

### 4.3 😊 `real_generate_emoji` — 表情符号生成工具

**流程：** 上下文 → DeepSeek LLM → 4-6 个 Emoji（失败时返回默认列表）

```python
import json

def real_generate_emoji(context: str) -> list:
    """调用 DeepSeek LLM 生成与上下文匹配的表情符号。"""
    try:
        logger.info(f"[Emoji Gen] 生成表情，上下文: {context[:50]}...")
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个表情符号专家。根据用户提供的文案上下文，"
                        "返回 4 到 6 个最匹配的 emoji 表情符号。"
                        "只返回一个 JSON 数组，不要包含任何其他文字。"
                        '示例: ["✨", "💖", "🌊", "💧"]'
                    ),
                },
                {"role": "user", "content": context},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip()
        emojis = json.loads(content)
        if isinstance(emojis, list) and 4 <= len(emojis) <= 6:
            logger.info(f"[Emoji Gen] 生成成功: {emojis}")
            return emojis
        logger.warning("[Emoji Gen] LLM 返回格式不符，使用默认表情")
        return DEFAULT_EMOJIS
    except Exception as e:
        logger.error(f"[Emoji Gen] 生成失败: {e}")
        return DEFAULT_EMOJIS
```

### 错误处理汇总

| 工具 | 错误场景 | 处理方式 |
|-----|---------|---------|
| `real_search_web` | API 超时/网络错误 | 返回 `"网页搜索失败: {错误描述}"` |
| `real_search_web` | 无搜索结果 | 返回 `"未找到关于 '{query}' 的搜索结果。"` |
| `real_search_web` | LLM 总结失败 | 跳过存储，正常返回搜索结果 |
| `real_query_product_database` | Milvus 连接失败 | 返回 `"商品数据查询失败: {错误描述}"` |
| `real_query_product_database` | 相似度低于阈值 | 返回 `"未找到匹配的商品"` |
| `real_generate_emoji` | LLM 调用失败 | 返回默认列表 `["✨", "🔥", "💖", "💯", "🎉"]` |
| `real_generate_emoji` | 返回格式不符 | 返回默认列表 |

---

## 5. 工具调度与 Agent 定义

> 定义 System Prompt、工具描述和工具路由字典。

### 5.1 System Prompt

```python
SYSTEM_PROMPT = """
你是一个资深的小红书爆款文案专家，擅长结合最新潮流和产品卖点，
创作引人入胜、高互动、高转化的笔记文案。

你的任务是根据用户提供的产品和需求，生成包含标题、正文、相关标签
和表情符号的完整小红书笔记。

请始终采用'Thought-Action-Observation'模式进行推理和行动。
文案风格需活泼、真诚、富有感染力。当完成任务后，请以JSON格式
直接输出最终文案。
"""
```

### 5.2 工具定义与路由

```python
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网上的实时信息...",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_product_database",
            "description": "查询内部产品数据库...",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "产品名称"}
                },
                "required": ["product_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_emoji",
            "description": "生成适合小红书风格的表情符号...",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "文案关键内容"}
                },
                "required": ["context"]
            }
        }
    }
]

# 工具路由字典：名称 → 真实函数
available_tools = {
    "search_web": real_search_web,
    "query_product_database": real_query_product_database,
    "generate_emoji": real_generate_emoji,
}
```

---

## 6. Agent 核心引擎

> ReAct（Thought → Action → Observation）循环，最多 10 轮迭代，包含统一异常捕获。

```python
import re

def generate_rednote(product_name: str, tone_style: str = "活泼甜美",
                     max_iterations: int = 10) -> str:
    """使用 DeepSeek Agent 生成小红书爆款文案。"""
    logger.info(f"🚀 启动文案生成，产品：{product_name}，风格：{tone_style}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请为产品「{product_name}」生成一篇小红书爆款文案..."}
    ]

    iteration_count = 0
    while iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"-- Iteration {iteration_count} --")

        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                tools=TOOLS_DEFINITION,
                tool_choice="auto"
            )
            response_message = response.choices[0].message

            # ① 工具调用分支
            if response_message.tool_calls:
                messages.append(response_message)
                tool_outputs = []
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments or "{}")

                    if function_name in available_tools:
                        try:
                            tool_result = available_tools[function_name](**function_args)
                        except Exception as e:
                            tool_result = f"工具 '{function_name}' 调用异常: {str(e)}"
                    else:
                        tool_result = f"错误：未知的工具 '{function_name}'"

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": str(tool_result)
                    })
                messages.extend(tool_outputs)

            # ② 文本响应分支 → 尝试提取 JSON
            elif response_message.content:
                json_match = re.search(
                    r"```json\s*(\{.*\})\s*```",
                    response_message.content, re.DOTALL
                )
                if json_match:
                    try:
                        final = json.loads(json_match.group(1))
                        return json.dumps(final, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        messages.append(response_message)
                else:
                    try:
                        final = json.loads(response_message.content)
                        return json.dumps(final, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        messages.append(response_message)
            else:
                break

        except Exception as e:
            logger.error(f"调用 DeepSeek API 时发生错误: {e}")
            break

    return "未能成功生成文案。"
```

---

## 7. 输出格式化

> 将 JSON 文案转换为可读的 Markdown 格式。

```python
def format_rednote_for_markdown(json_string: str) -> str:
    """将 JSON 格式的小红书文案转换为 Markdown 格式。"""
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return f"错误：无法解析 JSON 字符串 - {e}\n原始字符串：\n{json_string}"

    title = data.get("title", "无标题")
    body = data.get("body", "")
    hashtags = data.get("hashtags", [])

    markdown_output = f"## {title}\n\n"
    markdown_output += f"{body}\n\n"
    if hashtags:
        markdown_output += " ".join(hashtags) + "\n"
    return markdown_output.strip()
```

---

## 8. 端到端测试

```python
result = generate_rednote("深海蓝藻保湿面膜", "活泼甜美")

print("--- 格式化后的小红书文案 (Markdown) ---")
print(format_rednote_for_markdown(result))
```

---

## 9. 导出文案为 Markdown 文件

> 将生成的文案自动导出到 `md/` 目录，文件名包含标题和时间戳。

```python
from datetime import datetime
from pathlib import Path

def export_rednote_to_md(json_string: str, output_dir: str = "md") -> str:
    """将格式化后的小红书文案导出为 Markdown 文件。"""
    md_content = format_rednote_for_markdown(json_string)

    try:
        data = json.loads(json_string)
        title = data.get("title", "未命名文案")
    except Exception:
        title = "未命名文案"

    safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_title}_{timestamp}.md"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / filename

    file_path.write_text(md_content, encoding="utf-8")
    logger.info(f"[Export] 文案已导出: {file_path}")
    print(f"✅ 文案已导出到: {file_path}")
    return str(file_path)

# 导出
export_rednote_to_md(result)
```

---

## 📦 依赖清单

| 包名 | 用途 |
|-----|------|
| `openai` | DeepSeek API 客户端（OpenAI 兼容） |
| `pymilvus[model]` | Milvus 向量数据库客户端 + Embedding 模型 |
| `torch` | PyTorch，Embedding 模型依赖 |
| `tavily-python` | Tavily 搜索 API SDK |
| `python-dotenv` | 环境变量加载 |

## 🔑 环境变量

| 变量名 | 必填 | 说明 |
|--------|-----|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API 密钥 |
| `TAVILY_API_KEY` | ✅ | Tavily 搜索 API 密钥 |

## 🐳 基础设施

```bash
# 启动 Milvus（需要 Docker）
docker-compose up -d

# 安装 Python 依赖
pip install -r requirements.txt
```
