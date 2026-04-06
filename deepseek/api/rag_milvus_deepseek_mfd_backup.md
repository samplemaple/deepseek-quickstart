# 使用 Milvus 和 DeepSeek 构建 RAG（民法典版）

基于民法典物权编文档，使用 Milvus 向量数据库 + DeepSeek LLM 构建 RAG 管道。

流程：加载法律条文 → 生成嵌入向量 → 存入 Milvus → 检索 → LLM 生成回答

## 1. 加载环境变量


```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
print(f"[✓] API Key 已加载 (前8位: {api_key[:8]}...)")
```

## 2. 加载并拆分民法典条文

按 `**第xxx条**` 格式拆分，每条法律条文作为一个独立文本块。


```python
import re

with open("./mfd.md", "r", encoding="utf-8") as f:
    file_text = f.read()

# 用正则在每个 **第 前面切割，保留分隔符
raw_parts = re.split(r'(?=\*\*第)', file_text)

# 过滤掉空白和非条文内容，只保留以 **第 开头的条文
text_lines = [part.strip() for part in raw_parts if part.strip().startswith('**第')]

print(f'[✓] 共拆分出 {len(text_lines)} 条法律条文')
print(f'    示例: {text_lines[0][:80]}...')
```

## 3. 初始化 LLM 和 Embedding 模型

- LLM: DeepSeek (deepseek-chat)
- Embedding: BAAI/bge-small-zh-v1.5 (中文优化, 512维)


```python
from openai import OpenAI
from pymilvus import model

# DeepSeek LLM 客户端
deepseek_client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
)

# 中文 Embedding 模型
embedding_model = model.dense.SentenceTransformerEmbeddingFunction(
    model_name='BAAI/bge-small-zh-v1.5',
    device='cpu'
)

print("[✓] LLM 和 Embedding 模型已初始化")
```

## 4. 将数据加载到 Milvus

创建 Collection 并插入所有条文的嵌入向量。

需要先通过 Docker 启动 Milvus 服务（milvus-lite 不支持 Windows）。


```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="http://localhost:19530")
collection_name = "mfd_collection"

# 重建 collection
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=512,       # bge-small-zh-v1.5 输出 512 维
    metric_type="IP",    # 内积距离
    consistency_level="Strong",
)

print(f"[✓] Collection '{collection_name}' 已创建")
```


```python
from tqdm import tqdm

# 生成嵌入向量并插入
doc_embeddings = embedding_model.encode_documents(text_lines)

data = []
for i, line in enumerate(tqdm(text_lines, desc="插入数据")):
    data.append({"id": i, "vector": doc_embeddings[i], "text": line})

result = milvus_client.insert(collection_name=collection_name, data=data)
print(f"[✓] 已插入 {result['insert_count']} 条数据")
```


```python
# 验证数据
results = milvus_client.query(
    collection_name=collection_name,
    filter="id >= 0",
    output_fields=["id", "text"],
    limit=1000
)

print(f"共 {len(results)} 条数据\n")
for r in results[:5]:
    print(f"[id={r['id']}] {r['text'][:80]}...\n")
```

## 5. 构建 RAG 检索

支持两种检索模式：
1. 精确匹配：从问题中提取条文编号（支持阿拉伯数字和中文数字互转），直接文本匹配
2. 语义搜索：无条文编号时，回退到 Milvus 向量搜索


```python
import re
import json


def num_to_chinese(num):
    """阿拉伯数字转中文数字，如 205 -> 二百零五"""
    digits = '零一二三四五六七八九'
    units = ['', '十', '百', '千', '万']
    n = int(num)
    if n == 0:
        return '零'
    parts = []
    s = str(n)
    length = len(s)
    for i, ch in enumerate(s):
        d = int(ch)
        pos = length - 1 - i
        if d == 0:
            if parts and parts[-1] != '零':
                parts.append('零')
        else:
            parts.append(digits[d] + units[pos])
    return ''.join(parts).rstrip('零')


def extract_article_keywords(question):
    """从问题中提取条文编号，支持中文数字和阿拉伯数字，返回关键词列表"""
    keywords = []
    # 匹配中文数字: 第二百零五条
    cn_match = re.search(r'第[零一二三四五六七八九十百千万]+条', question)
    if cn_match:
        keywords.append(cn_match.group())
    # 匹配阿拉伯数字: 第205条 -> 转为中文
    ar_match = re.search(r'第(\d+)条', question)
    if ar_match:
        cn_num = num_to_chinese(ar_match.group(1))
        keywords.append(f'第{cn_num}条')
    return list(set(keywords))


def search_articles(question, text_lines, milvus_client, collection_name, embedding_model):
    """混合检索：优先精确匹配条文编号，否则回退到向量搜索"""
    keywords = extract_article_keywords(question)
    print(f'提取到的条文关键词: {keywords}')

    # 精确匹配
    exact_matches = []
    if keywords:
        for kw in keywords:
            exact_matches += [line for line in text_lines if kw in line]
        exact_matches = list(set(exact_matches))

    if exact_matches:
        print(f'[✓] 精确匹配到 {len(exact_matches)} 条结果')
        return [(m, 1.0) for m in exact_matches[:3]]

    # 回退到向量搜索
    print('[→] 未找到精确匹配，使用向量搜索')
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=embedding_model.encode_queries([question]),
        limit=3,
        search_params={'metric_type': 'IP', 'params': {}},
        output_fields=['text'],
    )
    return [(res['entity']['text'], res['distance']) for res in search_res[0]]
```


```python
question = "中华人民共和国民法典第205条是什么?"

retrieved_lines_with_distances = search_articles(
    question, text_lines, milvus_client, collection_name, embedding_model
)

# 输出搜索结果
print(json.dumps(retrieved_lines_with_distances, indent=4, ensure_ascii=False))

# 构建上下文
context = "\n".join([item[0] for item in retrieved_lines_with_distances])
```

## 6. 使用 DeepSeek LLM 生成回答

将检索到的法律条文作为上下文，提供给 LLM 生成最终回答。


```python
SYSTEM_PROMPT = """
你是一个专业的法律 AI 助手。你能够从提供的法律条文中准确找到问题的答案。
"""

USER_PROMPT = f"""
请使用以下用 <context> 标签括起来的法律条文来回答用 <question> 标签括起来的问题。
<context>
{context}
</context>
<question>
{question}
</question>
"""

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
)

print("回答:")
print("-" * 60)
print(response.choices[0].message.content)
print("-" * 60)
print("\n[✓] RAG 管道执行完成")
```
