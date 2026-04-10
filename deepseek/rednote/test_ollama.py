"""直接测试 Ollama 接口，绕过 notebook 环境排查 503 问题。"""
import requests
import json

url = "http://localhost:11434/v1/chat/completions"

payload = {
    "model": "deepseek-r1:8b",
    "temperature": 0.7,
    "messages": [
        {
            "role": "system",
            "content": (
                "你是一个资深的小红书爆款文案专家，擅长结合最新潮流和产品卖点，"
                "创作引人入胜、高互动、高转化的笔记文案。\n\n"
                "你的任务是根据用户提供的产品信息和参考资料，"
                "生成包含标题、正文、相关标签和表情符号的完整小红书笔记。\n\n"
                '请直接以 JSON 格式输出最终文案，用 ```json ... ``` 包裹，格式如下：\n'
                "```json\n"
                "{\n"
                '  "title": "小红书标题",\n'
                '  "body": "小红书正文",\n'
                '  "hashtags": ["#标签1", "#标签2", "#标签3", "#标签4", "#标签5"],\n'
                '  "emojis": ["✨", "🔥", "💖", "💯", "🎉"]\n'
                "}\n"
                "```\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "请为产品「360宠物喂食器」生成一篇小红书爆款文案。\n"
                "要求：语气严肃科普，包含标题、正文、至少5个标签和5个表情符号。\n\n"
                "【网络搜索参考】\n"
                "1. 霍曼智能喂食器：让你的宠物享受高端生活，免去喂食烦恼！\n"
                "   加宽加长双出粮滑道，适合多只宠物家庭。配备双陶瓷碗，"
                "食盆底高度为75mm，符合宠物用餐习惯。支持冻干、干粮、膨化粮、"
                "风干粮等多种宠粮。360°全密封锁鲜设计。\n"
                "2. 这喂食器不只投喂，是在重构宠物孤独经济的底层逻辑\n"
                "   市面上的宠物喂食器90%只是定时饭盒，而这款真正把陪伴做成了"
                "可编程的系统行为。双镜头+全角度激光移动检测。\n"
                "3. 霍曼智能喂食器：轻松喂养多宠家庭，告别卡粮烦恼！\n"
                "   配备双陶瓷碗，食盆底高度达到75mm，既方便宠物进食，"
                "也减少了脊椎的负担。支持冻干、干粮、膨化粮和风干粮。\n\n"
                "请直接输出 JSON，用 ```json ... ``` 包裹。"
            ),
        },
    ],
}

print("正在请求 Ollama...")
print(f"URL: {url}")
print(f"Model: {payload['model']}")
print("-" * 50)

try:
    resp = requests.post(url, json=payload, timeout=300)
    print(f"HTTP Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print("\n=== LLM 响应 ===")
        print(content)
    else:
        print(f"错误: {resp.text}")
except Exception as e:
    print(f"请求失败: {e}")
