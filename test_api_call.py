"""测试 API 调用脚本 — 逐步调用各接口"""
import asyncio
import json
import httpx

BASE_URL = "http://127.0.0.1:8000/api/v1"

# 模拟提取的内容（因为小红书反爬，直接构造测试数据）
MOCK_EXTRACTED = {
    "url": "https://www.xiaohongshu.com/explore/69e2194a0000000023015c81",
    "text": "2025年家用投影仪选购攻略，从入门到高端全覆盖。经过3个月实测10款主流投影仪，从亮度、分辨率、色彩、噪音、系统流畅度五个维度进行深度评测。极米H6 Pro以2200ANSI流明和4K分辨率拿下综合第一，当贝F6紧随其后，性价比之王是小明Q3 Pro。预算3000以下选小明，5000档选当贝，追求极致选极米。所有数据均为暗室环境实测，拒绝云评测。",
    "image_urls": ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
    "metadata": {
        "title": "2025年家用投影仪选购攻略｜10款实测对比",
        "author": "数码测评达人",
        "publish_time": "2025-04-20",
        "likes": 5200,
        "collects": 3800,
        "comments": 420
    },
    "extracted_at": "2025-04-25T10:00:00"
}


async def call_api(client, step_name, endpoint, payload):
    """通用 API 调用"""
    print(f"\n{'=' * 60}")
    print(f"【{step_name}】POST /api/v1/{endpoint}")
    print(f"{'=' * 60}")
    try:
        r = await client.post(f"{BASE_URL}/{endpoint}", json=payload)
        print(f"状态码: {r.status_code}")
        if r.text:
            result = r.json()
            # 截断过长的输出
            output = json.dumps(result, ensure_ascii=False, indent=2)
            if len(output) > 3000:
                print(output[:3000])
                print(f"\n... (输出已截断，总长度 {len(output)} 字符)")
            else:
                print(output)
            return result
        else:
            print("空响应体")
            return None
    except Exception as e:
        print(f"请求异常: {e}")
        return None


async def main():
    async with httpx.AsyncClient(timeout=120) as client:

        # ===== 步骤1：关键词分析 =====
        kw_result = await call_api(
            client, "关键词分析", "analyze-keywords",
            {"content": MOCK_EXTRACTED}
        )
        if not kw_result or not kw_result.get("success"):
            print("\n关键词分析失败，流程终止")
            return

        keywords = kw_result["data"]

        # ===== 步骤2：内容二创（榜单式模板） =====
        recreate_result = await call_api(
            client, "内容二创", "recreate",
            {
                "content": MOCK_EXTRACTED,
                "keywords": keywords,
                "template_type": "ranking",
                "business_type": "online"
            }
        )
        if not recreate_result or not recreate_result.get("success"):
            print("\n内容二创失败，流程终止")
            return

        recreated = recreate_result["data"]

        # ===== 步骤3：质量评分 =====
        score_result = await call_api(
            client, "质量评分", "score",
            {"content": recreated}
        )

        # ===== 步骤4：平台适配（豆包 + DeepSeek） =====
        adapt_result = await call_api(
            client, "平台适配", "adapt",
            {
                "content": recreated,
                "platforms": ["doubao", "deepseek"]
            }
        )

        # ===== 步骤5：排名监测 =====
        if recreated.get("titles"):
            title_text = recreated["titles"][0]["text"]
            monitor_result = await call_api(
                client, "排名监测", "monitor",
                {
                    "keyword": "投影仪推荐",
                    "platform": "doubao",
                    "content_title": title_text
                }
            )

        print(f"\n{'=' * 60}")
        print("全部接口调用完成")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
