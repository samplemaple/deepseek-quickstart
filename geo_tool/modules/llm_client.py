"""DeepSeek API 客户端封装

基于 OpenAI SDK 封装 DeepSeek API 调用，提供：
- 指数退避重试机制（最多3次）
- JSON 响应解析和验证
- 异步 chat() 和 chat_json() 方法
"""

import asyncio
import json
import logging
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

# 加载 .env 文件中的环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """DeepSeek API 客户端

    通过 OpenAI SDK 兼容接口调用 DeepSeek API，
    支持指数退避重试和 JSON 响应解析。
    """

    # 指数退避重试的基础等待时间（秒）
    _RETRY_DELAYS = [1, 2, 4]
    # 最大重试次数
    _MAX_RETRIES = 3

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """初始化 LLM 客户端

        参数优先级：显式传入 > 环境变量 > 默认值

        Args:
            api_key: DeepSeek API 密钥
            base_url: DeepSeek API 基础 URL
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

        if not self.api_key:
            logger.warning("未设置 DEEPSEEK_API_KEY，LLM 调用将会失败")

        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """发送消息并返回文本响应

        使用指数退避重试机制，最多重试3次。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 生成温度，控制随机性
            max_tokens: 最大生成 token 数

        Returns:
            模型返回的文本内容

        Raises:
            RuntimeError: 所有重试均失败时抛出
        """
        last_error: Exception | None = None

        for attempt in range(self._MAX_RETRIES):
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                return content or ""

            except (APIError, APIConnectionError, RateLimitError) as e:
                last_error = e
                if attempt < self._MAX_RETRIES - 1:
                    delay = self._RETRY_DELAYS[attempt]
                    logger.warning(
                        "DeepSeek API 调用失败（第 %d 次），%d 秒后重试: %s",
                        attempt + 1,
                        delay,
                        str(e),
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "DeepSeek API 调用失败，已达最大重试次数: %s", str(e)
                    )

        raise RuntimeError(
            f"DeepSeek API 调用失败，已重试 {self._MAX_RETRIES} 次: {last_error}"
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> dict:
        """发送消息并解析返回的 JSON

        先调用 chat() 获取文本响应，然后尝试解析为 JSON。
        JSON 解析失败时会重试一次。

        Args:
            messages: 消息列表
            temperature: 生成温度（JSON 模式下建议较低值）
            max_tokens: 最大生成 token 数

        Returns:
            解析后的 JSON 字典

        Raises:
            ValueError: JSON 解析两次均失败时抛出
            RuntimeError: API 调用失败时抛出
        """
        # 第一次尝试
        text = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        result = self._parse_json(text)
        if result is not None:
            return result

        logger.warning("JSON 解析失败，重试一次")

        # JSON 解析失败，重试一次
        retry_messages = messages.copy()
        retry_messages.append({"role": "assistant", "content": text})
        retry_messages.append({
            "role": "user",
            "content": "你的上一次回复不是有效的 JSON 格式。请只返回纯 JSON，不要包含 markdown 代码块或其他文字。",
        })

        text = await self.chat(
            retry_messages, temperature=temperature, max_tokens=max_tokens
        )
        result = self._parse_json(text)
        if result is not None:
            return result

        raise ValueError(f"无法解析 LLM 返回的 JSON，原始响应: {text[:500]}")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """尝试从文本中解析 JSON

        支持处理 markdown 代码块包裹的 JSON。

        Args:
            text: 待解析的文本

        Returns:
            解析成功返回字典，失败返回 None
        """
        if not text or not text.strip():
            return None

        cleaned = text.strip()

        # 尝试提取 markdown 代码块中的 JSON
        code_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
        )
        if code_block_match:
            cleaned = code_block_match.group(1).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            return None
        except (json.JSONDecodeError, TypeError):
            return None
