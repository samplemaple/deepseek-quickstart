"""LLMClient 单元测试

测试 DeepSeek API 客户端封装的核心行为：
- 环境变量配置读取
- 指数退避重试机制
- JSON 响应解析和验证
- chat() 和 chat_json() 方法
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIError, APIConnectionError, RateLimitError

from geo_tool.modules.llm_client import LLMClient


# ===== 初始化和配置测试 =====


class TestLLMClientInit:
    """测试 LLMClient 初始化配置"""

    def test_default_config_from_env(self):
        """应从环境变量读取配置"""
        with patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "test-key-123",
                "DEEPSEEK_BASE_URL": "https://custom.api.com",
                "DEEPSEEK_MODEL": "deepseek-v3",
            },
        ):
            # 重新加载以获取新的环境变量
            client = LLMClient()
            assert client.api_key == "test-key-123"
            assert client.base_url == "https://custom.api.com"
            assert client.model == "deepseek-v3"

    def test_explicit_params_override_env(self):
        """显式传入的参数应覆盖环境变量"""
        client = LLMClient(
            api_key="explicit-key",
            base_url="https://explicit.api.com",
            model="explicit-model",
        )
        assert client.api_key == "explicit-key"
        assert client.base_url == "https://explicit.api.com"
        assert client.model == "explicit-model"

    def test_default_values_when_no_env(self):
        """无环境变量时应使用默认值"""
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            client = LLMClient(api_key="key")
            assert client.base_url == "https://api.deepseek.com"
            assert client.model == "deepseek-v4-flash"


    def test_retry_delays_config(self):
        """重试延迟应为 [1, 2, 4] 秒"""
        assert LLMClient._RETRY_DELAYS == [1, 2, 4]

    def test_max_retries_is_three(self):
        """最大重试次数应为 3"""
        assert LLMClient._MAX_RETRIES == 3


# ===== JSON 解析测试 =====


class TestParseJson:
    """测试 _parse_json 静态方法"""

    def test_valid_json(self):
        """有效 JSON 字符串应正确解析"""
        result = LLMClient._parse_json('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_in_markdown_code_block(self):
        """markdown 代码块中的 JSON 应正确提取"""
        text = '```json\n{"key": "value"}\n```'
        result = LLMClient._parse_json(text)
        assert result == {"key": "value"}

    def test_json_in_plain_code_block(self):
        """无语言标记的代码块中的 JSON 应正确提取"""
        text = '```\n{"key": "value"}\n```'
        result = LLMClient._parse_json(text)
        assert result == {"key": "value"}

    def test_json_with_surrounding_whitespace(self):
        """带前后空白的 JSON 应正确解析"""
        result = LLMClient._parse_json('  \n  {"key": "value"}  \n  ')
        assert result == {"key": "value"}

    def test_invalid_json_returns_none(self):
        """无效 JSON 应返回 None"""
        result = LLMClient._parse_json("这不是 JSON")
        assert result is None

    def test_empty_string_returns_none(self):
        """空字符串应返回 None"""
        result = LLMClient._parse_json("")
        assert result is None

    def test_none_input_returns_none(self):
        """None 输入应返回 None"""
        result = LLMClient._parse_json(None)
        assert result is None

    def test_json_array_returns_none(self):
        """JSON 数组（非字典）应返回 None"""
        result = LLMClient._parse_json('[1, 2, 3]')
        assert result is None

    def test_nested_json(self):
        """嵌套 JSON 应正确解析"""
        text = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = LLMClient._parse_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}, "flag": True}

    def test_chinese_content_json(self):
        """包含中文的 JSON 应正确解析"""
        text = '{"标题": "测试内容", "关键词": ["词1", "词2"]}'
        result = LLMClient._parse_json(text)
        assert result["标题"] == "测试内容"


# ===== chat() 方法测试 =====


class TestChat:
    """测试 chat() 异步方法"""

    def _make_client(self) -> LLMClient:
        """创建测试用客户端"""
        return LLMClient(api_key="test-key")

    def _mock_completion(self, content: str):
        """构造模拟的 API 响应"""
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        return mock_response

    @pytest.mark.asyncio
    async def test_successful_chat(self):
        """成功调用应返回文本内容"""
        client = self._make_client()
        mock_resp = self._mock_completion("你好，这是回复")

        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.chat([{"role": "user", "content": "你好"}])
        assert result == "你好，这是回复"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_string(self):
        """模型返回 None content 时应返回空字符串"""
        client = self._make_client()
        mock_resp = self._mock_completion(None)

        client._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await client.chat([{"role": "user", "content": "测试"}])
        assert result == ""

    @pytest.mark.asyncio
    async def test_retry_on_api_error(self):
        """API 错误时应重试并最终成功"""
        client = self._make_client()
        mock_resp = self._mock_completion("重试成功")

        # 第一次失败，第二次成功
        mock_error = APIConnectionError(request=MagicMock())
        client._client.chat.completions.create = AsyncMock(
            side_effect=[mock_error, mock_resp]
        )

        # 使用 patch 跳过实际的 sleep
        with patch("geo_tool.modules.llm_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.chat([{"role": "user", "content": "测试"}])
            assert result == "重试成功"

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_runtime_error(self):
        """所有重试均失败时应抛出 RuntimeError"""
        client = self._make_client()

        mock_error = APIConnectionError(request=MagicMock())
        client._client.chat.completions.create = AsyncMock(
            side_effect=[mock_error, mock_error, mock_error]
        )

        with patch("geo_tool.modules.llm_client.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="已重试 3 次"):
                await client.chat([{"role": "user", "content": "测试"}])

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """限流错误时应重试"""
        client = self._make_client()
        mock_resp = self._mock_completion("限流后成功")

        mock_error = RateLimitError(
            message="rate limit",
            response=MagicMock(status_code=429, headers={}),
            body=None,
        )
        client._client.chat.completions.create = AsyncMock(
            side_effect=[mock_error, mock_resp]
        )

        with patch("geo_tool.modules.llm_client.asyncio.sleep", new_callable=AsyncMock):
            result = await client.chat([{"role": "user", "content": "测试"}])
            assert result == "限流后成功"

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """重试应使用指数退避延迟：1秒、2秒、4秒"""
        client = self._make_client()

        mock_error = APIConnectionError(request=MagicMock())
        client._client.chat.completions.create = AsyncMock(
            side_effect=[mock_error, mock_error, mock_error]
        )

        sleep_calls = []

        async def mock_sleep(seconds):
            sleep_calls.append(seconds)

        with patch("geo_tool.modules.llm_client.asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(RuntimeError):
                await client.chat([{"role": "user", "content": "测试"}])

        # 3次尝试，前2次失败后各 sleep 一次（第3次失败后不再 sleep）
        assert sleep_calls == [1, 2]


# ===== chat_json() 方法测试 =====


class TestChatJson:
    """测试 chat_json() 异步方法"""

    def _make_client(self) -> LLMClient:
        """创建测试用客户端"""
        return LLMClient(api_key="test-key")

    @pytest.mark.asyncio
    async def test_successful_json_parse(self):
        """成功解析 JSON 响应"""
        client = self._make_client()
        expected = {"keywords": ["测试", "关键词"], "score": 85}

        with patch.object(
            client, "chat", new_callable=AsyncMock, return_value=json.dumps(expected)
        ):
            result = await client.chat_json([{"role": "user", "content": "分析"}])
            assert result == expected

    @pytest.mark.asyncio
    async def test_json_parse_retry_on_failure(self):
        """第一次 JSON 解析失败时应重试一次"""
        client = self._make_client()
        expected = {"result": "ok"}

        # 第一次返回非 JSON，第二次返回有效 JSON
        call_count = 0

        async def mock_chat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "这不是 JSON 格式的回复"
            return json.dumps(expected)

        with patch.object(client, "chat", side_effect=mock_chat):
            result = await client.chat_json([{"role": "user", "content": "分析"}])
            assert result == expected
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_json_parse_both_fail_raises_value_error(self):
        """两次 JSON 解析均失败时应抛出 ValueError"""
        client = self._make_client()

        with patch.object(
            client,
            "chat",
            new_callable=AsyncMock,
            return_value="始终不是 JSON",
        ):
            with pytest.raises(ValueError, match="无法解析"):
                await client.chat_json([{"role": "user", "content": "分析"}])

    @pytest.mark.asyncio
    async def test_json_in_markdown_block(self):
        """markdown 代码块中的 JSON 应正确解析"""
        client = self._make_client()
        expected = {"data": "test"}

        with patch.object(
            client,
            "chat",
            new_callable=AsyncMock,
            return_value='```json\n{"data": "test"}\n```',
        ):
            result = await client.chat_json([{"role": "user", "content": "分析"}])
            assert result == expected

    @pytest.mark.asyncio
    async def test_retry_message_includes_correction_hint(self):
        """重试时应在消息中包含纠正提示"""
        client = self._make_client()
        captured_messages = []

        call_count = 0

        async def mock_chat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_messages.append(messages)
            if call_count == 1:
                return "不是 JSON"
            return '{"ok": true}'

        with patch.object(client, "chat", side_effect=mock_chat):
            await client.chat_json([{"role": "user", "content": "分析"}])

        # 第二次调用的消息应包含纠正提示
        retry_msgs = captured_messages[1]
        assert any("JSON" in msg["content"] for msg in retry_msgs if msg["role"] == "user")
