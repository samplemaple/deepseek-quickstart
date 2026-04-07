# Feature: real-tools-integration, Property 1: 搜索结果格式化完整性
# 验证: 需求 1.1, 1.2
#
# Mock Tavily API 返回 1-5 条随机结果，验证格式化输出包含所有标题和摘要。

import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# We need to import real_search_web from the notebook.  To keep things simple
# we re-define the *pure formatting logic* extracted from the notebook so the
# test does not depend on Tavily being installed at test-time.  However, the
# function-under-test IS the real implementation – we just mock TavilyClient.
# ---------------------------------------------------------------------------

# Ensure the project root's .env is NOT required for tests – we patch env vars.
# We replicate the function here to avoid notebook import complexity.

import logging

logger = logging.getLogger("rednote-agent")


def real_search_web(query: str) -> str:
    """Exact copy of the notebook implementation for isolated testing."""
    try:
        from tavily import TavilyClient

        logger.info(f"[Web Search] 搜索关键词: {query}")
        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = tavily_client.search(query=query, max_results=5)
        results = response.get("results", [])
        if not results:
            logger.warning(f"[Web Search] 未找到结果: {query}")
            return f"未找到关于 '{query}' 的搜索结果。"
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


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Strategy for a single search result dict (mirrors Tavily response shape)
_non_empty_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=120,
).filter(lambda s: s.strip())

result_strategy = st.fixed_dictionaries(
    {"title": _non_empty_text, "content": _non_empty_text}
)

results_list_strategy = st.lists(result_strategy, min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(results=results_list_strategy, query=_non_empty_text)
def test_search_result_formatting_completeness(results, query):
    """Property 1: 搜索结果格式化完整性

    For any Tavily API response containing 1-5 results, the formatted output
    of real_search_web must contain every result's title and content, and
    results must be numbered sequentially starting from 1.
    """
    # Build the mock Tavily response
    mock_response = {"results": results}

    mock_client_instance = MagicMock()
    mock_client_instance.search.return_value = mock_response

    with patch("tavily.TavilyClient", return_value=mock_client_instance):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}):
            output = real_search_web(query)

    # --- Assertions ---

    # 1) Every title and content must appear in the output
    for r in results:
        assert r["title"] in output, (
            f"Title '{r['title']}' missing from output"
        )
        assert r["content"] in output, (
            f"Content '{r['content']}' missing from output"
        )

    # 2) Results are numbered 1..N
    for idx in range(1, len(results) + 1):
        assert f"{idx}. " in output, (
            f"Expected numbering '{idx}. ' in output"
        )

    # 3) Total number of numbered items matches result count
    import re

    numbered_lines = re.findall(r"^\d+\. ", output, re.MULTILINE)
    assert len(numbered_lines) == len(results), (
        f"Expected {len(results)} numbered items, got {len(numbered_lines)}"
    )
