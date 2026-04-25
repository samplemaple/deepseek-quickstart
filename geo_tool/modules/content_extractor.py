"""小红书笔记内容提取模块

负责从小红书笔记链接中提取文字内容、图片URL和元数据。
优先使用 xhs-crawl 库提取，失败时降级为 httpx + beautifulsoup4 解析。
"""

import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from geo_tool.models.content import ExtractedContent, NoteMetadata

logger = logging.getLogger(__name__)

# 尝试导入 xhs-crawl 库
try:
    from xhs_crawl import XHSSpider
    HAS_XHS_CRAWL = True
except ImportError:
    HAS_XHS_CRAWL = False
    logger.warning("xhs-crawl 未安装，将使用备选 HTTP 提取方式")

# 常见浏览器 User-Agent 列表（至少5个）
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    # Chrome on Linux
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# 小红书链接正则模式
# 短链格式: https://xhslink.com/xxxxx 或 http://xhslink.com/xxxxx
SHORT_LINK_PATTERN = re.compile(
    r"^https?://xhslink\.com/[A-Za-z0-9]+/?$"
)
# 长链格式: https://www.xiaohongshu.com/explore/笔记ID 或 /discovery/item/笔记ID
LONG_LINK_PATTERN = re.compile(
    r"^https?://(www\.)?xiaohongshu\.com/(explore|discovery/item)/[a-f0-9]{24}/?(\?.*)?$"
)

# 最大重试次数
MAX_RETRIES = 3
# 请求间隔范围（秒）
MIN_DELAY = 1.0
MAX_DELAY = 3.0


class ContentExtractor:
    """小红书笔记内容提取器

    从小红书笔记链接中提取文字内容、图片URL和元数据。
    支持 xhslink.com 短链和 xiaohongshu.com 长链。
    """

    def __init__(self) -> None:
        """初始化提取器，从环境变量加载小红书 Cookie"""
        self._cookie = os.environ.get("XHS_COOKIE", "")
        if self._cookie:
            logger.info("已加载小红书 Cookie（长度 %d）", len(self._cookie))
        else:
            logger.warning("未配置 XHS_COOKIE，提取可能受反爬限制")

    def _validate_url(self, url: str) -> bool:
        """验证是否为有效的小红书链接

        支持两种格式：
        - 短链: https://xhslink.com/xxxxx
        - 长链: https://www.xiaohongshu.com/explore/笔记ID

        Args:
            url: 待验证的链接字符串

        Returns:
            True 表示链接有效，False 表示无效
        """
        if not url or not isinstance(url, str):
            return False

        url = url.strip()

        # 匹配短链或长链格式
        if SHORT_LINK_PATTERN.match(url):
            return True
        if LONG_LINK_PATTERN.match(url):
            return True

        return False

    def _get_headers(self) -> dict[str, str]:
        """生成随机请求头

        随机选择 User-Agent，模拟正常浏览器访问。
        如果配置了 Cookie，会自动带上。

        Returns:
            包含随机 UA 的请求头字典
        """
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.xiaohongshu.com/",
        }
        if self._cookie:
            headers["Cookie"] = self._cookie
        return headers

    async def _handle_anti_crawl(self, url: str) -> httpx.Response | None:
        """处理反爬机制，执行带重试的 HTTP 请求

        策略：
        - 随机 User-Agent
        - 请求间隔 1-3 秒
        - 最多重试 3 次

        Args:
            url: 请求目标 URL

        Returns:
            成功时返回 httpx.Response，全部重试失败返回 None
        """
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES):
            # 非首次请求时，等待随机间隔
            if attempt > 0:
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                await asyncio.sleep(delay)

            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=httpx.Timeout(15.0),
                ) as client:
                    response = await client.get(url, headers=self._get_headers())

                    # 状态码 200 表示成功
                    if response.status_code == 200:
                        return response

                    # 404 表示笔记不存在，无需重试
                    if response.status_code == 404:
                        return response

                    # 403/429 等反爬状态码，继续重试
                    last_exception = Exception(
                        f"HTTP {response.status_code}"
                    )

            except httpx.TimeoutException as e:
                last_exception = e
            except httpx.HTTPError as e:
                last_exception = e

        # 所有重试均失败
        return None

    def _parse_note(self, html: str, url: str) -> ExtractedContent | None:
        """解析笔记页面 HTML/JSON，提取文字、图片URL和元数据

        小红书笔记页面通常在 <script> 标签中嵌入 JSON 数据（window.__INITIAL_STATE__），
        包含笔记的完整信息。同时也从 HTML 结构中提取作为备选。

        Args:
            html: 页面 HTML 内容
            url: 原始请求 URL

        Returns:
            解析成功返回 ExtractedContent，失败返回 None
        """
        soup = BeautifulSoup(html, "html.parser")

        # 尝试从嵌入的 JSON 数据中提取（优先）
        result = self._parse_from_initial_state(soup, url)
        if result:
            return result

        # 备选：从 HTML 结构中提取
        result = self._parse_from_html(soup, url)
        if result:
            return result

        return None

    def _parse_from_initial_state(
        self, soup: BeautifulSoup, url: str
    ) -> ExtractedContent | None:
        """从 window.__INITIAL_STATE__ JSON 数据中提取笔记信息

        Args:
            soup: BeautifulSoup 解析对象
            url: 原始请求 URL

        Returns:
            解析成功返回 ExtractedContent，失败返回 None
        """
        for script in soup.find_all("script"):
            text = script.string or ""
            if "window.__INITIAL_STATE__" not in text:
                continue

            try:
                # 提取 JSON 字符串
                json_str = text.split("window.__INITIAL_STATE__=", 1)[1]
                # 处理可能的尾部分号或其他字符
                json_str = json_str.strip().rstrip(";")
                # 小红书有时使用 undefined 替代 null
                json_str = json_str.replace("undefined", "null")
                data = json.loads(json_str)

                return self._extract_from_state_data(data, url)
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

        return None

    def _extract_from_state_data(
        self, data: dict[str, Any], url: str
    ) -> ExtractedContent | None:
        """从解析后的 __INITIAL_STATE__ 数据中提取笔记字段

        Args:
            data: 解析后的 JSON 数据
            url: 原始请求 URL

        Returns:
            提取成功返回 ExtractedContent，失败返回 None
        """
        try:
            # 尝试多种可能的数据路径
            note_data = None

            # 路径1: note.noteDetailMap 中的第一个笔记
            note_detail_map = data.get("note", {}).get("noteDetailMap", {})
            if note_detail_map:
                first_key = next(iter(note_detail_map))
                note_data = note_detail_map[first_key].get("note", {})

            # 路径2: note.note
            if not note_data:
                note_data = data.get("note", {}).get("note", {})

            if not note_data:
                return None

            # 提取文字内容
            title = note_data.get("title", "")
            desc = note_data.get("desc", "")
            text = f"{title}\n\n{desc}" if desc else title

            if not text.strip():
                return None

            # 提取图片 URL 列表
            image_urls = []
            image_list = note_data.get("imageList", [])
            for img in image_list:
                # 优先使用原图 URL
                img_url = (
                    img.get("urlDefault")
                    or img.get("url")
                    or img.get("infoList", [{}])[0].get("url", "")
                )
                if img_url:
                    # 确保 URL 以 https 开头
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    image_urls.append(img_url)

            # 提取元数据
            user = note_data.get("user", {})
            interact_info = note_data.get("interactInfo", {})

            metadata = NoteMetadata(
                title=title,
                author=user.get("nickname", "未知作者"),
                publish_time=note_data.get("time", str(datetime.now())),
                likes=self._safe_int(interact_info.get("likedCount", "0")),
                collects=self._safe_int(interact_info.get("collectedCount", "0")),
                comments=self._safe_int(interact_info.get("commentCount", "0")),
            )

            return ExtractedContent(
                url=url,
                text=text,
                image_urls=image_urls,
                metadata=metadata,
            )

        except (KeyError, TypeError, ValueError):
            return None

    def _parse_from_html(
        self, soup: BeautifulSoup, url: str
    ) -> ExtractedContent | None:
        """从 HTML DOM 结构中提取笔记信息（备选方案）

        Args:
            soup: BeautifulSoup 解析对象
            url: 原始请求 URL

        Returns:
            解析成功返回 ExtractedContent，失败返回 None
        """
        try:
            # 提取标题
            title_el = soup.find("meta", property="og:title")
            title = title_el["content"] if title_el and title_el.get("content") else ""

            # 提取描述/正文
            desc_el = soup.find("meta", property="og:description")
            desc = desc_el["content"] if desc_el and desc_el.get("content") else ""

            # 提取作者
            author_el = soup.find("meta", attrs={"name": "author"})
            author = author_el["content"] if author_el and author_el.get("content") else "未知作者"

            text = f"{title}\n\n{desc}" if desc else title
            if not text.strip():
                return None

            # 提取图片
            image_urls = []
            og_image = soup.find("meta", property="og:image")
            if og_image and og_image.get("content"):
                image_urls.append(og_image["content"])

            # 从 img 标签中提取更多图片
            for img in soup.find_all("img", class_=re.compile(r"note|content")):
                src = img.get("src") or img.get("data-src", "")
                if src and src not in image_urls:
                    if src.startswith("//"):
                        src = "https:" + src
                    image_urls.append(src)

            metadata = NoteMetadata(
                title=title,
                author=author,
                publish_time=str(datetime.now()),
                likes=0,
                collects=0,
                comments=0,
            )

            return ExtractedContent(
                url=url,
                text=text,
                image_urls=image_urls,
                metadata=metadata,
            )

        except (KeyError, TypeError, AttributeError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int:
        """安全地将值转换为整数

        处理字符串数字（如 "1.2万"）和其他非标准格式。

        Args:
            value: 待转换的值

        Returns:
            转换后的整数，失败返回 0
        """
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            # 处理 "1.2万" 格式
            value = value.strip()
            if value.endswith("万"):
                try:
                    return int(float(value[:-1]) * 10000)
                except ValueError:
                    return 0
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    def _is_note_unavailable(self, html: str) -> str | None:
        """检测笔记是否已删除或设为私密

        Args:
            html: 页面 HTML 内容

        Returns:
            如果笔记不可用，返回提示信息；否则返回 None
        """
        soup = BeautifulSoup(html, "html.parser")

        # 检查页面中是否包含"已删除"或"私密"等提示
        page_text = soup.get_text()
        unavailable_keywords = [
            "笔记已删除",
            "内容已删除",
            "该笔记已被删除",
            "笔记不存在",
            "内容不存在",
            "无法查看",
            "仅自己可见",
            "该内容已被作者设为私密",
        ]
        for keyword in unavailable_keywords:
            if keyword in page_text:
                return f"该笔记无法访问：{keyword}"

        # 检查 __INITIAL_STATE__ 中的错误状态
        for script in soup.find_all("script"):
            text = script.string or ""
            if "window.__INITIAL_STATE__" in text:
                # 检查是否包含错误码
                if '"code":-1' in text or '"code":"-1"' in text:
                    return "该笔记无法访问：笔记可能已被删除或设为私密"
                if '"noteNotFound"' in text or '"noteDeleted"' in text:
                    return "该笔记无法访问：笔记不存在或已被删除"

        return None

    async def extract(self, url: str) -> dict[str, Any]:
        """从小红书链接提取笔记内容（主入口方法）

        优先使用 xhs-crawl 库提取，失败时降级为 httpx 直接请求。

        Args:
            url: 小红书笔记链接

        Returns:
            字典格式：
            - 成功: {"success": True, "data": ExtractedContent}
            - 失败: {"success": False, "error": "错误描述"}
        """
        # 步骤1: 验证 URL 格式
        if not self._validate_url(url):
            return {
                "success": False,
                "error": "链接格式不正确：请输入有效的小红书笔记链接（支持 xhslink.com 短链或 xiaohongshu.com 长链）",
            }

        # 步骤2: 优先使用 xhs-crawl 提取
        if HAS_XHS_CRAWL:
            logger.info("使用 xhs-crawl 提取笔记内容...")
            result = await self._extract_with_xhs_crawl(url)
            if result and result.get("success"):
                return result
            logger.warning("xhs-crawl 提取失败，降级为 HTTP 方式: %s", result.get("error", ""))

        # 步骤3: 降级为 httpx 直接请求
        logger.info("使用 HTTP 方式提取笔记内容...")
        return await self._extract_with_http(url)

    async def _extract_with_xhs_crawl(self, url: str) -> dict[str, Any]:
        """使用 xhs-crawl 库提取笔记内容

        Args:
            url: 小红书笔记链接

        Returns:
            提取结果字典
        """
        spider = XHSSpider()
        try:
            post = await spider.get_post_data(url)
            if not post:
                return {
                    "success": False,
                    "error": "xhs-crawl 未能获取笔记数据，笔记可能已删除或为私密",
                }

            title = getattr(post, "title", "") or ""
            content_text = getattr(post, "content", "") or ""
            images = getattr(post, "images", []) or []

            text = f"{title}\n\n{content_text}" if content_text else title
            if not text.strip():
                return {
                    "success": False,
                    "error": "xhs-crawl 提取到的内容为空",
                }

            metadata = NoteMetadata(
                title=title,
                author="未知作者",
                publish_time=str(datetime.now()),
                likes=0,
                collects=0,
                comments=0,
            )

            extracted = ExtractedContent(
                url=url,
                text=text,
                image_urls=list(images),
                metadata=metadata,
            )

            return {"success": True, "data": extracted}

        except Exception as e:
            logger.error("xhs-crawl 提取异常: %s", str(e))
            return {
                "success": False,
                "error": f"xhs-crawl 提取异常：{e}",
            }
        finally:
            try:
                await spider.close()
            except Exception:
                pass

    async def _extract_with_http(self, url: str) -> dict[str, Any]:
        """使用 httpx 直接请求提取笔记内容（备选方案）

        Args:
            url: 小红书笔记链接

        Returns:
            提取结果字典
        """

        # 发送 HTTP 请求（含反爬处理）
        response = await self._handle_anti_crawl(url)

        if response is None:
            return {
                "success": False,
                "error": "小红书访问受限，请稍后重试（已重试3次均失败）",
            }

        # 检查 HTTP 状态码
        if response.status_code == 404:
            return {
                "success": False,
                "error": "该笔记不存在：链接指向的笔记可能已被删除",
            }

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"请求失败：HTTP状态码 {response.status_code}",
            }

        html = response.text

        # 检测笔记是否已删除/私密
        unavailable_msg = self._is_note_unavailable(html)
        if unavailable_msg:
            return {
                "success": False,
                "error": unavailable_msg,
            }

        # 解析笔记内容
        content = self._parse_note(html, url)
        if content is None:
            return {
                "success": False,
                "error": "内容解析失败：无法从页面中提取笔记内容，页面结构可能已变更",
            }

        return {
            "success": True,
            "data": content,
        }
