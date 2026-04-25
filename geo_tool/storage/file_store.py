"""JSON 文件存储模块

使用 Python 内置 json 模块实现提取结果和中间数据的持久化。
文件名格式：{type}_{timestamp}.json
适配 2核4G 云服务器，避免一次性加载大量数据到内存。
"""

import json
import os
from datetime import datetime
from typing import Any


# 存储目录，从环境变量读取，默认 data/json_store
DEFAULT_STORAGE_DIR = "data/json_store"


class FileStore:
    """JSON 文件存储管理类

    提供 JSON 文件的读写和列表操作。
    自动创建存储目录。
    """

    def __init__(self, storage_dir: str | None = None):
        """初始化文件存储

        Args:
            storage_dir: 存储目录路径，为 None 时从环境变量 JSON_STORAGE_DIR 读取
        """
        self.storage_dir = storage_dir or os.environ.get(
            "JSON_STORAGE_DIR", DEFAULT_STORAGE_DIR
        )
        os.makedirs(self.storage_dir, exist_ok=True)

    def _generate_filename(self, file_type: str) -> str:
        """生成带时间戳的文件名

        Args:
            file_type: 文件类型前缀，如 'extract'、'keyword'

        Returns:
            格式为 {type}_{YYYYMMDD_HHMMSS}.json 的文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{file_type}_{timestamp}.json"

    def save_json(self, data: Any, file_type: str, filename: str | None = None) -> str:
        """保存数据为 JSON 文件

        Args:
            data: 要保存的数据（可序列化为 JSON 的任意对象）
            file_type: 文件类型前缀，用于自动生成文件名
            filename: 自定义文件名，为 None 时自动生成

        Returns:
            保存的文件完整路径
        """
        if filename is None:
            filename = self._generate_filename(file_type)

        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def load_json(self, filename: str) -> Any:
        """读取 JSON 文件内容

        Args:
            filename: 文件名（不含目录路径）

        Returns:
            解析后的 JSON 数据

        Raises:
            FileNotFoundError: 文件不存在时抛出
        """
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_files(self, file_type: str | None = None) -> list[str]:
        """列出存储目录中的 JSON 文件

        Args:
            file_type: 可选的文件类型前缀过滤，为 None 时返回所有 JSON 文件

        Returns:
            文件名列表，按名称排序
        """
        files = []
        for name in os.listdir(self.storage_dir):
            if not name.endswith(".json"):
                continue
            if file_type is not None and not name.startswith(file_type):
                continue
            files.append(name)
        return sorted(files)
