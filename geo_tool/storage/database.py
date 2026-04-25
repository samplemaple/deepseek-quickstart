"""SQLite 数据库存储模块

使用 Python 内置 sqlite3 模块实现轻量级持久化存储。
包含两个表：task_records（任务记录）和 score_history（评分历史）。
适配 2核4G 云服务器部署环境，SQLite 完全够用。
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Any


# 数据库文件路径，从环境变量读取，默认 data/geo_tool.db
DEFAULT_DB_PATH = "data/geo_tool.db"


class Database:
    """SQLite 数据库管理类

    提供任务记录和评分历史的增删改查操作。
    自动创建数据库文件和表结构。
    """

    def __init__(self, db_path: str | None = None):
        """初始化数据库连接

        Args:
            db_path: 数据库文件路径，为 None 时从环境变量 SQLITE_DB_PATH 读取
        """
        self.db_path = db_path or os.environ.get("SQLITE_DB_PATH", DEFAULT_DB_PATH)
        self._ensure_directory()
        self._init_tables()

    def _ensure_directory(self) -> None:
        """确保数据库文件所在目录存在"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接，启用 Row 工厂以便按列名访问"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """初始化数据库表结构（如果不存在则创建）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # 任务记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS task_records (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    template_type TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                )
            """)
            # 评分历史表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS score_history (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    weighted_total REAL NOT NULL,
                    quality_level TEXT NOT NULL,
                    dimension_scores_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES task_records(id)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def save_task(self, task: dict[str, Any]) -> None:
        """保存任务记录

        Args:
            task: 任务数据字典，必须包含 id, url, status, created_at 字段
                  可选字段：template_type, completed_at, error
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_records
                    (id, url, template_type, status, created_at, completed_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task["id"],
                    task["url"],
                    task.get("template_type"),
                    task["status"],
                    task["created_at"],
                    task.get("completed_at"),
                    task.get("error"),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """根据 ID 查询任务记录

        Args:
            task_id: 任务唯一标识

        Returns:
            任务数据字典，不存在时返回 None
        """
        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT * FROM task_records WHERE id = ?", (task_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def save_score(self, score: dict[str, Any]) -> None:
        """保存评分记录

        Args:
            score: 评分数据字典，必须包含 id, task_id, weighted_total,
                   quality_level, dimension_scores_json, created_at 字段。
                   dimension_scores_json 可以是 dict/list（会自动序列化）或 JSON 字符串。
        """
        # 如果 dimension_scores_json 不是字符串，自动序列化
        dim_json = score["dimension_scores_json"]
        if not isinstance(dim_json, str):
            dim_json = json.dumps(dim_json, ensure_ascii=False)

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO score_history
                    (id, task_id, weighted_total, quality_level, dimension_scores_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    score["id"],
                    score["task_id"],
                    score["weighted_total"],
                    score["quality_level"],
                    dim_json,
                    score["created_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_scores_by_task(self, task_id: str) -> list[dict[str, Any]]:
        """查询指定任务的所有评分记录

        Args:
            task_id: 任务唯一标识

        Returns:
            评分记录列表，按创建时间降序排列
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT * FROM score_history WHERE task_id = ? ORDER BY created_at DESC",
                (task_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
