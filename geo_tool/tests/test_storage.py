"""存储层单元测试

测试 Database（SQLite）和 FileStore（JSON 文件）的核心功能。
"""

import json
import os
import tempfile

import pytest

from geo_tool.storage.database import Database
from geo_tool.storage.file_store import FileStore


# ==================== Database 测试 ====================


class TestDatabase:
    """SQLite 数据库存储测试"""

    def setup_method(self):
        """每个测试方法前创建临时数据库"""
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.db = Database(db_path=self.db_path)

    def test_init_creates_db_file(self):
        """初始化时应自动创建数据库文件"""
        assert os.path.exists(self.db_path)

    def test_init_creates_tables(self):
        """初始化时应自动创建 task_records 和 score_history 表"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "task_records" in tables
        assert "score_history" in tables

    def test_save_and_get_task(self):
        """保存任务后应能正确查询"""
        task = {
            "id": "task-001",
            "url": "https://www.xiaohongshu.com/explore/abc123",
            "template_type": "ranking",
            "status": "pending",
            "created_at": "2025-01-01T12:00:00",
        }
        self.db.save_task(task)
        result = self.db.get_task("task-001")

        assert result is not None
        assert result["id"] == "task-001"
        assert result["url"] == task["url"]
        assert result["template_type"] == "ranking"
        assert result["status"] == "pending"

    def test_get_task_not_found(self):
        """查询不存在的任务应返回 None"""
        result = self.db.get_task("nonexistent")
        assert result is None

    def test_save_task_upsert(self):
        """重复保存同一 ID 的任务应更新记录"""
        task = {
            "id": "task-002",
            "url": "https://www.xiaohongshu.com/explore/xyz",
            "status": "pending",
            "created_at": "2025-01-01T12:00:00",
        }
        self.db.save_task(task)

        # 更新状态
        task["status"] = "completed"
        task["completed_at"] = "2025-01-01T13:00:00"
        self.db.save_task(task)

        result = self.db.get_task("task-002")
        assert result["status"] == "completed"
        assert result["completed_at"] == "2025-01-01T13:00:00"

    def test_save_task_with_error(self):
        """保存带错误信息的任务"""
        task = {
            "id": "task-003",
            "url": "https://invalid.url",
            "status": "failed",
            "created_at": "2025-01-01T12:00:00",
            "error": "无效的小红书链接",
        }
        self.db.save_task(task)
        result = self.db.get_task("task-003")
        assert result["error"] == "无效的小红书链接"

    def test_save_and_get_scores(self):
        """保存评分后应能按任务 ID 查询"""
        # 先创建任务
        self.db.save_task({
            "id": "task-010",
            "url": "https://www.xiaohongshu.com/explore/test",
            "status": "completed",
            "created_at": "2025-01-01T12:00:00",
        })

        scores_data = [
            {
                "id": "score-001",
                "task_id": "task-010",
                "weighted_total": 82.5,
                "quality_level": "good",
                "dimension_scores_json": [
                    {"dimension": "结构化", "score": 85, "weight": 0.30},
                ],
                "created_at": "2025-01-01T12:01:00",
            },
            {
                "id": "score-002",
                "task_id": "task-010",
                "weighted_total": 90.0,
                "quality_level": "excellent",
                "dimension_scores_json": json.dumps(
                    [{"dimension": "结构化", "score": 95, "weight": 0.30}],
                    ensure_ascii=False,
                ),
                "created_at": "2025-01-01T13:00:00",
            },
        ]

        for s in scores_data:
            self.db.save_score(s)

        results = self.db.get_scores_by_task("task-010")
        assert len(results) == 2
        # 按 created_at 降序
        assert results[0]["id"] == "score-002"
        assert results[1]["id"] == "score-001"

    def test_get_scores_empty(self):
        """查询无评分记录的任务应返回空列表"""
        results = self.db.get_scores_by_task("nonexistent-task")
        assert results == []

    def test_default_db_path_from_env(self, monkeypatch):
        """应从环境变量 SQLITE_DB_PATH 读取数据库路径"""
        custom_path = os.path.join(self.tmp_dir, "custom", "env.db")
        monkeypatch.setenv("SQLITE_DB_PATH", custom_path)
        db = Database()
        assert db.db_path == custom_path
        assert os.path.exists(custom_path)


# ==================== FileStore 测试 ====================


class TestFileStore:
    """JSON 文件存储测试"""

    def setup_method(self):
        """每个测试方法前创建临时存储目录"""
        self.tmp_dir = tempfile.mkdtemp()
        self.store = FileStore(storage_dir=self.tmp_dir)

    def test_init_creates_directory(self):
        """初始化时应自动创建存储目录"""
        new_dir = os.path.join(self.tmp_dir, "sub", "dir")
        store = FileStore(storage_dir=new_dir)
        assert os.path.isdir(new_dir)

    def test_save_and_load_json(self):
        """保存后应能正确读取 JSON 数据"""
        data = {"url": "https://example.com", "text": "测试内容", "tags": ["标签1"]}
        filepath = self.store.save_json(data, file_type="extract")

        # 从文件名加载
        filename = os.path.basename(filepath)
        loaded = self.store.load_json(filename)

        assert loaded["url"] == data["url"]
        assert loaded["text"] == data["text"]
        assert loaded["tags"] == data["tags"]

    def test_save_json_custom_filename(self):
        """支持自定义文件名"""
        data = {"key": "value"}
        filepath = self.store.save_json(data, file_type="test", filename="custom.json")
        assert filepath.endswith("custom.json")

        loaded = self.store.load_json("custom.json")
        assert loaded["key"] == "value"

    def test_save_json_filename_format(self):
        """自动生成的文件名应符合 {type}_{timestamp}.json 格式"""
        filepath = self.store.save_json({"a": 1}, file_type="extract")
        filename = os.path.basename(filepath)

        assert filename.startswith("extract_")
        assert filename.endswith(".json")
        # 时间戳部分格式：YYYYMMDD_HHMMSS
        ts_part = filename[len("extract_"):-len(".json")]
        assert len(ts_part) == 15  # 20250101_120000

    def test_save_json_chinese_content(self):
        """应正确保存和读取中文内容"""
        data = {"标题": "小红书笔记", "内容": "这是一段中文测试内容"}
        filepath = self.store.save_json(data, file_type="note")
        filename = os.path.basename(filepath)
        loaded = self.store.load_json(filename)
        assert loaded["标题"] == "小红书笔记"

    def test_load_json_not_found(self):
        """读取不存在的文件应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            self.store.load_json("nonexistent.json")

    def test_list_files_all(self):
        """列出所有 JSON 文件"""
        self.store.save_json({}, file_type="extract", filename="extract_001.json")
        self.store.save_json({}, file_type="keyword", filename="keyword_001.json")

        files = self.store.list_files()
        assert len(files) == 2
        assert "extract_001.json" in files
        assert "keyword_001.json" in files

    def test_list_files_by_type(self):
        """按类型前缀过滤文件"""
        self.store.save_json({}, file_type="extract", filename="extract_001.json")
        self.store.save_json({}, file_type="extract", filename="extract_002.json")
        self.store.save_json({}, file_type="keyword", filename="keyword_001.json")

        files = self.store.list_files(file_type="extract")
        assert len(files) == 2
        assert all(f.startswith("extract") for f in files)

    def test_list_files_empty(self):
        """空目录应返回空列表"""
        files = self.store.list_files()
        assert files == []

    def test_list_files_sorted(self):
        """文件列表应按名称排序"""
        self.store.save_json({}, file_type="a", filename="c_file.json")
        self.store.save_json({}, file_type="a", filename="a_file.json")
        self.store.save_json({}, file_type="a", filename="b_file.json")

        files = self.store.list_files()
        assert files == ["a_file.json", "b_file.json", "c_file.json"]

    def test_default_storage_dir_from_env(self, monkeypatch):
        """应从环境变量 JSON_STORAGE_DIR 读取存储目录"""
        custom_dir = os.path.join(self.tmp_dir, "env_store")
        monkeypatch.setenv("JSON_STORAGE_DIR", custom_dir)
        store = FileStore()
        assert store.storage_dir == custom_dir
        assert os.path.isdir(custom_dir)
