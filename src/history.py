"""
SQLite 诊断历史持久化模块

每个诊断结果写入本地 SQLite 数据库，支持历史查询和聚合统计。
数据库文件默认位于项目根目录 edge-sense/history.db。
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).resolve().parent.parent / "history.db"


class DiagnosisStore:
    """线程安全的 SQLite 诊断存储"""

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS diagnoses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    frame_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    cause TEXT DEFAULT '',
                    confidence REAL DEFAULT 0.0,
                    inference_time_s REAL DEFAULT 0.0,
                    details TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON diagnoses(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON diagnoses(status)
            """)
            conn.commit()
            conn.close()

    def save(self, diagnosis: dict):
        """保存一条诊断记录"""
        details = {
            "color": diagnosis.get("color", {}),
            "motion": diagnosis.get("motion", {}),
            "flicker": diagnosis.get("flicker", {}),
            "ocr": diagnosis.get("ocr", {}),
        }
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                """INSERT INTO diagnoses
                   (frame_id, timestamp, status, cause, confidence, inference_time_s, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    diagnosis.get("frame_id", 0),
                    diagnosis.get("timestamp", time.time()),
                    diagnosis.get("status", "unknown"),
                    diagnosis.get("cause", ""),
                    diagnosis.get("confidence", 0.0),
                    diagnosis.get("inference_time_s", 0.0),
                    json.dumps(details, ensure_ascii=False),
                ),
            )
            conn.commit()
            conn.close()

    def query(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """查询最近诊断记录（按时间倒序）"""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT id, frame_id, timestamp, status, cause, confidence,
                          inference_time_s, details
                   FROM diagnoses
                   ORDER BY timestamp DESC
                   LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            conn.close()

        return [_row_to_dict(r) for r in rows]

    def stats(self) -> dict:
        """聚合统计：总数、按状态分布、平均置信度"""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            total = conn.execute("SELECT COUNT(*) as n FROM diagnoses").fetchone()["n"]

            by_status = {}
            if total > 0:
                rows = conn.execute(
                    "SELECT status, COUNT(*) as n FROM diagnoses GROUP BY status"
                ).fetchall()
                by_status = {r["status"]: r["n"] for r in rows}

            avg_conf = 0.0
            avg_latency = 0.0
            if total > 0:
                avg_conf = conn.execute(
                    "SELECT AVG(confidence) as v FROM diagnoses"
                ).fetchone()["v"] or 0.0
                avg_latency = conn.execute(
                    "SELECT AVG(inference_time_s) as v FROM diagnoses"
                ).fetchone()["v"] or 0.0

            conn.close()

        return {
            "total": total,
            "by_status": by_status,
            "avg_confidence": round(avg_conf, 3),
            "avg_inference_time_s": round(avg_latency, 2),
        }

    def recent_confidence_series(self, limit: int = 50) -> list[dict]:
        """获取最近 N 条记录的置信度时间序列（用于趋势图）"""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT frame_id, timestamp, confidence, status
                   FROM diagnoses
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            conn.close()

        # 按时间升序返回（适合前端绑图）
        series = [
            {
                "frame_id": r["frame_id"],
                "timestamp": r["timestamp"],
                "confidence": r["confidence"],
                "status": r["status"],
            }
            for r in reversed(rows)
        ]
        return series


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    try:
        d["details"] = json.loads(d["details"])
    except (json.JSONDecodeError, KeyError):
        d["details"] = {}
    return d


# 模块级单例
_store: Optional[DiagnosisStore] = None


def get_store() -> DiagnosisStore:
    global _store
    if _store is None:
        _store = DiagnosisStore()
    return _store
