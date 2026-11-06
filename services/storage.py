"""Local-only SQLite history: no source datasets are persisted."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any

DB_PATH = Path(__file__).resolve().parents[1] / "storage" / "runs.db"


def _connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db(path: Path = DB_PATH) -> None:
    with _connect(path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                dataset_sha256 TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
        """)


def save_run(result: dict[str, Any], dataset_name: str,
             dataset_sha256: str, path: Path = DB_PATH) -> str:
    init_db(path)
    run_id = uuid.uuid4().hex[:12]
    created_at = result.get("created_at", datetime.now(timezone.utc).isoformat())
    with _connect(path) as con:
        con.execute(
            "INSERT INTO runs(run_id, created_at, dataset_name, dataset_sha256, result_json) VALUES(?,?,?,?,?)",
            (run_id, created_at, dataset_name, dataset_sha256,
             json.dumps(result, ensure_ascii=False, allow_nan=False)),
        )
    return run_id


def list_runs(path: Path = DB_PATH, limit: int = 50) -> list[dict[str, Any]]:
    init_db(path)
    with _connect(path) as con:
        rows = con.execute(
            "SELECT run_id, created_at, dataset_name, dataset_sha256, result_json "
            "FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [
        {"run_id": row["run_id"], "created_at": row["created_at"],
         "dataset_name": row["dataset_name"], "dataset_sha256": row["dataset_sha256"],
         "result": json.loads(row["result_json"])}
        for row in rows
    ]
