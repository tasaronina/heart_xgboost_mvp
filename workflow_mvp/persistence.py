"""Локальные метаданные запусков в отдельной БД. Без сырых строк пациентов."""
from __future__ import annotations
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import uuid

DB = Path(os.environ.get(
    "WORKFLOW_DB_PATH",
    Path(__file__).resolve().parent / "_data" / "workflow_runs.db",
))


def _fallback_study_name(summary: dict, dataset: str, method: str, created: str) -> str:
    """Понятное имя для старых записей, где название исследования ещё не сохранялось."""
    from .spec import METHODS

    title = METHODS.get(method, {}).get("title", method)
    date = str(created).replace("T", " ")[:16]
    return summary.get("study_name") or f"Анализ {dataset} — {title} ({date})"


def safe_summary(summary: dict) -> dict:
    # Оценки по отдельным строкам и вероятности остаются только в RAM.
    return {k: v for k, v in summary.items() if k != "_viz"}


def _ensure_schema(cn: sqlite3.Connection) -> None:
    cn.execute(
        "CREATE TABLE IF NOT EXISTS runs "
        "(id TEXT PRIMARY KEY, created TEXT, dataset TEXT, method TEXT, summary TEXT)"
    )
    cn.execute(
        "CREATE TABLE IF NOT EXISTS run_assets ("
        "run_id TEXT NOT NULL, asset_key TEXT NOT NULL, mime TEXT NOT NULL, "
        "content BLOB NOT NULL, PRIMARY KEY (run_id, asset_key), "
        "FOREIGN KEY (run_id) REFERENCES runs(id))"
    )


def save_summary(
    summary: dict,
    figures: dict[str, bytes] | None = None,
    report: bytes | None = None,
    db: Path = DB,
) -> str:
    db.parent.mkdir(parents=True, exist_ok=True)
    rid = uuid.uuid4().hex[:12]
    with closing(sqlite3.connect(db)) as cn:
        with cn:
            _ensure_schema(cn)
            cn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                       (rid, summary["created_at"], summary["dataset"]["dataset_name"],
                        summary["model"], json.dumps(safe_summary(summary), ensure_ascii=False, default=str)))
            for key, content in (figures or {}).items():
                cn.execute(
                    "INSERT OR REPLACE INTO run_assets VALUES (?, ?, ?, ?)",
                    (rid, f"figure:{key}", "image/png", sqlite3.Binary(content)),
                )
            if report:
                cn.execute(
                    "INSERT OR REPLACE INTO run_assets VALUES (?, ?, ?, ?)",
                    (
                        rid,
                        "report:docx",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        sqlite3.Binary(report),
                    ),
                )
    return rid


def recent_runs(db: Path = DB) -> list[dict]:
    if not db.exists():
        return []
    with closing(sqlite3.connect(db)) as cn:
        _ensure_schema(cn)
        rows = cn.execute(
            "SELECT r.id,r.created,r.dataset,r.method,r.summary,"
            "SUM(CASE WHEN a.asset_key LIKE 'figure:%' THEN 1 ELSE 0 END),"
            "MAX(CASE WHEN a.asset_key='report:docx' THEN 1 ELSE 0 END) "
            "FROM runs r LEFT JOIN run_assets a ON a.run_id=r.id "
            "GROUP BY r.id,r.created,r.dataset,r.method,r.summary "
            "ORDER BY created DESC LIMIT 30"
        ).fetchall()
    result = []
    for rid, created, dataset, method, raw, figure_count, has_report in rows:
        try:
            summary = json.loads(raw)
            metrics = summary.get("metrics", {})
        except (TypeError, json.JSONDecodeError):
            summary = {}
            metrics = {}
        result.append(
            {
                "id": rid,
                "created": created,
                "dataset": dataset,
                "method": method,
                "study_name": _fallback_study_name(summary, dataset, method, created),
                "n_classes": summary.get("n_classes"),
                "figure_count": int(figure_count or 0),
                "has_report": bool(has_report),
                "accuracy": metrics.get("accuracy"),
                "f1": metrics.get("f1", metrics.get("f1_macro")),
            }
        )
    return result


def get_run(run_id: str, db: Path = DB) -> dict | None:
    """Возвращает сохранённые результаты и доступные артефакты запуска."""
    if not db.exists():
        return None
    with closing(sqlite3.connect(db)) as cn:
        _ensure_schema(cn)
        row = cn.execute(
            "SELECT id,created,dataset,method,summary FROM runs WHERE id=?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        assets = cn.execute(
            "SELECT asset_key,mime,content FROM run_assets WHERE run_id=?", (run_id,)
        ).fetchall()
    try:
        summary = json.loads(row[4])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Сохранённая запись содержит повреждённый JSON.") from exc
    figures: dict[str, bytes] = {}
    report = None
    for key, _mime, content in assets:
        if key.startswith("figure:"):
            figures[key.split(":", 1)[1]] = bytes(content)
        elif key == "report:docx":
            report = bytes(content)
    return {
        "id": row[0],
        "created": row[1],
        "dataset": row[2],
        "method": row[3],
        "summary": summary,
        "figures": figures,
        "report": report,
    }
