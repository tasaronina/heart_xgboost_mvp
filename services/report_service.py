"""Machine-readable result export for sharing a reproducible experiment."""
from __future__ import annotations
import json
import pandas as pd
from typing import Any


def report_json(result: dict[str, Any], dataset_name: str, sha256: str) -> str:
    document = {
        "project": "MVP: визуально настраиваемый анализ табличных данных / XGBoost",
        "dataset_name": dataset_name,
        "dataset_sha256": sha256,
        "research_only": True,
        "result": result,
    }
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)


def predictions_csv(result: dict[str, Any]) -> bytes:
    return pd.DataFrame(result["predictions"]).to_csv(index=False).encode("utf-8-sig")
