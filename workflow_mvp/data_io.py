"""Чтение CSV с разумными пределами для локального прототипа."""
from __future__ import annotations
import hashlib
from io import BytesIO
import pandas as pd

MAX_BYTES = 15 * 1024 * 1024


def load_csv(raw: bytes, file_name: str) -> tuple[pd.DataFrame, dict]:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("CSV пустой или превышает лимит 15 МБ.")
    df = None
    last = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            df = pd.read_csv(BytesIO(raw), sep=None, engine="python", encoding=encoding)
            break
        except UnicodeError as e:
            last = e
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as e:
            raise ValueError(f"Ошибка чтения CSV: {e}") from e
    if df is None:
        raise ValueError(f"Неверная кодировка CSV: {last}")
    if len(df) < 4 or len(df.columns) < 2:
        raise ValueError("Слишком мало строк или столбцов для классификации.")
    if not df.columns.is_unique:
        raise ValueError("У CSV повторяются заголовки столбцов.")
    if len(df) > 100_000 or len(df.columns) > 200:
        raise ValueError("Ограничение MVP: не более 100 000 строк и 200 столбцов.")
    return df, {"dataset_name": file_name, "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": int(len(df)), "columns": int(len(df.columns)),
                "duplicates": int(df.duplicated().sum()),
                "missing_cells": int(df.isna().sum().sum())}


def replace_markers(frame: pd.DataFrame, rules: dict) -> pd.DataFrame:
    out = frame.copy()
    for column, tokens in rules.items():
        if column not in out.columns:
            continue
        series = out[column]
        for token in tokens:
            val = str(token).strip()
            if not val:
                continue
            if pd.api.types.is_numeric_dtype(series):
                try:
                    num = float(val)
                except ValueError:
                    continue
                out[column] = out[column].mask(out[column].eq(num))
            else:
                out[column] = out[column].mask(out[column].astype("string").str.strip().eq(val).fillna(False))
            series = out[column]
    return out
