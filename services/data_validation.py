
"""Загрузка и проверка табличных данных для бинарной классификации."""
from __future__ import annotations

import io
from dataclasses import dataclass
import hashlib

import pandas as pd


class UserDataError(ValueError):
    """Ошибка входных данных с понятным сообщением для исследователя."""


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    name: str
    sha256: str


def read_csv_dataset(contents: bytes, name: str) -> LoadedDataset:
    if not contents:
        raise UserDataError("Файл пустой. Выберите CSV с заголовками столбцов.")
    if len(contents) > 15 * 1024 * 1024:
        raise UserDataError("В MVP поддерживаются CSV размером не более 15 МБ.")
    frame = None
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            frame = pd.read_csv(
                io.BytesIO(contents),
                sep=None,
                engine="python",
                encoding=encoding
            )
            break
        except UnicodeDecodeError:
            continue
        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
            raise UserDataError(f"Не удалось прочитать CSV: {exc}") from exc

    if frame is None:
        raise UserDataError("Не удалось определить кодировку CSV. Используйте UTF-8.")
    if frame.empty:
        raise UserDataError("В таблице нет строк с данными.")
    if frame.shape[1] < 2:
        raise UserDataError("Для классификации нужны минимум два столбца: признак и цель.")
    if not frame.columns.is_unique:
        raise UserDataError("Названия столбцов должны быть уникальными.")

    return LoadedDataset(frame, name, hashlib.sha256(contents).hexdigest())


def validate_classification_input(
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    positive_label: object,
    test_size: float,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    """Return validated copy; don't silently drop rows with missing outcomes."""
    if target_column not in frame.columns:
        raise UserDataError("Выбранный целевой столбец отсутствует в данных.")
    if not feature_columns:
        raise UserDataError("Выберите хотя бы один признак для обучения.")
    if len(set(feature_columns)) != len(feature_columns):
        raise UserDataError("Список признаков содержит повторы.")
    if target_column in feature_columns:
        raise UserDataError("Целевой столбец не может быть входным признаком: это утечка данных.")
    if any(col not in frame.columns for col in feature_columns):
        raise UserDataError("Некоторые выбранные признаки отсутствуют в таблице.")
    if not set(categorical_columns).issubset(feature_columns):
        raise UserDataError("Категориальные признаки должны входить в список используемых признаков.")
    if not (0.1 <= test_size <= 0.4):
        raise UserDataError("Доля тестовой выборки должна быть от 0.1 до 0.4.")

    clean = frame.drop_duplicates().copy() if remove_duplicates else frame.copy()

    if clean[target_column].isna().any():
        raise UserDataError("Целевой столбец содержит пропуски. Исправьте данные перед запуском.")

    classes = list(pd.unique(clean[target_column]))
    if len(classes) != 2:
        raise UserDataError(
            f"Требуется ровно два класса в цели. Обнаружено: {len(classes)}."
        )
    if positive_label not in classes:
        raise UserDataError("Выбранный положительный класс отсутствует в данных.")

    class_counts = clean[target_column].value_counts(dropna=False)
    if (class_counts < 2).any():
        raise UserDataError(
            "Для стратифицированного разделения в каждом классе нужно минимум два наблюдения."
        )

    test_count = int(__import__("math").ceil(len(clean) * test_size))
    train_count = len(clean) - test_count
    if min(test_count, train_count) < 2:
        raise UserDataError("Слишком мало наблюдений для разделения на два класса.")

    for col in feature_columns:
        if clean[col].isna().all():
            raise UserDataError(f"Признак «{col}» целиком состоит из пропусков.")

    # Проверяем числовые типы после применения пользовательских кодов пропусков.
    return clean
