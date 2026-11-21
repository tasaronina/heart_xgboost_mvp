"""Чистые операции над рабочим черновиком, независимые от Streamlit-виджетов."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .spec import METHODS, commit_active_params, normalize_settings, select_method


def initialize_dataset_settings(
    settings: dict,
    frame: pd.DataFrame,
    preserve_existing: bool = False,
) -> tuple[dict, list[str]]:
    """Инициализирует Y/X при смене CSV до открытия панели подготовки.

    При восстановлении шаблона валидные имена сохраняются. Несовместимые поля не
    замалчиваются: возвращаем сообщения, которые интерфейс показывает пользователю.
    """
    draft = normalize_settings(settings)
    preprocess = draft["preprocess"]
    columns = list(frame.columns)
    warnings: list[str] = []
    if not columns:
        return draft, ["В CSV нет доступных столбцов."]

    previous_target = preprocess.get("target", "")
    if preserve_existing and previous_target in columns:
        target = previous_target
    else:
        target = (
            "target" if "target" in columns else
            "Outcome" if "Outcome" in columns else
            "outcome" if "outcome" in columns else
            columns[-1]
        )
        if preserve_existing and previous_target and previous_target not in columns:
            warnings.append(
                f"Сохранённый целевой столбец «{previous_target}» отсутствует в новом CSV. "
                f"Временно выбран «{target}»; проверьте блок «Подготовка данных»."
            )
    preprocess["target"] = target

    available_features = [column for column in columns if column != target]
    previous_features = list(preprocess.get("features", []))
    if preserve_existing and previous_features:
        features = [column for column in previous_features if column in available_features]
        missing = [column for column in previous_features if column not in available_features]
        if missing:
            warnings.append(
                "В новом CSV отсутствуют сохранённые признаки: " + ", ".join(missing) +
                ". Проверьте список X."
            )
        if not features:
            features = available_features
    else:
        features = available_features
    preprocess["features"] = features

    previous_categories = list(preprocess.get("categorical", []))
    if preserve_existing:
        preprocess["categorical"] = [column for column in previous_categories if column in features]
    else:
        preprocess["categorical"] = [
            column for column in features if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        preprocess["missing_rules"] = {}
        preprocess["remove_duplicates"] = False

    classes = sorted(frame[target].dropna().astype(str).unique().tolist())
    previous_positive = str(preprocess.get("positive", ""))
    if len(classes) == 2:
        preprocess["positive"] = (
            previous_positive if preserve_existing and previous_positive in classes
            else "1" if "1" in classes else classes[-1]
        )
    else:
        preprocess["positive"] = ""

    numeric = [column for column in features if column not in preprocess["categorical"]]
    previous_descriptive = list(preprocess.get("descriptive_columns", []))
    if preserve_existing:
        descriptive = [column for column in previous_descriptive if column in numeric]
        preprocess["descriptive_columns"] = descriptive or numeric[:6]
    else:
        preprocess["descriptive_columns"] = numeric[:6]
    return draft, warnings


def update_method(settings: dict, method: str) -> dict:
    """Публичная операция переключения метода для UI и стресс-тестов."""
    return select_method(settings, method)


def update_method_params(settings: dict, method: str, changes: dict) -> dict:
    """Обновляет только параметры указанного метода, не затрагивая остальные."""
    draft = normalize_settings(settings)
    if method not in METHODS:
        raise ValueError("Неизвестный метод анализа.")
    params = deepcopy(draft["analysis"]["params_by_method"][method])
    params.update({key: value for key, value in changes.items() if key in params})
    draft["analysis"]["params_by_method"][method] = params
    if draft["analysis"]["method"] == method:
        commit_active_params(draft, params)
    return draft
