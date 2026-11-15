"""Описания блоков, методов, графиков и правила совместимости."""
from __future__ import annotations

from copy import deepcopy
from math import ceil

import pandas as pd

BLOCKS = ("input", "preprocess", "analysis", "output")
BLOCK_TITLES = {
    "input": "1. Входные данные",
    "preprocess": "2. Подготовка данных",
    "analysis": "3. Анализ",
    "output": "4. Вывод",
}
EXPECTED_EDGES = [("input", "preprocess"), ("preprocess", "analysis"), ("analysis", "output")]

METHODS = {
    "xgboost": {
        "title": "XGBoost",
        "description": "Градиентный бустинг деревьев",
        "scaling": "none",
        "parameters": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
                       "subsample": 0.8, "colsample_bytree": 0.8},
        "importance_title": "Значимость признаков: gain",
        "shap": True,
    },
    "random_forest": {
        "title": "Random Forest",
        "description": "Ансамбль случайных деревьев",
        "scaling": "none",
        "parameters": {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 2},
        "importance_title": "Значимость признаков: impurity",
        "shap": False,
    },
    "logistic_regression": {
        "title": "Логистическая регрессия",
        "description": "Линейный классификатор с регуляризацией",
        "scaling": "standard",
        "parameters": {"C": 1.0, "max_iter": 800},
        "importance_title": "Модуль коэффициентов модели",
        "shap": False,
    },
}

CHARTS = {
    "confusion": "Матрица ошибок",
    "roc": "ROC-кривая",
    "pr": "PR-кривая",
    "importance": "Значимость признаков / коэффициенты",
    "shap": "SHAP Beeswarm (XGBoost)",
    "hist": "Гистограммы",
    "box": "Boxplot",
    "corr": "Корреляционная карта",
}


def default_settings() -> dict:
    return {
        "input": {"source": "Демонстрационный heart.csv"},
        "preprocess": {
            "target": "", "positive": "", "features": [], "categorical": [],
            "test_size": 0.2, "random_state": 42, "scaling": "auto",
            "remove_duplicates": False, "missing_rules": {},
            "descriptive": True, "descriptive_columns": [],
        },
        "analysis": {"method": "xgboost", "params": deepcopy(METHODS["xgboost"]["parameters"])},
        "output": {"charts": ["confusion", "roc", "importance"], "shap_class": ""},
    }


def available_charts(settings: dict, n_classes: int) -> dict[str, tuple[bool, str]]:
    """Отделяет недоступность из-за типа задачи от временной неподдержки."""
    m = settings["analysis"]["method"]
    descr = bool(settings["preprocess"].get("descriptive"))
    status = {}
    for k in CHARTS:
        if k == "shap" and not METHODS[m]["shap"]:
            status[k] = (
                False,
                "SHAP применим и к этому классу моделей, но в текущей версии "
                "программы график реализован только для XGBoost.",
            )
        elif k in ("hist", "box", "corr") and not descr:
            status[k] = (False, "Включите описательную статистику во втором блоке.")
        elif k in ("roc", "pr") and not 2 <= n_classes <= 4:
            status[k] = (False, "Нужна классификация с 2–4 классами.")
        else:
            status[k] = (True, "")
    return status


def validate_edges(node_ids: list[str], edge_pairs: list[tuple[str, str]]) -> None:
    if len(node_ids) != 4 or set(node_ids) != set(BLOCKS):
        raise ValueError("Для этого MVP нужны четыре разных блока: Вход → Подготовка → Анализ → Вывод.")
    if len(edge_pairs) != 3 or set(edge_pairs) != set(EXPECTED_EDGES):
        raise ValueError("Соедините блоки именно так: Вход → Подготовка → Анализ → Вывод. Лишние линии и циклы не допускаются.")


def validate_settings(settings: dict, columns: list[str], n_classes: int) -> None:
    p = settings["preprocess"]
    m = settings["analysis"]["method"]
    if m not in METHODS:
        raise ValueError("Выбран неизвестный аналитический метод.")
    if not 2 <= n_classes <= 4:
        raise ValueError(f"В целевой переменной {n_classes} классов; требуется 2–4.")
    target = p["target"]
    feats = p["features"]
    if not target or target not in columns:
        raise ValueError("Выберите целевую переменную в блоке подготовки.")
    if not feats or len(set(feats)) != len(feats) or target in feats or any(f not in columns for f in feats):
        raise ValueError("Проверьте список X: минимум один столбец, без цели Y и неизвестных имён.")
    if not set(p.get("categorical", [])).issubset(feats):
        raise ValueError("Категориальные признаки должны входить в X.")
    if p.get("scaling", "auto") not in ("auto", "none", "standard"):
        raise ValueError("Неподдерживаемый способ масштабирования.")
    if not 0.1 <= float(p.get("test_size", 0.2)) <= 0.4:
        raise ValueError("Доля Test должна быть от 0.1 до 0.4.")
    for col, tokens in p.get("missing_rules", {}).items():
        if col not in columns or not isinstance(tokens, list) or len(tokens) > 20:
            raise ValueError("Некорректное правило пропусков.")
    for chart in settings["output"].get("charts", []):
        if chart not in CHARTS:
            raise ValueError(f"Неизвестный график: {chart}")
        ok, why = available_charts(settings, n_classes)[chart]
        if not ok:
            raise ValueError(f"Недоступен график «{CHARTS[chart]}»: {why}")


def validate_preflight(settings: dict, frame: pd.DataFrame) -> int:
    """Полная предпусковая проверка без разделения данных и обучения модели."""
    from .data_io import replace_markers

    p = settings["preprocess"]
    target = p.get("target", "")
    if not target or target not in frame.columns:
        raise ValueError("Блок 2 «Подготовка»: выберите целевой столбец Y.")
    if frame[target].isna().any():
        raise ValueError(
            "Блок 2 «Подготовка»: в целевом столбце Y есть пропуски. "
            "Их нельзя заполнять автоматически."
        )
    classes = sorted(frame[target].astype(str).unique().tolist())
    n_classes = len(classes)
    try:
        validate_settings(settings, list(frame.columns), n_classes)
    except ValueError as exc:
        raise ValueError(f"Настройки пайплайна: {exc}") from exc

    counts = frame[target].astype(str).value_counts()
    if counts.empty or int(counts.min()) < 2:
        raise ValueError(
            "Блок 2 «Подготовка»: для Train/Test требуется не менее двух "
            "наблюдений каждого класса."
        )
    test_size = float(p.get("test_size", 0.2))
    test_rows = ceil(len(frame) * test_size)
    if min(test_rows, len(frame) - test_rows) < n_classes:
        raise ValueError(
            "Блок 2 «Подготовка»: размер Train или Test меньше числа классов. "
            "Измените долю Test или используйте больше наблюдений."
        )

    features = list(p["features"])
    categorical = set(p.get("categorical", []))
    prepared = replace_markers(frame[features], p.get("missing_rules", {}))
    for column in features:
        if prepared[column].isna().all():
            raise ValueError(
                f"Блок 2 «Подготовка»: в признаке «{column}» после учёта "
                "маркеров пропусков нет ни одного значения."
            )
        if column not in categorical:
            try:
                pd.to_numeric(prepared[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Блок 2 «Подготовка»: столбец «{column}» содержит текст. "
                    "Отметьте его как категориальный или исправьте данные."
                ) from exc
    return n_classes
