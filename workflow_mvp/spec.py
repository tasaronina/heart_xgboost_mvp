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
        "description": "Градиентный бустинг деревьев.",
        "scaling": "none",
        "parameters": {
            "n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        },
        "importance_title": "Значимость признаков: gain",
        "shap_explainer": "TreeExplainer",
    },
    "random_forest": {
        "title": "Random Forest",
        "description": "Ансамбль случайных деревьев.",
        "scaling": "none",
        "parameters": {"n_estimators": 150, "max_depth": 6, "min_samples_leaf": 2},
        "importance_title": "Значимость признаков: impurity",
        "shap_explainer": "TreeExplainer",
    },
    "logistic_regression": {
        "title": "Логистическая регрессия",
        "description": "Линейный классификатор с настраиваемой регуляризацией.",
        "scaling": "standard",
        "parameters": {
            "regularization": "l2", "C": 1.0, "class_weight": None,
            "l1_ratio": 0.5, "max_iter": 800, "tol": 0.0001,
            "fit_intercept": True,
        },
        "importance_title": "Модуль коэффициентов модели",
        "shap_explainer": "LinearExplainer",
    },
}

# ``shap`` оставлен для старых сохранённых запусков. В новом конструкторе
# пользователь выбирает Beeswarm и Bar отдельно.
CHARTS = {
    "confusion": "Матрица ошибок",
    "roc": "ROC-кривая",
    "pr": "PR-кривая",
    "importance": "Значимость признаков / коэффициенты",
    "shap_beeswarm": "SHAP Beeswarm",
    "shap_bar": "SHAP Bar",
    "shap": "SHAP Beeswarm",
    "hist": "Гистограммы",
    "box": "Boxplot",
    "corr": "Корреляционная карта",
}
SELECTABLE_CHARTS = (
    "confusion", "roc", "pr", "importance", "shap_beeswarm", "shap_bar",
    "hist", "box", "corr",
)

CHART_HELP = {
    "confusion": (
        "На диагонали — верные классы, за её пределами — ошибки. В бинарном "
        "случае FP — ложные тревоги, FN — пропущенные положительные случаи."
    ),
    "roc": (
        "Чувствительность (TPR) против доли ложноположительных прогнозов (FPR) "
        "при разных порогах. В многоклассовом случае используется One-vs-Rest."
    ),
    "pr": "Precision и Recall при разных порогах; особенно полезна при анализе редкого класса.",
    "importance": (
        "XGBoost показывает gain, Random Forest — impurity importance, "
        "логистическая регрессия — модуль коэффициента. Эти показатели не доказывают причинность."
    ),
    "shap_beeswarm": (
        "Каждая точка — наблюдение; положение по оси показывает SHAP-вклад в "
        "выход модели, цвет — значение признака. Beeswarm выбран для обзора общих "
        "закономерностей по множеству наблюдений; Waterfall для одного прогноза в MVP не включён."
    ),
    "shap_bar": (
        "Ранжирует признаки по средней абсолютной величине SHAP-вкладов; "
        "показывает силу вкладов, но не их направление."
    ),
    "shap": "Каждая точка — наблюдение; положение по оси показывает SHAP-вклад в выход модели.",
    "hist": "Показывает распределение значений выбранных числовых признаков.",
    "box": (
        "Показывает медиану, квартильный разброс и отдельные необычные значения; "
        "это не автоматическое доказательство ошибки измерения."
    ),
    "corr": (
        "Показывает попарные статистические связи между числовыми признаками; "
        "корреляция не означает причинность."
    ),
}


def default_settings() -> dict:
    params_by_method = {method: deepcopy(meta["parameters"]) for method, meta in METHODS.items()}
    return {
        "study": {"name": ""},
        "editor": {"orientation": "horizontal", "library_collapsed": False},
        "input": {"source": "Демонстрационный heart.csv"},
        "preprocess": {
            "target": "", "positive": "", "features": [], "categorical": [],
            "test_size": 0.2, "random_state": 42, "scaling": "auto",
            "remove_duplicates": False, "missing_rules": {},
            "descriptive": True, "descriptive_columns": [],
        },
        "analysis": {
            "method": "xgboost",
            "params": deepcopy(params_by_method["xgboost"]),
            "params_by_method": params_by_method,
        },
        "output": {
            "charts": ["confusion", "roc", "importance"],
            "chart_preferences": ["confusion", "roc", "importance"],
            "shap_class": "",
        },
    }


def normalize_settings(settings: dict | None) -> dict:
    """Дополняет старую конфигурацию новыми полями без потери известных значений."""
    incoming = settings if isinstance(settings, dict) else {}
    result = default_settings()
    for section in ("study", "editor", "input", "preprocess", "output"):
        values = incoming.get(section, {})
        if isinstance(values, dict):
            result[section].update(values)

    old_analysis = incoming.get("analysis", {})
    if not isinstance(old_analysis, dict):
        old_analysis = {}
    method = old_analysis.get("method", "xgboost")
    if method not in METHODS:
        method = "xgboost"
    result["analysis"]["method"] = method
    provided_by_method = old_analysis.get("params_by_method", {})
    if not isinstance(provided_by_method, dict):
        provided_by_method = {}
    for method_name, meta in METHODS.items():
        merged = deepcopy(meta["parameters"])
        candidate = provided_by_method.get(method_name, {})
        if isinstance(candidate, dict):
            merged.update({k: v for k, v in candidate.items() if k in merged})
        result["analysis"]["params_by_method"][method_name] = merged
    legacy_params = old_analysis.get("params", {})
    if isinstance(legacy_params, dict):
        active = result["analysis"]["params_by_method"][method]
        active.update({k: v for k, v in legacy_params.items() if k in active})
    result["analysis"]["params"] = deepcopy(result["analysis"]["params_by_method"][method])

    if result["editor"].get("orientation") not in ("horizontal", "vertical"):
        result["editor"]["orientation"] = "horizontal"
    charts = result["output"].get("charts", [])
    if not isinstance(charts, list):
        charts = []
    charts = ["shap_beeswarm" if key == "shap" else key for key in charts]
    result["output"]["charts"] = list(dict.fromkeys(k for k in charts if k in SELECTABLE_CHARTS))
    prefs = result["output"].get("chart_preferences", result["output"]["charts"])
    if not isinstance(prefs, list):
        prefs = result["output"]["charts"]
    result["output"]["chart_preferences"] = list(dict.fromkeys(
        "shap_beeswarm" if key == "shap" else key for key in prefs if key in CHARTS
    ))
    return result


def select_method(settings: dict, method: str) -> dict:
    """Переключает метод, сохраняя отдельные параметры каждого алгоритма."""
    normalized = normalize_settings(settings)
    if method not in METHODS:
        raise ValueError("Выбран неизвестный аналитический метод.")
    normalized["analysis"]["method"] = method
    normalized["analysis"]["params"] = deepcopy(
        normalized["analysis"]["params_by_method"][method]
    )
    return normalized


def commit_active_params(settings: dict, params: dict) -> None:
    """Записывает параметры активного метода в оба совместимых представления."""
    method = settings["analysis"]["method"]
    settings["analysis"]["params_by_method"][method] = deepcopy(params)
    settings["analysis"]["params"] = deepcopy(params)


def available_charts(settings: dict, n_classes: int) -> dict[str, tuple[bool, str]]:
    """Матрица доступности по типу задачи и реально реализованным графикам."""
    settings = normalize_settings(settings)
    descriptive = bool(settings["preprocess"].get("descriptive"))
    status: dict[str, tuple[bool, str]] = {}
    for key in CHARTS:
        if key in ("hist", "box", "corr") and not descriptive:
            status[key] = (False, "Включите описательную статистику в блоке «Подготовка данных».")
        elif key in ("roc", "pr") and not 2 <= n_classes <= 4:
            status[key] = (False, "Нужна классификация с 2–4 классами.")
        elif key in ("shap", "shap_beeswarm", "shap_bar") and not 2 <= n_classes <= 4:
            status[key] = (False, "SHAP в этой версии рассчитан для классификации с 2–4 классами.")
        else:
            status[key] = (True, "")
    return status


def validate_edges(node_ids: list[str], edge_pairs: list[tuple[str, str]]) -> None:
    if len(node_ids) != 4 or set(node_ids) != set(BLOCKS):
        raise ValueError("Для этого MVP нужны четыре разных блока: Вход → Подготовка → Анализ → Вывод.")
    if len(edge_pairs) != 3 or set(edge_pairs) != set(EXPECTED_EDGES):
        raise ValueError("Соедините блоки именно так: Вход → Подготовка → Анализ → Вывод. Лишние линии и циклы не допускаются.")


def validate_settings(settings: dict, columns: list[str], n_classes: int) -> None:
    normalized = normalize_settings(settings)
    settings.clear()
    settings.update(normalized)
    preprocess = settings["preprocess"]
    method = settings["analysis"]["method"]
    if not 2 <= n_classes <= 4:
        raise ValueError(f"В целевой переменной {n_classes} классов; требуется 2–4.")
    target = preprocess["target"]
    features = preprocess["features"]
    if not target or target not in columns:
        raise ValueError("Выберите целевую переменную в блоке подготовки.")
    if not features or len(set(features)) != len(features) or target in features or any(f not in columns for f in features):
        raise ValueError("Проверьте список X: минимум один столбец, без цели Y и неизвестных имён.")
    if not set(preprocess.get("categorical", [])).issubset(features):
        raise ValueError("Категориальные признаки должны входить в X.")
    if preprocess.get("scaling", "auto") not in ("auto", "none", "standard"):
        raise ValueError("Неподдерживаемый способ масштабирования.")
    if not 0.1 <= float(preprocess.get("test_size", 0.2)) <= 0.4:
        raise ValueError("Доля Test должна быть от 0.1 до 0.4.")
    for column, tokens in preprocess.get("missing_rules", {}).items():
        if column not in columns or not isinstance(tokens, list) or len(tokens) > 20:
            raise ValueError("Некорректное правило пропусков.")

    params = settings["analysis"]["params_by_method"][method]
    if method == "logistic_regression":
        regularization = params.get("regularization", "l2")
        if regularization not in ("l2", "l1", "elasticnet"):
            raise ValueError("Выберите L2, L1 или Elastic Net для логистической регрессии.")
        if float(params.get("C", 0)) <= 0:
            raise ValueError("Параметр C должен быть положительным.")
        if regularization == "elasticnet" and not 0 <= float(params.get("l1_ratio", -1)) <= 1:
            raise ValueError("l1_ratio должен находиться от 0 до 1.")
        if float(params.get("tol", 0)) <= 0 or int(params.get("max_iter", 0)) <= 0:
            raise ValueError("tol и max_iter должны быть положительными.")

    statuses = available_charts(settings, n_classes)
    for chart in settings["output"].get("charts", []):
        if chart not in CHARTS:
            raise ValueError(f"Неизвестный график: {chart}")
        ok, why = statuses[chart]
        if not ok:
            raise ValueError(f"Недоступен график «{CHARTS[chart]}»: {why}")


def validate_preflight(settings: dict, frame: pd.DataFrame) -> int:
    """Полная предпусковая проверка без разделения данных и обучения модели."""
    from .data_io import replace_markers

    preprocess = settings["preprocess"]
    target = preprocess.get("target", "")
    if not target or target not in frame.columns:
        raise ValueError(
            "Блок 2 «Подготовка данных»: целевой столбец Y не выбран или отсутствует. "
            "Откройте блок и выберите столбец результата."
        )
    if frame[target].isna().any():
        raise ValueError(
            "Блок 2 «Подготовка данных»: в целевом столбце Y есть пропуски. "
            "Выберите другой столбец или исправьте исходный CSV."
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
            "Блок 2 «Подготовка данных»: для Train/Test требуется не менее двух "
            "наблюдений каждого класса. Добавьте наблюдения или выберите другую Y."
        )
    test_size = float(preprocess.get("test_size", 0.2))
    test_rows = ceil(len(frame) * test_size)
    if min(test_rows, len(frame) - test_rows) < n_classes:
        raise ValueError(
            "Блок 2 «Подготовка данных»: размер Train или Test меньше числа классов. "
            "Измените долю Test или используйте больше наблюдений."
        )

    features = list(preprocess["features"])
    categorical = set(preprocess.get("categorical", []))
    prepared = replace_markers(frame[features], preprocess.get("missing_rules", {}))
    for column in features:
        if prepared[column].isna().all():
            raise ValueError(
                f"Блок 2 «Подготовка данных»: в признаке «{column}» после учёта "
                "маркеров пропусков нет ни одного значения. Измените правило пропусков."
            )
        if column not in categorical:
            try:
                pd.to_numeric(prepared[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Блок 2 «Подготовка данных»: столбец «{column}» содержит текст. "
                    "Отметьте его как категориальный или исправьте данные."
                ) from exc
    return n_classes
