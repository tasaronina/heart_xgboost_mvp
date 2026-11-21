"""Независимое выполнение 4 блоков: data → preprocess → model → output."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import inspect
from math import ceil
from typing import Any
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import xgboost

from .data_io import replace_markers
from .spec import METHODS, normalize_settings, validate_settings


@dataclass
class RunArtifacts:
    """Только в оперативной памяти: обученная модель и подготовленные данные."""
    summary: dict[str, Any]
    model: Any
    feature_names: list[str]
    X_explain: np.ndarray
    X_background: np.ndarray
    descriptive_frame: pd.DataFrame
    labels: list[str]


def _classes_as_strings(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("В целевой переменной есть пропуски: их нельзя автоматически заполнять.")
    return series.astype(str)


def _prepare_X(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    X = X.copy()
    for column in X.columns:
        if column in categorical:
            X[column] = X[column].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        else:
            try:
                X[column] = pd.to_numeric(X[column], errors="raise")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Столбец «{column}» содержит текст. Укажите его как категориальный либо настройте пропуски."
                ) from e
        if X[column].isna().all():
            raise ValueError(f"В признаке «{column}» нет ни одного непустого значения.")
    return X


def logistic_solver_details(params: dict, n_classes: int) -> tuple[dict, dict]:
    """Возвращает совместимые аргументы sklearn и сохраняемые пояснения.

    В sklearn 1.9 ``penalty`` уже deprecated, поэтому тип регуляризации
    выражается через l1_ratio. Для поддерживаемых requirements старых версий
    используется прежний параметр penalty.
    """
    regularization = str(params.get("regularization", "l2"))
    if regularization not in ("l2", "l1", "elasticnet"):
        raise ValueError("Неизвестный тип регуляризации логистической регрессии.")
    solver = "lbfgs" if regularization == "l2" else "saga"
    if regularization == "l2":
        ratio = 0.0
    elif regularization == "l1":
        ratio = 1.0
    else:
        ratio = float(params.get("l1_ratio", 0.5))
    if not 0 <= ratio <= 1:
        raise ValueError("l1_ratio должен находиться от 0 до 1.")

    reason = (
        f"Решатель {solver} выбран автоматически: он совместим с регуляризацией "
        f"{regularization.upper() if regularization != 'elasticnet' else 'Elastic Net'} "
        f"и задачей с {n_classes} классами в scikit-learn {sklearn.__version__}."
    )
    class_weight = params.get("class_weight")
    if class_weight not in (None, "balanced"):
        raise ValueError("class_weight должен быть None или balanced.")
    args = {
        "C": float(params.get("C", 1.0)),
        "class_weight": class_weight,
        "tol": float(params.get("tol", 0.0001)),
        "fit_intercept": bool(params.get("fit_intercept", True)),
        "max_iter": int(params.get("max_iter", 800)),
        "solver": solver,
    }
    signature = inspect.signature(LogisticRegression)
    penalty_default = signature.parameters["penalty"].default
    if penalty_default == "deprecated":
        args["l1_ratio"] = ratio
    else:
        args["penalty"] = regularization
        if regularization == "elasticnet":
            args["l1_ratio"] = ratio
    saved = {
        "regularization": regularization,
        "C": args["C"],
        "class_weight": class_weight,
        "l1_ratio": ratio,
        "max_iter": args["max_iter"],
        "tol": args["tol"],
        "fit_intercept": args["fit_intercept"],
        "solver": solver,
        "solver_reason": reason,
    }
    return args, saved


def _model_for(settings: dict, n_classes: int):
    name = settings["analysis"]["method"]
    cfg = {
        **METHODS[name]["parameters"],
        **settings["analysis"].get("params_by_method", {}).get(name, {}),
        **settings["analysis"].get("params", {}),
    }
    seed = int(settings["preprocess"].get("random_state", 42))
    if name == "xgboost":
        obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
        args = dict(objective=obj, eval_metric="logloss" if n_classes == 2 else "mlogloss",
                    tree_method="hist", random_state=seed, n_jobs=2,
                    importance_type="gain", n_estimators=int(cfg["n_estimators"]),
                    max_depth=int(cfg["max_depth"]), learning_rate=float(cfg["learning_rate"]),
                    subsample=float(cfg["subsample"]), colsample_bytree=float(cfg["colsample_bytree"]))
        if n_classes > 2:
            args["num_class"] = n_classes
        model = XGBClassifier(**args)
    elif name == "random_forest":
        model = RandomForestClassifier(n_estimators=int(cfg["n_estimators"]),
                    max_depth=int(cfg["max_depth"]), min_samples_leaf=int(cfg["min_samples_leaf"]),
                    random_state=seed, n_jobs=2)
    else:
        args, cfg = logistic_solver_details(cfg, n_classes)
        args["random_state"] = seed
        model = LogisticRegression(**args)
    return model, cfg


def run_analysis(frame: pd.DataFrame, settings: dict, dataset_info: dict) -> RunArtifacts:
    """Fit after stratified split; all fitted transformations learn on Train only."""
    settings = normalize_settings(settings)
    p = settings["preprocess"]
    target = p.get("target")
    if target not in frame.columns:
        raise ValueError("Сначала выберите целевой столбец Y.")
    data = frame.drop_duplicates().copy() if p.get("remove_duplicates") else frame.copy()
    y_raw = _classes_as_strings(data[target])
    classes = sorted(pd.unique(y_raw).tolist())
    n_classes = len(classes)
    validate_settings(settings, list(data.columns), n_classes)
    if n_classes == 2:
        pos = str(p.get("positive", ""))
        if pos not in classes:
            raise ValueError("Выберите действительный положительный класс.")
        classes = [x for x in classes if x != pos] + [pos]
    y = y_raw.map({v: i for i, v in enumerate(classes)}).astype(int)
    counts = y.value_counts()
    if counts.min() < 2:
        raise ValueError("Для Train/Test требуется не менее двух наблюдений каждого класса.")
    test_size = float(p.get("test_size", .2))
    if min(ceil(len(data)*test_size), len(data)-ceil(len(data)*test_size)) < n_classes:
        raise ValueError("Размер Train или Test меньше числа классов. Увеличьте выборку или измените долю Test.")

    columns = list(p["features"])
    cats = list(p.get("categorical", []))
    X = replace_markers(data[columns], p.get("missing_rules", {}))
    X = _prepare_X(X, cats)
    desc_columns = [c for c in p.get("descriptive_columns", []) if c in X.columns]
    if not desc_columns:
        desc_columns = [c for c in columns if c not in cats][:8]
    # Это обзор полной таблицы для описания данных, НЕ статистики для обучения.
    desc_frame = X[[c for c in desc_columns if c not in cats]].select_dtypes(include="number").copy() \
        if p.get("descriptive", False) else pd.DataFrame()
    stats = []
    for col in desc_frame.columns:
        series = desc_frame[col]
        stats.append({"Признак": col, "N": int(series.count()), "Пропуски": int(series.isna().sum()),
                      "Среднее": float(series.mean()) if series.count() else None,
                      "Медиана": float(series.median()) if series.count() else None,
                      "SD": float(series.std()) if series.count() > 1 else None,
                      "Мин": float(series.min()) if series.count() else None,
                      "Макс": float(series.max()) if series.count() else None})

    numeric = [c for c in columns if c not in cats]
    scaling = p.get("scaling", "auto")
    if scaling == "auto":
        scaling = METHODS[settings["analysis"]["method"]]["scaling"]
    numerical_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard":
        numerical_steps.append(("scale", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numerical_steps), numeric))
    if cats:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cats))
    processor = ColumnTransformer(transformers, sparse_threshold=0)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(p.get("random_state", 42)), stratify=y
        )
    except ValueError as e:
        raise ValueError(f"Ошибка стратифицированного разделения: {e}") from e
    for c in columns:
        if X_train[c].isna().all():
            raise ValueError(f"В Train признак «{c}» содержит только пропуски.")
    model, model_params = _model_for(settings, n_classes)
    try:
        Xtr = np.asarray(processor.fit_transform(X_train), dtype=float)
        Xte = np.asarray(processor.transform(X_test), dtype=float)
        if Xtr.shape[1] > 1000:
            raise ValueError("После One-Hot получилось более 1000 столбцов. Проверьте выбор категорий.")
        model.fit(Xtr, y_train)
        prediction = model.predict(Xte)
        probabilities = np.asarray(model.predict_proba(Xte), dtype=float)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Ошибка обучения модели: {e}") from e

    labels = list(range(n_classes))
    cm = confusion_matrix(y_test, prediction, labels=labels)
    precision, recall, f1, supports = precision_recall_fscore_support(
        y_test, prediction, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "precision_weighted": float(np.average(precision, weights=supports)),
        "recall_weighted": float(np.average(recall, weights=supports)),
        "f1_weighted": float(np.average(f1, weights=supports)),
    }
    if n_classes == 2:
        metrics.update(precision=float(precision[1]), recall=float(recall[1]), f1=float(f1[1]),
                       specificity=float(cm[0, 0]/max(cm[0].sum(), 1)))
        metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities[:, 1]))
    else:
        metrics["roc_auc_ovr_macro"] = float(
            roc_auc_score(y_test, probabilities, labels=labels, multi_class="ovr", average="macro")
        )
    by_class = []
    for i, name in enumerate(classes):
        tp = int(cm[i, i]); fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        by_class.append({"class": str(name), "precision": float(precision[i]),
                         "recall": float(recall[i]), "f1": float(f1[i]),
                         "support": int(supports[i]),
                         "specificity_ovr": float(tn / (tn + fp)) if tn + fp else 0.0})
    names = [str(x).replace("num__", "").replace("cat__", "")
             for x in processor.get_feature_names_out().tolist()]
    if settings["analysis"]["method"] == "logistic_regression":
        coefficients = np.asarray(model.coef_)
        weights = np.mean(np.abs(coefficients), axis=0)
        importance_kind = "abs_coefficient"
    else:
        weights = np.asarray(model.feature_importances_, dtype=float)
        importance_kind = "gain" if settings["analysis"]["method"] == "xgboost" else "impurity"
    importance = sorted([
        {"feature": n, "value": float(w)} for n, w in zip(names, weights)
    ], key=lambda row: row["value"], reverse=True)
    created_at = datetime.now(timezone.utc)
    study_name = str(settings.get("study", {}).get("name", "")).strip()
    if not study_name:
        study_name = (
            f"Анализ {dataset_info.get('dataset_name', 'данных')} — "
            f"{METHODS[settings['analysis']['method']]['title']} ({created_at.date().isoformat()})"
        )
        settings["study"]["name"] = study_name
    summary = {
        "created_at": created_at.isoformat(),
        "study_name": study_name,
        "dataset": dict(dataset_info),
        "pipeline": ["input", "preprocess", "analysis", "output"],
        "config": deepcopy(settings),
        "model": settings["analysis"]["method"],
        "model_parameters": model_params,
        "scaling_used": scaling,
        "n_classes": n_classes,
        "classes": classes,
        "rows_after_duplicates": int(len(data)),
        "duplicates_removed": int(len(frame) - len(data)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "class_distribution": {classes[i]: int((y == i).sum()) for i in labels},
        "metrics": metrics, "metrics_per_class": by_class,
        "confusion_matrix": cm.tolist(),
        "importance": importance[:50],
        "importance_kind": importance_kind,
        "descriptive": stats,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "shap": package_version("shap")},
        "methodology": (
            "Стратифицированное Train/Test. Imputer/One-Hot/StandardScaler при выборе "
            "подгоняются исключительно на Train. Описательная статистика рассчитана по "
            "указанным столбцам исходной таблицы и не используется при fit. "
            "Важность признаков и SHAP не доказывают причинность."
        ),
    }
    # Для визуализаций нужны прогнозы/оценки, но исходные записи не сохраняются в отчёте/БД.
    summary["_viz"] = {"y_true": y_test.tolist(), "y_pred": np.asarray(prediction).astype(int).tolist(),
                       "probabilities": probabilities.tolist()}
    return RunArtifacts(summary=summary, model=model, feature_names=names,
                        X_explain=Xte, X_background=Xtr,
                        descriptive_frame=desc_frame, labels=classes)
