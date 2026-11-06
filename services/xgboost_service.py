"""Вычислительное ядро бинарной классификации XGBoost, не зависящее от Streamlit."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .data_validation import UserDataError, validate_classification_input


@dataclass(frozen=True)
class RunConfig:
    target_column: str
    feature_columns: list[str]
    categorical_columns: list[str]
    positive_label: Any = 1
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    remove_duplicates: bool = False
    duplicate_action: str = "keep"  # keep, review или remove
    # Значения, обозначающие отсутствие данных: для каждого столбца своё правило.
    # Пример: {"Glucose": ["0"], "Temperature": ["-999"]}.
    missing_values: dict[str, list[str]] = field(default_factory=dict)


def _safe(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _replace_custom_missing(
    X: pd.DataFrame,
    rules: dict[str, list[str]]
) -> pd.DataFrame:
    """Помечает заданные значения как NaN; не рассчитывает статистик по данным."""
    X = X.copy()
    for column, tokens in rules.items():
        if column not in X.columns:
            continue  # Правило для неиспользуемого в этом запуске признака.

        series = X[column]
        mask = pd.Series(False, index=X.index)
        for raw in tokens:
            token = str(raw).strip()
            if not token:
                continue  # Пустые ячейки уже распознаются pandas как NaN.
            if pd.api.types.is_numeric_dtype(series):
                try:
                    candidate = float(token)
                except ValueError:
                    continue
                if np.isfinite(candidate):
                    mask |= series.eq(candidate).fillna(False)
            else:
                mask |= series.astype("string").str.strip().eq(token).fillna(False)

        if mask.any():
            X[column] = series.mask(mask, np.nan)
    return X


def run_classification(
    frame: pd.DataFrame,
    config: RunConfig
) -> dict[str, Any]:
    clean = validate_classification_input(
        frame,
        config.target_column,
        config.feature_columns,
        config.categorical_columns,
        config.positive_label,
        config.test_size,
        config.remove_duplicates,
    )

    if not (1 <= config.n_estimators <= 10000 and 1 <= config.max_depth <= 30):
        raise UserDataError("Некорректное количество деревьев или глубина модели.")

    if not (
        0 < config.learning_rate <= 1
        and 0 < config.subsample <= 1
        and 0 < config.colsample_bytree <= 1
    ):
        raise UserDataError("Некорректные параметры XGBoost.")

    y = (clean[config.target_column] == config.positive_label).astype(int)
    X = _replace_custom_missing(
        clean[config.feature_columns],
        config.missing_values
    )

    for col in config.feature_columns:
        if X[col].isna().all():
            raise UserDataError(
                f"В столбце «{col}» после применения правил нет непустых значений."
            )

    numeric = [
        c for c in config.feature_columns
        if c not in config.categorical_columns
    ]
    categorical = list(config.categorical_columns)

    for col in numeric:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors="raise")
            except (TypeError, ValueError) as exc:
                raise UserDataError(
                    f"Столбец «{col}» содержит текстовые значения. "
                    "Уточните специальные обозначения пропусков или отметьте его "
                    "как категориальный признак."
                ) from exc

    for col in categorical:
        # Текстовое представление кодов, при этом пропуски сохраняются.
        X[col] = X[col].map(
            lambda v: str(v) if pd.notna(v) else np.nan
        ).astype(object)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            stratify=y,
            random_state=config.random_state,
        )
    except ValueError as exc:
        raise UserDataError(f"Невозможно разделить данные: {exc}") from exc

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise UserDataError("Один из классов отсутствует в обучении или тесте.")

    for col in config.feature_columns:
        if X_train[col].isna().all():
            raise UserDataError(
                f"В обучающей выборке у признака «{col}» остались только пропуски. "
                "Проверьте правила или измените разбиение."
            )

    transformers = []

    if numeric:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median"),
            numeric
        ))

    if categorical:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipeline, categorical))

    preprocessor = ColumnTransformer(transformers=transformers)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=config.random_state,
        n_jobs=2,
        importance_type="gain",
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    try:
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test).astype(int)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    except (ValueError, TypeError, xgboost.core.XGBoostError) as exc:
        raise UserDataError(f"Модель не удалось обучить: {exc}") from exc

    cm = confusion_matrix(y_test, predicted, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=1)

    try:
        transformed_names = preprocessor.get_feature_names_out().tolist()
        importances = model.feature_importances_.tolist()

        importance = sorted(
            (
                {
                    "feature": str(name)
                    .replace("numeric__", "")
                    .replace("categorical__", ""),
                    "importance": float(score),
                }
                for name, score in zip(transformed_names, importances)
            ),
            key=lambda item: item["importance"],
            reverse=True
        )
    except (AttributeError, ValueError):
        importance = []

    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(
            precision_score(y_test, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predicted, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "positive_label": _safe(config.positive_label)
        },
        "sample": {
            "rows_original": int(len(frame)),
            "rows_after_duplicate_removal": int(len(clean)),
            "duplicate_rows_removed": int(len(frame) - len(clean)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "positive_rows": int(y.sum()),
            "negative_rows": int(len(clean) - y.sum()),
            "positive_test_rows": int(y_test.sum()),
            "negative_test_rows": int(len(y_test) - y_test.sum()),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        },
        "feature_importance": importance,
        "predictions": [
            {
                "row_index": _safe(index),
                "actual": int(actual),
                "predicted": int(pred),
                "positive_probability": float(prob),
            }
            for index, actual, pred, prob in zip(
                X_test.index.tolist(),
                y_test.tolist(),
                predicted,
                probabilities
            )
        ],
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "methodology": (
            "Указанные пользователем специальные значения отмечены как пропуски. "
            "Train/Test разделены со стратификацией. Медианы и наиболее частые категории "
            "вычислены только на обучающих данных. На тестовой выборке рассчитаны метрики. "
            "Положительный класс закодирован как 1, отрицательный как 0."
        ),
    }
