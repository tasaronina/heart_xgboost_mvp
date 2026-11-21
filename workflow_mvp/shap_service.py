"""Ленивый SHAP-сервис: один расчёт на запуск для Beeswarm и Bar."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import RunArtifacts


@dataclass
class ShapExplanation:
    explanation: object
    class_name: str
    class_index: int
    explainer_name: str
    output_scale: str
    explained_rows: int
    background_rows: int

    def metadata(self) -> dict:
        return {
            "class": self.class_name,
            "class_index": self.class_index,
            "explainer": self.explainer_name,
            "output_scale": self.output_scale,
            "explained_rows": self.explained_rows,
            "background_rows": self.background_rows,
            "causality_note": "SHAP-вклады описывают модель и не доказывают причинность.",
        }


def _class_index(artifact: RunArtifacts, requested: str) -> int:
    if len(artifact.labels) == 2:
        # В engine положительный класс всегда переставляется на позицию 1.
        return 1
    return artifact.labels.index(requested) if requested in artifact.labels else 0


def compute_shap_explanation(
    artifact: RunArtifacts,
    requested_class: str = "",
    max_rows: int = 120,
    max_background: int = 80,
) -> ShapExplanation:
    """Вычисляет общую Explanation без сохранения построчных значений в БД."""
    import shap

    X = np.asarray(artifact.X_explain[:min(max_rows, len(artifact.X_explain))], dtype=float)
    background = np.asarray(
        artifact.X_background[:min(max_background, len(artifact.X_background))], dtype=float
    )
    if len(X) < 2 or len(background) < 2:
        raise ValueError("Недостаточно наблюдений Train/Test для расчёта SHAP.")

    method = artifact.summary["model"]
    if method in ("xgboost", "random_forest"):
        model_output = "raw" if method == "xgboost" else "probability"
        explainer = shap.TreeExplainer(
            artifact.model,
            data=background,
            feature_perturbation="interventional",
            model_output=model_output,
        )
        explainer_name = "TreeExplainer"
        output_scale = (
            "raw margin (для двух классов — log-odds положительного класса)"
            if method == "xgboost"
            else "вероятность выбранного класса"
        )
    else:
        explainer = shap.LinearExplainer(artifact.model, background)
        explainer_name = "LinearExplainer"
        output_scale = "линейный выход модели (для двух классов — log-odds положительного класса)"

    raw = explainer(X)
    values = np.asarray(raw.values)
    index = _class_index(artifact, requested_class)
    class_name = artifact.labels[index]
    if values.ndim == 3:
        if values.shape[2] != len(artifact.labels):
            raise ValueError(
                f"SHAP вернул форму {values.shape}, не согласованную с {len(artifact.labels)} классами."
            )
        values = values[:, :, index]
    elif values.ndim == 2:
        if len(artifact.labels) > 2:
            raise ValueError(
                f"Для многоклассовой модели SHAP вернул неподдерживаемую форму {values.shape}."
            )
        class_name = artifact.labels[1]
        index = 1
    else:
        raise ValueError(f"SHAP вернул неподдерживаемую форму {values.shape}.")
    if values.shape != X.shape:
        raise ValueError(
            f"Размер SHAP {values.shape} не совпадает с преобразованными признаками {X.shape}."
        )
    explanation = shap.Explanation(values=values, data=X, feature_names=artifact.feature_names)
    return ShapExplanation(
        explanation=explanation,
        class_name=str(class_name),
        class_index=index,
        explainer_name=explainer_name,
        output_scale=output_scale,
        explained_rows=len(X),
        background_rows=len(background),
    )
