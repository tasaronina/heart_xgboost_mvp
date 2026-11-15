"""Графики результатов. Возвращаем PNG-байты для Streamlit и DOCX."""
from __future__ import annotations

from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from .engine import RunArtifacts
from .spec import CHARTS, available_charts

plt.rcParams["font.family"] = "DejaVu Sans"
SURFACE = "#F3F6FA"
INK = "#26364A"
GRID = "#C9D3DF"
ACCENT = "#2D8F87"


def _style_figure(fig) -> None:
    """Единая читаемая светлая поверхность графиков без изменения данных."""
    fig.patch.set_facecolor(SURFACE)
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK, labelsize=9)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)
        ax.title.set_color(INK)
        for spine in ax.spines.values():
            spine.set_color("#7D8DA3")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(SURFACE)
            legend.get_frame().set_edgecolor(GRID)
            for text in legend.get_texts():
                text.set_color(INK)


def _png(fig) -> bytes:
    buf = BytesIO()
    _style_figure(fig)
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return buf.getvalue()


def _beeswarm(artifact: RunArtifacts, class_name: str = "") -> bytes:
    import shap  # дорогую библиотеку импортируем ТОЛЬКО при выбранном SHAP

    X = artifact.X_explain[:min(120, len(artifact.X_explain))]
    if len(X) < 2:
        raise ValueError("Недостаточно наблюдений для SHAP Beeswarm.")
    e = shap.TreeExplainer(artifact.model, model_output="raw")
    explanation = e(X)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        i = artifact.labels.index(class_name) if class_name in artifact.labels else 0
        values = values[:, :, i]
    if values.ndim != 2:
        raise ValueError(f"SHAP вернул неподдерживаемую форму {values.shape}.")
    exp = shap.Explanation(values=values, data=X, feature_names=artifact.feature_names)
    plt.figure(figsize=(7.2, 4.8), facecolor=SURFACE)
    shap.plots.beeswarm(exp, max_display=min(12, len(artifact.feature_names)), show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm · raw output (не вероятность)", fontsize=11)
    return _png(fig)


def create_chart(artifact: RunArtifacts, chart: str, shap_class: str = "") -> bytes:
    summary = artifact.summary
    classes = summary["classes"]
    n = len(classes)
    v = summary["_viz"]
    y = np.asarray(v["y_true"], dtype=int)
    p = np.asarray(v["probabilities"], dtype=float)
    if chart == "shap":
        return _beeswarm(artifact, shap_class)
    if chart == "confusion":
        cm = np.asarray(summary["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(max(5.4, n * 1.25), max(4.2, n * 1.0)), facecolor=SURFACE)
        image = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n), classes, rotation=20)
        ax.set_yticks(range(n), classes)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title("Матрица ошибок")
        fig.colorbar(image, ax=ax, fraction=0.045)
        return _png(fig)
    if chart in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
        for i, label in enumerate(classes):
            if n == 2 and i == 0:
                continue
            binary = (y == i).astype(int)
            if chart == "roc":
                xs, ys, _ = roc_curve(binary, p[:, i])
                desc = f"{label}: AUC {auc(xs, ys):.3f}"
                ax.plot(xs, ys, label=desc, color=ACCENT if n == 2 else None, linewidth=2)
            else:
                precision, recall, _ = precision_recall_curve(binary, p[:, i])
                ap = average_precision_score(binary, p[:, i])
                ax.plot(recall, precision, label=f"{label}: AP {ap:.3f}",
                        color=ACCENT if n == 2 else None, linewidth=2)
        if chart == "roc":
            ax.plot([0, 1], [0, 1], "--", color="grey", alpha=.6)
            ax.set(xlabel="Доля ложноположительных (FPR)", ylabel="Чувствительность (TPR)",
                   title="ROC-кривая" if n == 2 else "ROC-кривые OvR: каждый класс против остальных")
        else:
            ax.set(xlabel="Полнота (Recall)", ylabel="Точность (Precision)",
                   title="PR-кривая" if n == 2 else "PR-кривые OvR: каждый класс против остальных")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=.15)
        return _png(fig)
    if chart == "importance":
        data = [row for row in summary["importance"] if row["value"] > 0][:12]
        if not data:
            raise ValueError("Ненулевые оценки значимости отсутствуют.")
        data.reverse()
        fig, ax = plt.subplots(figsize=(7.2, max(3.2, len(data)*.38)), facecolor=SURFACE)
        ax.barh([r["feature"] for r in data], [r["value"] for r in data], color=ACCENT)
        suffix = {"gain": "XGBoost gain", "impurity": "Random Forest impurity",
                  "abs_coefficient": "|коэффициент| логистической регрессии"}
        ax.set(xlabel=suffix.get(summary["importance_kind"], "Значение"),
               title="Значимость признаков (не причинность)")
        return _png(fig)
    data = artifact.descriptive_frame
    if data.empty:
        raise ValueError("Описательная статистика не включена или нет числовых столбцов.")
    if chart == "corr":
        if len(data.columns) < 2:
            raise ValueError("Для корреляционной карты нужны хотя бы два числовых признака.")
        corr = data.corr(numeric_only=True).fillna(0)
        fig, ax = plt.subplots(figsize=(max(5.2, len(corr.columns)*.65),
                                        max(4.2, len(corr.columns)*.58)), facecolor=SURFACE)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(im, ax=ax)
        ax.set_title("Корреляция Пирсона — описательно")
        return _png(fig)
    shown = list(data.columns)[:4]
    ncols = 1 if len(shown) == 1 else 2
    nrows = int(np.ceil(len(shown) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, max(3.4, nrows * 3.0)),
                             facecolor=SURFACE, squeeze=False)
    flat_axes = list(axes.flat)
    for ax, col in zip(flat_axes, shown):
        vals = data[col].dropna()
        if chart == "hist":
            ax.hist(vals, bins=min(25, max(5, int(np.sqrt(len(vals))))),
                    color=ACCENT, edgecolor=SURFACE)
            ax.set_title(f"Распределение: {col}")
            ax.set_ylabel("Количество")
        elif chart == "box":
            ax.boxplot(vals, orientation="horizontal", patch_artist=True,
                       boxprops={"facecolor": "#A9DCD6", "edgecolor": ACCENT},
                       medianprops={"color": "#1E4D55", "linewidth": 2})
            ax.set_title(f"Boxplot: {col}")
        ax.set_xlabel(str(col))
    for ax in flat_axes[len(shown):]:
        ax.remove()
    return _png(fig)


def render_selected(artifact: RunArtifacts) -> tuple[dict[str, bytes], list[str]]:
    """Графики вычисляются только по выбору, включая ленивый SHAP."""
    s = artifact.summary["config"]
    statuses = available_charts(s, artifact.summary["n_classes"])
    figures, warnings = {}, []
    for chart in s["output"].get("charts", []):
        ok, msg = statuses.get(chart, (False, "Неизвестный график"))
        if not ok:
            warnings.append(f"{CHARTS.get(chart,chart)}: {msg}")
            continue
        try:
            figures[chart] = create_chart(artifact, chart, s["output"].get("shap_class", ""))
        except Exception as exc:
            warnings.append(f"Не удалось построить «{CHARTS[chart]}»: {exc}")
    return figures, warnings
