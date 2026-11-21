"""Настоящий Word-отчёт DOCX: операции, метрики, выбранные графики."""
from __future__ import annotations
from io import BytesIO
from docx import Document
from docx.shared import Cm, Pt
from .spec import BLOCK_TITLES, CHARTS, CHART_HELP, METHODS


def _table(doc, title: str, rows: list[tuple[str, str]]):
    doc.add_heading(title, 2)
    t = doc.add_table(rows=1, cols=2)
    t.style = "Light Shading Accent 1"
    t.rows[0].cells[0].text = "Показатель"
    t.rows[0].cells[1].text = "Значение"
    for k, v in rows:
        cells = t.add_row().cells
        cells[0].text = str(k)
        cells[1].text = str(v)


def _figure_keys(base: str, figures: dict[str, bytes]) -> list[str]:
    keys = [key for key in figures if key.split("__", 1)[0] == base]
    return sorted(
        keys,
        key=lambda key: 1 if "__" not in key else int(key.rsplit("__", 1)[1]),
    )


def create_docx(summary: dict, figures: dict[str, bytes], warnings: list[str] | None = None) -> bytes:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10)
    doc.add_heading("Отчёт по аналитическому исследованию", 0)
    doc.add_heading(summary.get("study_name", "Исследование без названия"), 1)
    doc.add_paragraph("Экспериментальный прототип. Не для принятия клинических решений.")
    d = summary["dataset"]
    _table(doc, "1. Исходные данные", [("Набор", d["dataset_name"]),
           ("SHA-256", d["sha256"]), ("Исходных строк", str(d["rows"])),
           ("Столбцов", str(d["columns"])),
           ("Целевая переменная", summary["config"]["preprocess"]["target"]),
           ("Классы", ", ".join(summary["classes"]))])
    p = summary["config"]["preprocess"]
    _table(doc, "2. Схема и подготовка", [
        ("Последовательность", " → ".join(BLOCK_TITLES[b] for b in summary["pipeline"])),
        ("Признаки X", ", ".join(p["features"])),
        ("Категориальные признаки", ", ".join(p.get("categorical", [])) or "Нет"),
        ("Правила пропусков", str(p.get("missing_rules", {}))),
        ("Удаление дубликатов", "Да" if p.get("remove_duplicates") else "Нет"),
        ("Удалено строк", str(summary["duplicates_removed"])),
        ("Train/Test", f"{summary['train_rows']} / {summary['test_rows']}"),
        ("test_size", str(p["test_size"])),
        ("random_state", str(p["random_state"])),
        ("Фактическое масштабирование", summary["scaling_used"]),
        ("Описательная статистика", "Да" if p.get("descriptive") else "Нет"),
    ])
    model_rows = [("Метод", METHODS[summary["model"]]["title"]),
                  *[(str(k), str(v)) for k, v in summary["model_parameters"].items()]]
    _table(doc, "3. Аналитический метод", model_rows)
    if summary["model"] == "logistic_regression":
        params = summary["model_parameters"]
        doc.add_paragraph(
            "Solver выбран автоматически для совместимости с регуляризацией: "
            f"{params.get('solver', '—')}. {params.get('solver_reason', '')}"
        )
    _table(doc, "4. Метрики", [(k, f"{v:.4f}") for k, v in summary["metrics"].items()])
    doc.add_heading("По каждому классу", 2)
    t = doc.add_table(rows=1, cols=6)
    t.style = "Light Shading Accent 1"
    for c, name in zip(t.rows[0].cells, ["Класс", "Precision", "Recall", "F1", "N", "Specificity OvR"]):
        c.text = name
    for info in summary["metrics_per_class"]:
        for c, v in zip(t.add_row().cells, [info["class"], f"{info['precision']:.3f}",
                   f"{info['recall']:.3f}", f"{info['f1']:.3f}", str(info["support"]),
                   f"{info['specificity_ovr']:.3f}"]):
            c.text = v
    if summary.get("descriptive"):
        doc.add_heading("5. Описательная статистика", 1)
        def fmt(value):
            return f"{value:.3f}" if isinstance(value, (int, float)) and value is not None else str(value)
        for item in summary["descriptive"]:
            doc.add_paragraph(f"{item['Признак']}: N={item['N']}; пропуски={item['Пропуски']}; "
                              f"среднее={fmt(item['Среднее'])}; медиана={fmt(item['Медиана'])}; "
                              f"SD={fmt(item['SD'])}; мин={fmt(item['Мин'])}; макс={fmt(item['Макс'])}",
                              style="List Bullet")
    doc.add_heading("6. Выбранные визуализации", 1)
    groups = summary.get("descriptive_visualization", {}).get("groups", {})
    for base_key in summary["config"]["output"].get("charts", []):
        for key in _figure_keys(base_key, figures):
            title = CHARTS[base_key]
            columns = groups.get(key, [])
            if columns:
                title += " · " + ", ".join(columns)
            doc.add_heading(title, 2)
            doc.add_paragraph(CHART_HELP.get(base_key, ""))
            if base_key in ("shap", "shap_beeswarm", "shap_bar") and summary.get("shap_metadata"):
                meta = summary["shap_metadata"]
                doc.add_paragraph(
                    f"Интерпретируемый класс: {meta.get('class', meta.get('class_name', '—'))}; "
                    f"алгоритм объяснения: {meta.get('explainer', '—')}; "
                    f"шкала выхода: {meta.get('output_scale', '—')}. "
                    "SHAP-вклады описывают поведение модели и не доказывают причинность."
                )
            doc.add_picture(BytesIO(figures[key]), width=Cm(15.5))
    skipped = summary.get("descriptive_visualization", {}).get("skipped", [])
    if skipped:
        doc.add_paragraph(
            "Не включены в описательные графики как неподходящие или нечисловые: "
            + ", ".join(skipped)
        )
    if warnings:
        doc.add_heading("Предупреждения при визуализации", 2)
        for w in warnings:
            doc.add_paragraph(w, style="List Bullet")
    doc.add_heading("7. Воспроизводимость и ограничения", 1)
    doc.add_paragraph(summary["methodology"])
    doc.add_paragraph("Версии: " + ", ".join(f"{k}={v}" for k,v in summary["versions"].items()))
    doc.add_paragraph("Запуск (UTC): " + summary["created_at"])
    doc.add_paragraph("Исходная таблица и обученная модель в документ не включены.")
    buffer = BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
