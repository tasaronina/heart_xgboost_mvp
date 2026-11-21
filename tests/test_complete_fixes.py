"""Регрессии для полного пакета исправлений от 15.11.2026."""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest
from docx import Document
from sklearn.datasets import make_classification

from workflow_mvp.data_io import load_csv
from workflow_mvp.engine import logistic_solver_details, run_analysis
from workflow_mvp.persistence import get_run, recent_runs, save_summary
from workflow_mvp.report_docx import create_docx
from workflow_mvp.spec import METHODS, available_charts, default_settings, select_method
from workflow_mvp.state import initialize_dataset_settings, update_method, update_method_params
from workflow_mvp import visuals


ROOT = Path(__file__).resolve().parents[1]


def _frame(classes: int = 2, rows: int = 120) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=rows,
        n_features=6,
        n_informative=5,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=37 + classes,
    )
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    frame["target"] = y
    return frame


def _configured(frame: pd.DataFrame, method: str = "xgboost") -> dict:
    settings, warnings = initialize_dataset_settings(default_settings(), frame)
    assert warnings == []
    return select_method(settings, method)


def _info(frame: pd.DataFrame) -> dict:
    return {
        "dataset_name": "synthetic.csv", "sha256": "a" * 64,
        "rows": len(frame), "columns": len(frame.columns),
        "duplicates": 0, "missing_cells": 0,
    }


def _doc_text(payload: bytes) -> str:
    document = Document(BytesIO(payload))
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def test_demo_dataset_initializes_target_and_features_before_panel_is_open():
    raw = (ROOT / "sample_data" / "heart.csv").read_bytes()
    frame, _ = load_csv(raw, "heart.csv")
    settings, warnings = initialize_dataset_settings(default_settings(), frame)
    assert warnings == []
    assert settings["preprocess"]["target"] == "target"
    assert settings["preprocess"]["features"] == [c for c in frame if c != "target"]
    assert settings["preprocess"]["positive"] == "1"


def test_five_hundred_method_switches_preserve_each_method_and_other_blocks():
    frame = _frame()
    settings = _configured(frame)
    settings["study"]["name"] = "Устойчивый черновик"
    settings["preprocess"]["test_size"] = 0.3
    settings["output"]["charts"] = ["confusion", "shap_bar"]
    expected = {
        "xgboost": {"max_depth": 7},
        "random_forest": {"min_samples_leaf": 5},
        "logistic_regression": {"C": 0.35, "regularization": "elasticnet"},
    }
    for method, changes in expected.items():
        settings = update_method_params(settings, method, changes)
    methods = tuple(METHODS)
    for index in range(500):
        settings = update_method(settings, methods[index % len(methods)])
    for method, changes in expected.items():
        for key, value in changes.items():
            assert settings["analysis"]["params_by_method"][method][key] == value
    assert settings["study"]["name"] == "Устойчивый черновик"
    assert settings["preprocess"]["test_size"] == 0.3
    assert settings["output"]["charts"] == ["confusion", "shap_bar"]


@pytest.mark.parametrize("regularization,solver", [
    ("l2", "lbfgs"), ("l1", "saga"), ("elasticnet", "saga"),
])
def test_logistic_solver_selection_and_parameters(regularization: str, solver: str):
    params = deepcopy(METHODS["logistic_regression"]["parameters"])
    params.update({
        "regularization": regularization, "C": 0.7,
        "class_weight": "balanced", "l1_ratio": 0.25,
        "max_iter": 1200, "tol": 0.0003, "fit_intercept": False,
    })
    args, saved = logistic_solver_details(params, 4)
    assert args["solver"] == solver
    assert saved["solver"] == solver
    assert saved["class_weight"] == "balanced"
    assert saved["tol"] == 0.0003
    assert saved["fit_intercept"] is False
    assert "выбран автоматически" in saved["solver_reason"]


@pytest.mark.parametrize("regularization", ["l2", "l1", "elasticnet"])
@pytest.mark.parametrize("classes", [2, 3])
def test_logistic_regularizations_really_fit_binary_and_multiclass(
    regularization: str, classes: int,
):
    frame = _frame(classes)
    settings = _configured(frame, "logistic_regression")
    settings = update_method_params(settings, "logistic_regression", {
        "regularization": regularization,
        "l1_ratio": 0.35,
        "class_weight": "balanced",
        "max_iter": 1500,
        "tol": 0.0002,
        "fit_intercept": False,
    })
    artifact = run_analysis(frame, settings, _info(frame))
    params = artifact.summary["model_parameters"]
    assert params["regularization"] == regularization
    assert params["class_weight"] == "balanced"
    assert params["fit_intercept"] is False
    assert params["solver"] == ("lbfgs" if regularization == "l2" else "saga")


@pytest.mark.parametrize("classes", [2, 3, 4])
@pytest.mark.parametrize("method", list(METHODS))
def test_all_methods_run_for_two_to_four_classes(classes: int, method: str):
    frame = _frame(classes, 100 if classes < 4 else 120)
    settings = _configured(frame, method)
    if method == "logistic_regression":
        settings = update_method_params(settings, method, {
            "regularization": "elasticnet", "l1_ratio": 0.4,
            "class_weight": "balanced", "max_iter": 1200,
        })
    artifact = run_analysis(frame, settings, _info(frame))
    assert artifact.summary["n_classes"] == classes
    assert len(artifact.summary["metrics_per_class"]) == classes
    if method == "logistic_regression":
        assert artifact.summary["model_parameters"]["solver"] == "saga"


def test_chart_matrix_enables_both_shap_views_for_all_supported_tasks():
    for method in METHODS:
        settings = select_method(default_settings(), method)
        for classes in (2, 3, 4):
            matrix = available_charts(settings, classes)
            assert matrix["shap_beeswarm"][0]
            assert matrix["shap_bar"][0]
            assert matrix["roc"][0]
            assert matrix["pr"][0]


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("classes", [2, 3, 4])
def test_shap_beeswarm_and_bar_render_for_every_method(method: str, classes: int):
    frame = _frame(classes, 100 if classes < 4 else 120)
    settings = _configured(frame, method)
    settings["output"]["charts"] = ["shap_beeswarm", "shap_bar"]
    settings["output"]["shap_class"] = str(classes - 1)
    artifact = run_analysis(frame, settings, _info(frame))
    figures, warnings = visuals.render_selected(artifact)
    assert warnings == []
    assert set(figures) == {"shap_beeswarm", "shap_bar"}
    assert all(value.startswith(b"\x89PNG") for value in figures.values())
    assert artifact.summary["shap_metadata"]["explainer"] == METHODS[method]["shap_explainer"]


def test_two_shap_plots_share_one_calculation(monkeypatch):
    frame = _frame()
    settings = _configured(frame, "random_forest")
    settings["output"]["charts"] = ["shap_beeswarm", "shap_bar"]
    artifact = run_analysis(frame, settings, _info(frame))
    original = visuals.compute_shap_explanation
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(visuals, "compute_shap_explanation", counted)
    figures, warnings = visuals.render_selected(artifact)
    assert warnings == []
    assert len(figures) == 2
    assert calls == 1


def test_shap_is_not_computed_when_not_selected(monkeypatch):
    frame = _frame()
    settings = _configured(frame)
    settings["output"]["charts"] = ["confusion"]
    artifact = run_analysis(frame, settings, _info(frame))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("SHAP не должен вычисляться")

    monkeypatch.setattr(visuals, "compute_shap_explanation", forbidden)
    figures, warnings = visuals.render_selected(artifact)
    assert warnings == []
    assert set(figures) == {"confusion"}


def test_old_database_schema_migrates_idempotently_without_losing_rows(tmp_path):
    db = tmp_path / "old.db"
    summary = {
        "created_at": "2025-01-02T03:04:05", "dataset": {"dataset_name": "old.csv"},
        "model": "xgboost", "metrics": {"accuracy": 0.5, "f1": 0.4},
        "config": default_settings(),
    }
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE runs (id TEXT PRIMARY KEY, created TEXT, dataset TEXT, method TEXT, summary TEXT)"
        )
        connection.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?)",
            ("legacy", summary["created_at"], "old.csv", "xgboost", json.dumps(summary)),
        )
    assert recent_runs(db)[0]["id"] == "legacy"
    assert recent_runs(db)[0]["id"] == "legacy"
    assert get_run("legacy", db)["summary"]["dataset"]["dataset_name"] == "old.csv"


def test_saved_assets_and_report_reopen_without_training(tmp_path):
    frame = _frame()
    settings = _configured(frame, "logistic_regression")
    settings["study"]["name"] = "Проверка истории"
    settings["output"]["charts"] = ["confusion", "shap_bar"]
    artifact = run_analysis(frame, settings, _info(frame))
    figures, warnings = visuals.render_selected(artifact)
    report = create_docx(artifact.summary, figures, warnings)
    db = tmp_path / "history.db"
    run_id = save_summary(artifact.summary, figures, report, db)
    restored = get_run(run_id, db)
    assert restored is not None
    assert restored["figures"] == figures
    assert restored["report"] == report
    assert restored["summary"]["study_name"] == "Проверка истории"


def test_repeated_run_gets_new_id_and_restored_copy_is_immutable(tmp_path):
    frame = _frame()
    artifact = run_analysis(frame, _configured(frame), _info(frame))
    db = tmp_path / "separate_runs.db"
    first_id = save_summary(artifact.summary, db=db)
    second_id = save_summary(artifact.summary, db=db)
    assert first_id != second_id
    original = get_run(first_id, db)["summary"]
    working_copy = deepcopy(original["config"])
    working_copy["analysis"]["method"] = "random_forest"
    assert get_run(first_id, db)["summary"]["config"]["analysis"]["method"] == "xgboost"


def test_word_report_explains_solver_and_shap_scale():
    frame = _frame()
    settings = _configured(frame, "logistic_regression")
    settings["study"]["name"] = "Логистическое исследование"
    settings["output"]["charts"] = ["shap_beeswarm", "shap_bar"]
    artifact = run_analysis(frame, settings, _info(frame))
    figures, warnings = visuals.render_selected(artifact)
    payload = create_docx(artifact.summary, figures, warnings)
    text = _doc_text(payload)
    assert "Логистическое исследование" in text
    assert "Solver выбран автоматически" in text
    assert "LinearExplainer" in text
    assert "log-odds" in text
    assert "Интерпретируемый класс: 1" in text
    assert "не доказывают причинность" in text
