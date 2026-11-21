"""Smoke & regression tests that don't import Streamlit UI."""
from __future__ import annotations
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, load_iris

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_mvp.engine import run_analysis
from workflow_mvp.spec import (METHODS, available_charts, default_settings,
                               validate_edges, validate_preflight)
from workflow_mvp.visuals import render_selected
from workflow_mvp.report_docx import create_docx
from workflow_mvp.persistence import get_run, recent_runs, safe_summary, save_summary


def _settings(df, target, method="xgboost", positive="1"):
    s = default_settings()
    s["preprocess"].update(target=target, features=[c for c in df if c != target], positive=positive,
                           descriptive_columns=[c for c in df.columns if c != target][:4])
    s["analysis"].update(method=method, params=deepcopy(METHODS[method]["parameters"]))
    return s


def _info(df):
    return {"dataset_name": "test.csv", "sha256": "0" * 64,
            "rows": len(df), "columns": len(df.columns), "duplicates": 0, "missing_cells": 0}


def test_valid_graph():
    validate_edges(["output", "input", "analysis", "preprocess"],
                   [("input", "preprocess"), ("preprocess", "analysis"), ("analysis", "output")])
    with pytest.raises(ValueError):
        validate_edges(["input", "preprocess", "analysis", "output"],
                       [("input", "analysis")])


def test_multiclass_and_binary_methods():
    iris = load_iris(as_frame=True).frame.rename(columns={"target":"diagnosis"})
    iris["diagnosis"] = iris["diagnosis"].map({0:"mild",1:"moderate",2:"severe"})
    for method in METHODS:
        s = _settings(iris, "diagnosis", method)
        a = run_analysis(iris, s, _info(iris))
        assert a.summary["n_classes"] == 3
        assert len(a.summary["confusion_matrix"]) == 3
        assert len(a.summary["metrics_per_class"]) == 3
        assert a.summary["metrics"]["roc_auc_ovr_macro"] >= .5
        assert a.summary["train_rows"] + a.summary["test_rows"] == len(iris)

    xy, binary = make_classification(n_samples=160, n_features=6, n_redundant=0,
                                     random_state=1)
    heart = pd.DataFrame(xy, columns=[f"b{i}" for i in range(6)])
    heart["target"] = binary
    a = run_analysis(heart, _settings(heart, "target"), _info(heart))
    assert a.summary["n_classes"] == 2
    assert 0 <= a.summary["metrics"]["roc_auc"] <= 1
    assert a.summary["classes"][-1] == "1"


def test_four_classes_and_different_scaling():
    X, y = make_classification(n_samples=200, n_features=7, n_informative=6,
                                n_redundant=0, n_classes=4, n_clusters_per_class=1,
                                random_state=21)
    df = pd.DataFrame(X, columns=[f"v_{i}" for i in range(7)])
    df["diagnosis"] = [f"grade_{v}" for v in y]
    for method in METHODS:
        settings = _settings(df, "diagnosis", method)
        a = run_analysis(df, settings, _info(df))
        assert a.summary["n_classes"] == 4
        assert len(a.summary["metrics_per_class"]) == 4
        assert a.summary["scaling_used"] == ("standard" if method == "logistic_regression" else "none")


def test_missing_category_and_report():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100),
                       "group": ["A" if i % 3 else "B" for i in range(100)],
                       "Y": [i % 2 for i in range(100)]})
    df.loc[12:19, "a"] = 0.0
    df.loc[11:17, "group"] = None
    s = _settings(df, "Y", "random_forest")
    s["preprocess"]["categorical"] = ["group"]
    s["preprocess"]["missing_rules"] = {"a": ["0"]}
    s["output"]["charts"] = ["confusion", "importance", "hist", "box", "corr"]
    a = run_analysis(df, s, _info(df))
    figures, errors = render_selected(a)
    assert not errors, errors
    assert all(content.startswith(b"\x89PNG") for content in figures.values())
    docx = create_docx(a.summary, figures)
    assert docx[:2] == b"PK"
    from docx import Document
    d = Document(BytesIO(docx))
    assert any("Отчёт по аналитическому" in p.text for p in d.paragraphs)
    assert "_viz" not in safe_summary(a.summary)


def test_chart_restrictions():
    s = default_settings()
    s["analysis"]["method"] = "random_forest"
    assert available_charts(s, 3)["shap_beeswarm"][0] is True
    assert available_charts(s, 3)["shap_bar"][0] is True
    assert available_charts(s, 3)["roc"][0] is True
    s["preprocess"]["descriptive"] = False
    assert available_charts(s, 3)["hist"][0] is False


def test_preflight_checks_settings_without_training():
    frame = pd.DataFrame({
        "age": [31, 42, 55, 62, 38, 49, 57, 69, 44, 51],
        "group": ["A", "B"] * 5,
        "target": [0, 1] * 5,
    })
    settings = _settings(frame, "target")
    settings["preprocess"]["categorical"] = ["group"]
    assert validate_preflight(settings, frame) == 2

    broken = deepcopy(settings)
    broken["preprocess"]["target"] = ""
    with pytest.raises(ValueError, match="целевой столбец"):
        validate_preflight(broken, frame)


def test_history_keeps_summary_figures_and_report(tmp_path):
    db = tmp_path / "workflow_runs.db"
    summary = {
        "created_at": "2026-11-07T12:00:00",
        "dataset": {"dataset_name": "test.csv"},
        "model": "xgboost",
        "metrics": {"accuracy": 0.8, "f1": 0.75},
        "config": default_settings(),
    }
    png = b"\x89PNG\r\n\x1a\nexample"
    report = b"PK\x03\x04example"
    run_id = save_summary(summary, {"confusion": png}, report, db)

    rows = recent_runs(db)
    restored = get_run(run_id, db)
    assert rows == [{
        "id": run_id,
        "created": summary["created_at"],
        "dataset": "test.csv",
        "method": "xgboost",
        "study_name": "Анализ test.csv — XGBoost (2026-11-07 12:00)",
        "n_classes": None,
        "figure_count": 1,
        "has_report": True,
        "accuracy": 0.8,
        "f1": 0.75,
    }]
    assert restored is not None
    assert restored["summary"] == summary
    assert restored["figures"]["confusion"] == png
    assert restored["report"] == report
