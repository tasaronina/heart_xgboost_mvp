from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from docx import Document

from workflow_mvp.engine import run_analysis
from workflow_mvp.persistence import get_run, save_summary
from workflow_mvp.report_docx import create_docx
from workflow_mvp.spec import default_settings
from workflow_mvp.visuals import descriptive_feature_groups, render_selected


def test_descriptive_feature_groups_keep_every_feature():
    columns = [f"feature_{index}" for index in range(9)]
    groups = descriptive_feature_groups(columns)
    assert [len(group) for group in groups] == [4, 4, 1]
    assert [name for group in groups for name in group] == columns


def test_all_descriptive_plots_are_saved_and_added_to_docx(tmp_path):
    rng = np.random.default_rng(21)
    features = [f"feature_{index}" for index in range(8)]
    frame = pd.DataFrame(rng.normal(size=(120, 8)), columns=features)
    frame["target"] = [index % 2 for index in range(len(frame))]
    settings = default_settings()
    settings["preprocess"].update(
        target="target",
        features=features,
        positive="1",
        descriptive=True,
        descriptive_columns=features,
    )
    settings["analysis"]["method"] = "random_forest"
    settings["output"]["charts"] = ["hist", "box"]
    info = {
        "dataset_name": "eight_features.csv",
        "sha256": "1" * 64,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicates": 0,
        "missing_cells": 0,
    }

    artifact = run_analysis(frame, settings, info)
    figures, warnings = render_selected(artifact)

    assert warnings == []
    assert list(figures) == ["hist", "hist__2", "box", "box__2"]
    groups = artifact.summary["descriptive_visualization"]["groups"]
    assert [name for key in ("hist", "hist__2") for name in groups[key]] == features

    report = create_docx(artifact.summary, figures)
    document = Document(BytesIO(report))
    assert len(document.inline_shapes) == 4

    db = tmp_path / "runs.db"
    run_id = save_summary(artifact.summary, figures, report, db)
    restored = get_run(run_id, db)
    assert restored is not None
    assert restored["figures"].keys() == figures.keys()
    assert restored["summary"]["descriptive_visualization"]["groups"] == groups
