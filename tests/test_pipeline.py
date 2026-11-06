import json
from pathlib import Path

import pandas as pd
import pytest

from services.data_validation import UserDataError, read_csv_dataset
from services.report_service import report_json
from services.storage import save_run, list_runs
from services.xgboost_service import RunConfig, run_classification

PATH = Path(__file__).resolve().parents[1] / "sample_data" / "heart.csv"


def make_config(**changes):
    options = dict(
        target_column="target",
        feature_columns=["age", "sex", "chest_pain_type", "resting_bp", "max_hr", "thal"],
        categorical_columns=["sex", "chest_pain_type", "thal"],
        positive_label=1, test_size=0.2, n_estimators=25,
        max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42,
    )
    options.update(changes)
    return RunConfig(**options)


def test_read_dataset():
    loaded = read_csv_dataset(PATH.read_bytes(), "heart.csv")
    assert loaded.frame.shape == (303, 14)
    assert loaded.frame.isna().sum().sum() == 0
    assert loaded.frame.duplicated().sum() == 1


def test_end_to_end_binary_classification():
    result = run_classification(pd.read_csv(PATH), make_config())
    assert result["sample"]["rows_after_duplicate_removal"] == 302
    assert result["sample"]["duplicate_rows_removed"] == 1
    assert sum(sum(row) for row in result["confusion_matrix"]) == result["sample"]["test_rows"]
    assert 0 <= result["metrics"]["accuracy"] <= 1
    assert 0 <= result["metrics"]["roc_auc"] <= 1
    assert result["sample"]["positive_test_rows"] > 0
    assert result["feature_importance"]
    assert len(result["predictions"]) == result["sample"]["test_rows"]
    assert json.loads(report_json(result, "heart.csv", "dummy"))["result"]["config"]["max_depth"] == 3


def test_reproducibility_with_seed():
    frame = pd.read_csv(PATH)
    first = run_classification(frame, make_config())
    second = run_classification(frame, make_config())
    assert first["predictions"] == second["predictions"]
    assert first["metrics"] == second["metrics"]


def test_target_cannot_be_a_feature():
    with pytest.raises(UserDataError, match="утечка"):
        run_classification(pd.read_csv(PATH), make_config(feature_columns=["age", "target"]))


def test_rejects_nonbinary_target():
    frame = pd.read_csv(PATH)
    frame.loc[0,"target"] = 4
    with pytest.raises(UserDataError, match="ровно два класса"):
        run_classification(frame, make_config())


def test_rejects_missing_target():
    frame = pd.read_csv(PATH)
    frame.loc[0,"target"] = float("nan")
    with pytest.raises(UserDataError, match="пропуски"):
        run_classification(frame, make_config())


def test_missing_features_are_imputed_safely():
    frame = pd.read_csv(PATH)
    frame.loc[:9,"age"] = float("nan")
    frame.loc[30:34,"thal"] = float("nan")
    result = run_classification(frame, make_config())
    assert result["metrics"]["f1"] >= 0


def test_history_saved_and_loaded(tmp_path):
    frame = pd.read_csv(PATH)
    result = run_classification(frame, make_config())
    db = tmp_path / "history.db"
    uid = save_run(result,"heart.csv","abc",db)
    rows = list_runs(db)
    assert len(rows) == 1
    assert rows[0]["run_id"] == uid
    assert rows[0]["result"]["metrics"] == result["metrics"]
