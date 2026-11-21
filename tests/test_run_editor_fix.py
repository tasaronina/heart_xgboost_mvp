"""UI-state regressions for the focused run/editor repair of 16.11.2026."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest
from streamlit_flow.state import StreamlitFlowState

from workflow_mvp.spec import default_settings


APP = Path(__file__).resolve().parents[1] / "app.py"


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    return app


def _run_buttons(app: AppTest) -> dict[str, bool]:
    labels = {"Проверить готовность к запуску", "Выполнить анализ"}
    return {button.label: button.disabled for button in app.button if button.label in labels}


def test_interrupted_running_flag_recovers_instead_of_disabling_forever():
    app = _app()
    app.session_state["wf_is_running"] = True
    app.session_state["wf_run_requested"] = False
    app.run()
    assert app.session_state["wf_is_running"] is False
    assert _run_buttons(app) == {
        "Проверить готовность к запуску": False,
        "Выполнить анализ": False,
    }
    assert "была прервана" in app.session_state["wf_run_error"]


def test_ready_check_reports_structure_y_and_x_together():
    app = _app()
    flow = app.session_state["wf_flow"]
    app.session_state["wf_flow"] = StreamlitFlowState(list(flow.nodes), list(flow.edges)[:2])
    app.session_state["wf_cfg"]["preprocess"]["target"] = ""
    app.session_state["wf_cfg"]["preprocess"]["features"] = []
    app.run()
    next(button for button in app.button if button.label == "Проверить готовность к запуску").click().run()
    message = app.session_state["wf_validation"]["message"]
    assert "Схема:" in message
    assert "целевой столбец Y" in message
    assert "хотя бы один признак X" in message
    assert not any(_run_buttons(app).values())


def test_frontend_local_drawer_and_auto_connect_preserve_graph():
    app = _app()
    node_ids = [node.id for node in app.session_state["wf_flow"].nodes]
    edge_pairs = [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges]
    # The drawer is now local React state: Python reruns must not expose or
    # depend on the former Streamlit toggle button, and the canonical graph
    # must remain stable. Actual drawer toggling is covered in browser tests.
    assert not any(button.key == "toggle_block_library" for button in app.button)
    for _ in range(10):
        app.run()
    assert [node.id for node in app.session_state["wf_flow"].nodes] == node_ids
    assert [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges] == edge_pairs

    flow = app.session_state["wf_flow"]
    app.session_state["wf_flow"] = StreamlitFlowState(list(flow.nodes), [])
    app.run()
    next(button for button in app.button if button.label == "Соединить по порядку").click().run()
    restored = app.session_state["wf_flow"].edges
    assert [(edge.source, edge.target) for edge in restored] == edge_pairs
    assert all(edge.style["strokeWidth"] == 3 for edge in restored)


DEMO_SOURCE = "Демонстрационный heart.csv"
CUSTOM_SOURCE = "Загрузить свой CSV"


def _widget(items, label):
    return next(item for item in items if item.label == label)


def _add_input_block(app: AppTest, input_node) -> AppTest:
    app.session_state["wf_flow"] = StreamlitFlowState([input_node], [])
    app.session_state["wf_active_block"] = "input"
    app.session_state["wf_graph_revision"] += 1
    return app.run()


def test_first_custom_csv_requires_confirmation_and_clears_only_after_confirm():
    app = _app()
    assert not any(button.label == "Новое исследование с собственным CSV" for button in app.button)
    input_node = app.session_state["wf_flow"].nodes[0]
    original_nodes = [node.id for node in app.session_state["wf_flow"].nodes]
    raw = b"feature,target\n1,0\n2,1\n3,0\n4,1\n"
    _widget(app.text_input, "Название исследования").set_value("Текущий черновик").run()
    _widget(app.radio, "Источник данных").set_value(CUSTOM_SOURCE).run()
    _widget(app.file_uploader, "CSV-файл").set_value(
        ("first.csv", raw, "text/csv")
    ).run()

    assert app.session_state["wf_cfg"]["input"]["source"] == DEMO_SOURCE
    assert app.session_state["wf_csv_bytes"] is None
    assert [node.id for node in app.session_state["wf_flow"].nodes] == original_nodes
    assert app.session_state["wf_study_name_input"] == "Текущий черновик"

    _widget(app.button, "Подтвердить смену").click().run()
    assert app.session_state["wf_cfg"]["input"]["source"] == CUSTOM_SOURCE
    assert app.session_state["wf_csv_name"] == "first.csv"
    assert app.session_state["wf_csv_bytes"] == raw
    assert app.session_state["wf_flow"].nodes == []
    assert app.session_state["wf_flow"].edges == []
    assert app.session_state["wf_study_name_input"] == ""
    assert app.session_state["wf_cfg"]["analysis"]["method"] == "xgboost"
    assert app.session_state["wf_cfg"]["preprocess"]["target"] == "target"

    _add_input_block(app, input_node)
    assert _widget(app.file_uploader, "CSV-файл").value is None


def test_replacement_and_cancel_leave_current_custom_draft_unchanged():
    app = _app()
    input_node = app.session_state["wf_flow"].nodes[0]
    old_raw = b"feature,target\n1,0\n2,1\n"
    new_raw = b"feature,target\n10,0\n20,1\n"
    app.session_state["wf_cfg"]["input"]["source"] = CUSTOM_SOURCE
    app.session_state["wf_csv_bytes"] = old_raw
    app.session_state["wf_csv_name"] = "old.csv"
    app.session_state["wf_file_id"] = None
    app.session_state["wf_input_control_revision"] += 1
    app.session_state["wf_study_name_input"] = "Не менять"
    app.run()

    _widget(app.file_uploader, "CSV-файл").clear().run()
    assert app.session_state["wf_dataset_change_pending"] is None
    assert app.session_state["wf_csv_bytes"] == old_raw

    _widget(app.file_uploader, "CSV-файл").set_value(
        ("new.csv", new_raw, "text/csv")
    ).run()
    assert app.session_state["wf_csv_bytes"] == old_raw
    assert app.session_state["wf_csv_name"] == "old.csv"
    assert [node.id for node in app.session_state["wf_flow"].nodes] == [
        "input", "preprocess", "analysis", "output"
    ]

    _widget(app.button, "Отмена").click().run()
    assert app.session_state["wf_dataset_change_pending"] is None
    assert app.session_state["wf_csv_bytes"] == old_raw
    assert app.session_state["wf_csv_name"] == "old.csv"
    assert app.session_state["wf_study_name_input"] == "Не менять"
    assert _widget(app.file_uploader, "CSV-файл").value is None

    _widget(app.file_uploader, "CSV-файл").set_value(
        ("new.csv", new_raw, "text/csv")
    ).run()
    _widget(app.button, "Подтвердить смену").click().run()
    assert app.session_state["wf_csv_bytes"] == new_raw
    assert app.session_state["wf_csv_name"] == "new.csv"
    assert app.session_state["wf_flow"].nodes == []

    _add_input_block(app, input_node)
    assert _widget(app.file_uploader, "CSV-файл").value is None


def test_switch_to_demo_is_cancelable_and_confirmed_switch_starts_clean_draft():
    app = _app()
    old_raw = b"feature,target\n1,0\n2,1\n"
    app.session_state["wf_cfg"]["input"]["source"] = CUSTOM_SOURCE
    app.session_state["wf_csv_bytes"] = old_raw
    app.session_state["wf_csv_name"] = "old.csv"
    app.session_state["wf_file_id"] = None
    app.session_state["wf_input_control_revision"] += 1
    app.run()

    _widget(app.radio, "Источник данных").set_value(DEMO_SOURCE).run()
    assert app.session_state["wf_cfg"]["input"]["source"] == CUSTOM_SOURCE
    assert app.session_state["wf_csv_bytes"] == old_raw
    _widget(app.button, "Отмена").click().run()
    assert app.session_state["wf_cfg"]["input"]["source"] == CUSTOM_SOURCE
    assert _widget(app.radio, "Источник данных").value == CUSTOM_SOURCE

    _widget(app.radio, "Источник данных").set_value(DEMO_SOURCE).run()
    _widget(app.button, "Подтвердить смену").click().run()
    assert app.session_state["wf_cfg"]["input"]["source"] == DEMO_SOURCE
    assert app.session_state["wf_csv_bytes"] is None
    assert app.session_state["wf_csv_name"] is None
    assert app.session_state["wf_flow"].nodes == []


def test_csv_attaches_to_restored_configuration_without_clearing_it():
    app = _app()
    raw = (APP.parent / "sample_data" / "heart.csv").read_bytes()
    original_nodes = [node.id for node in app.session_state["wf_flow"].nodes]
    original_edges = [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges]
    app.session_state["wf_cfg"]["input"]["source"] = CUSTOM_SOURCE
    app.session_state["wf_cfg"]["study"]["name"] = "Восстановленная конфигурация"
    app.session_state["wf_study_name_input"] = "Восстановленная конфигурация"
    app.session_state["wf_csv_bytes"] = None
    app.session_state["wf_csv_name"] = None
    app.session_state["wf_file_id"] = None
    app.session_state["wf_preserve_settings"] = True
    app.session_state["wf_input_control_revision"] += 1
    app.run()

    _widget(app.file_uploader, "CSV-файл").set_value(
        ("heart-restored.csv", raw, "text/csv")
    ).run()

    assert [node.id for node in app.session_state["wf_flow"].nodes] == original_nodes
    assert [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges] == original_edges
    assert app.session_state["wf_study_name_input"] == "Восстановленная конфигурация"
    assert app.session_state["wf_cfg"]["input"]["source"] == CUSTOM_SOURCE
    assert app.session_state["wf_csv_name"] == "heart-restored.csv"
    assert app.session_state["wf_preserve_settings"] is False


def test_history_restore_then_csv_upload_keeps_restored_graph(monkeypatch):
    import workflow_mvp.persistence as persistence

    raw = (APP.parent / "sample_data" / "heart.csv").read_bytes()
    settings = default_settings()
    settings["input"]["source"] = CUSTOM_SOURCE
    settings["study"]["name"] = "Из истории"
    scheme = {
        "schema_version": 1,
        "orientation": "horizontal",
        "node_positions": {
            "input": {"x": 0, "y": 110},
            "preprocess": {"x": 250, "y": 110},
            "analysis": {"x": 500, "y": 110},
            "output": {"x": 750, "y": 110},
        },
        "edges": [
            {"source": "input", "target": "preprocess"},
            {"source": "preprocess", "target": "analysis"},
            {"source": "analysis", "target": "output"},
        ],
        "settings": deepcopy(settings),
    }
    summary = {
        "study_name": "Из истории",
        "pipeline": ["input", "preprocess", "analysis", "output"],
        "pipeline_scheme": scheme,
        "config": deepcopy(settings),
        "dataset": {
            "dataset_name": "history-source.csv",
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    saved = {
        "id": "history-test",
        "summary": summary,
        "figures": {},
        "report": None,
    }
    row = {
        "id": "history-test",
        "created": "2026-11-21T12:00:00+00:00",
        "dataset": "history-source.csv",
        "method": "xgboost",
        "study_name": "Из истории",
        "n_classes": 2,
        "figure_count": 0,
        "has_report": False,
        "accuracy": 0.8,
        "f1": 0.75,
    }
    monkeypatch.setattr(persistence, "recent_runs", lambda: [row])
    monkeypatch.setattr(persistence, "get_run", lambda run_id: saved)

    app = _app()
    app.session_state["wf_view"] = "История"
    app.run()
    _widget(app.button, "Восстановить настройки").click().run()
    _widget(app.button, "Создать рабочую копию").click().run()

    restored_nodes = [node.id for node in app.session_state["wf_flow"].nodes]
    restored_edges = [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges]
    assert app.session_state["wf_cfg"]["input"]["source"] == CUSTOM_SOURCE
    assert app.session_state["wf_preserve_settings"] is True

    _widget(app.file_uploader, "CSV-файл").set_value(
        ("history-source.csv", raw, "text/csv")
    ).run()
    assert [node.id for node in app.session_state["wf_flow"].nodes] == restored_nodes
    assert [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges] == restored_edges
    assert app.session_state["wf_study_name_input"] == "Из истории"
    assert app.session_state["wf_dataset_change_pending"] is None


def test_json_restore_then_csv_upload_keeps_restored_graph():
    raw = (APP.parent / "sample_data" / "heart.csv").read_bytes()
    settings = default_settings()
    settings["input"]["source"] = CUSTOM_SOURCE
    settings["study"]["name"] = "Из JSON"
    scheme = {
        "schema_version": 1,
        "orientation": "vertical",
        "node_positions": {
            "input": {"x": 220, "y": 0},
            "preprocess": {"x": 220, "y": 145},
            "analysis": {"x": 220, "y": 290},
            "output": {"x": 220, "y": 435},
        },
        "edges": [
            {"source": "input", "target": "preprocess"},
            {"source": "preprocess", "target": "analysis"},
            {"source": "analysis", "target": "output"},
        ],
        "settings": settings,
    }
    payload = json.dumps(scheme, ensure_ascii=False).encode("utf-8")

    app = _app()
    _widget(app.file_uploader, "Файл схемы").set_value(
        ("scheme.json", payload, "application/json")
    ).run()
    _widget(app.button, "Восстановить").click().run()
    restored_nodes = [node.id for node in app.session_state["wf_flow"].nodes]
    restored_edges = [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges]

    _widget(app.file_uploader, "CSV-файл").set_value(
        ("json-source.csv", raw, "text/csv")
    ).run()
    assert [node.id for node in app.session_state["wf_flow"].nodes] == restored_nodes
    assert [(edge.source, edge.target) for edge in app.session_state["wf_flow"].edges] == restored_edges
    assert app.session_state["wf_study_name_input"] == "Из JSON"
    assert app.session_state["wf_dataset_change_pending"] is None
