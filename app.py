"""Единый визуальный конструктор дипломного MVP."""
from __future__ import annotations

from copy import deepcopy
import base64
import hashlib
import html
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState

from workflow_mvp.data_io import load_csv
from workflow_mvp.engine import logistic_solver_details, run_analysis
from workflow_mvp.editor_component import graph_editor
from workflow_mvp.editor_logic import apply_graph_event, GraphEditError
from workflow_mvp.persistence import get_run, recent_runs, save_summary
from workflow_mvp.report_docx import create_docx
from workflow_mvp.spec import (BLOCKS, BLOCK_TITLES, CHARTS, CHART_HELP,
                               EXPECTED_EDGES, METHODS, SELECTABLE_CHARTS,
                               available_charts, commit_active_params,
                               default_settings, normalize_settings, select_method,
                               validate_edges, validate_preflight)
from workflow_mvp.state import initialize_dataset_settings
from workflow_mvp.visuals import figure_base_key, render_selected

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "sample_data" / "heart.csv"
DEMO_SOURCE = "Демонстрационный heart.csv"
CUSTOM_SOURCE = "Загрузить свой CSV"
LOGGER = logging.getLogger("workflow_mvp")
st.set_page_config(page_title="Диплом — конструктор аналитических пайплайнов", layout="wide")
METRIC_HELP = {
    "accuracy": "Accuracy = (TP + TN) / (TP + TN + FP + FN). Доля всех правильных ответов на Test.",
    "precision": "Precision = TP / (TP + FP). Среди положительных прогнозов — доля действительно положительных.",
    "recall": "Recall = TP / (TP + FN). Среди реальных положительных случаев — доля обнаруженных моделью.",
    "f1": "F1 = 2 × Precision × Recall / (Precision + Recall). Баланс Precision и Recall положительного класса.",
    "specificity": "Specificity = TN / (TN + FP). Среди реальных отрицательных — доля распознанных правильно.",
    "roc_auc": "Площадь под ROC-кривой: разделение классов по оценкам модели при разных порогах. Это не доля правильных ответов.",
    "precision_macro": "Среднее Precision по классам, где каждый класс имеет одинаковый вес.",
    "recall_macro": "Среднее Recall по классам, где каждый класс имеет одинаковый вес.",
    "f1_macro": "Среднее F1 по классам, где каждый класс имеет одинаковый вес.",
    "roc_auc_ovr_macro": "Среднее ROC AUC для схемы «каждый класс против остальных» с равным весом классов.",
}

BLOCK_DESCRIPTIONS = {
    "input": "CSV, предпросмотр и сведения о наборе.",
    "preprocess": "X/Y, пропуски, категории и Train/Test.",
    "analysis": "Метод классификации и его параметры.",
    "output": "Метрики, графики и Word-отчёт.",
}


def apply_interface_style(view: str) -> None:
    """Светлая рабочая поверхность и фиксированная навигация без влияния на расчёты."""
    max_width = 1480 if view == "Конструктор" else 1360
    bottom_padding = "7rem" if view == "Результаты" else "3rem"
    st.markdown(
        f"""
        <style>
        .stApp {{ background: #F7F6F1; color: #20343A; }}
        .stAppHeader {{
            position: relative !important;
            flex: 0 0 3.75rem;
            background: #F7F6F1 !important;
            border-bottom: 1px solid #D9E0D9;
            z-index: 10 !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            max-width: {max_width}px;
            width: 100%;
            padding: 1.35rem clamp(1rem, 2.1vw, 2rem) {bottom_padding};
        }}
        [data-testid="stMain"] {{ overflow-y: auto !important; }}
        input[aria-label="Название исследования"]:focus::placeholder {{ color: transparent; }}
        [data-testid="stSidebar"] {{ background: #214E59; min-width: 210px; max-width: 210px; }}
        [data-testid="stSidebar"] > div {{ width: 210px; }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #F4F8F6; }}
        [data-testid="stSidebar"] [data-testid="stButton"] button {{
            background: transparent;
            color: #F4F8F6;
            border-color: transparent;
            justify-content: flex-start;
            font-weight: 600;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {{
            background: #E7EDD6;
            border-color: #E7EDD6;
            color: #20343A;
        }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] * {{ color: #20343A !important; }}
        [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {{ color: #20343A; }}
        h1 a, h2 a, h3 a, [data-testid="stHeaderActionElements"] {{ display: none !important; }}
        .sidebar-brand {{ font-size: 1.18rem; font-weight: 750; color: #FFFFFF; margin: .25rem 0 .15rem; }}
        .sidebar-subtitle {{ color: #D6E3E1; font-size: .86rem; line-height: 1.35; margin-bottom: 1.4rem; }}
        h1 {{ color: #20343A; font-size: clamp(1.65rem, 2.8vw, 1.9rem) !important; margin-bottom: .15rem !important; }}
        h2 {{ color: #20343A; font-size: 1.35rem !important; }}
        h3 {{ color: #20343A; font-size: 1.08rem !important; }}
        p, label, [data-testid="stMarkdownContainer"] {{ font-size: .98rem; }}
        [data-testid="stCaptionContainer"] {{ color: #526167; font-size: .88rem; }}
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {{
            border-color: #B8C7C4;
            min-height: 2.55rem;
            border-radius: 10px;
            color: #20343A;
        }}
        [data-testid="stButton"] button[kind="primary"] {{
            background: #235D66;
            border-color: #235D66;
            color: #FFFFFF;
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {{ border-color: #235D66; color: #20343A; }}
        [data-testid="stButton"] button:focus-visible,
        [data-testid="stDownloadButton"] button:focus-visible {{
            outline: 3px solid rgba(35, 93, 102, .32);
            outline-offset: 2px;
        }}
        [data-testid="stButton"] button:disabled {{ opacity: .46; }}
        .prototype-note {{
            color: #526167;
            border-left: 3px solid #5798D7;
            padding: .35rem .75rem;
            margin: .55rem 0 1rem;
            font-size: .9rem;
        }}
        .page-lead {{ color: #526167; margin: -.15rem 0 1rem; font-size: .95rem; }}
        .result-interpretation {{
            background: #E7EDD6;
            border: 1px solid #CBD8B6;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            color: #20343A;
            margin: .6rem 0 1.25rem;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.metric-card-marker) {{
            background: #FFFFFF;
            border-color: #D9E0D9;
            min-height: 220px;
            height: 100%;
            padding: .1rem;
            box-shadow: 0 4px 14px rgba(32, 52, 58, .05);
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.chart-card-marker) {{
            background: #FFFFFF;
            border-color: #D9E0D9;
            height: 100%;
            box-shadow: 0 4px 14px rgba(32, 52, 58, .05);
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.chart-card-marker) [data-testid="stImage"] img {{
            width: 100%;
            height: clamp(240px, 29vw, 360px);
            object-fit: contain;
            background: #F3F6FA;
        }}
        .metric-card-marker, .chart-card-marker {{ display: none; }}
        .drawer-tab-label {{
            writing-mode: vertical-rl;
            transform: rotate(180deg);
            color: #526167;
            font-size: .78rem;
            letter-spacing: .04em;
            margin: .45rem auto 0;
        }}
        .analysis-chart-card {{
            height: 620px;
            box-sizing: border-box;
            display: grid;
            grid-template-rows: minmax(2.8rem, auto) 430px 1fr;
            gap: .65rem;
            padding: 1rem;
            border: 1px solid #D9E0D9;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 4px 14px rgba(32, 52, 58, .05);
            overflow: hidden;
        }}
        .analysis-chart-title {{
            color: #20343A;
            font-size: 1rem;
            font-weight: 700;
            line-height: 1.35;
        }}
        .analysis-chart-image {{
            height: 430px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #F3F6FA;
            border-radius: 8px;
            overflow: hidden;
        }}
        .analysis-chart-image img {{ width: 100%; height: 100%; object-fit: contain; }}
        .analysis-chart-caption {{
            color: #526167;
            font-size: .88rem;
            line-height: 1.45;
            overflow: auto;
        }}
        [data-testid="stMetric"] {{ padding: 0; }}
        [data-testid="stMetricLabel"] {{ font-size: .93rem; color: #20343A; }}
        [data-testid="stMetricValue"] {{ font-size: clamp(1.65rem, 3vw, 2rem); color: #20343A; }}
        [data-testid="stAlert"] {{ padding-top: .65rem; padding-bottom: .65rem; }}
        div[data-testid="stImage"] img {{ border-radius: 8px; }}
        [data-testid="stDataFrame"] {{ border: 1px solid #D9E0D9; border-radius: 10px; overflow: hidden; }}
        [data-testid="stExpander"] {{ background: #FFFFFF; border-color: #D9E0D9; border-radius: 10px; }}
        @media (max-width: 900px) {{
            [data-testid="stMainBlockContainer"] {{ padding-left: .85rem; padding-right: .85rem; }}
            [data-testid="stVerticalBlockBorderWrapper"]:has(.metric-card-marker) {{ min-height: 0; }}
            .analysis-chart-card {{ height: auto; grid-template-rows: auto 320px auto; }}
            .analysis-chart-image {{ height: 320px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def node(block: str, x: float, y: float, orientation: str = "horizontal"):
    typ = "input" if block == "input" else "output" if block == "output" else "default"
    vertical = orientation == "vertical"
    return StreamlitFlowNode(
        id=block, pos=(x, y), data={"content": BLOCK_TITLES[block]},
        node_type=typ, source_position="bottom" if vertical else "right",
        target_position="top" if vertical else "left",
        selectable=True, connectable=True, draggable=True, deletable=True,
        style={
            "width": "205px", "minHeight": "66px", "padding": "16px", "backgroundColor": "#ffffff",
            "color": "#173c34", "border": "2px solid #3f8f77", "borderRadius": "12px",
            "fontWeight": "600", "boxShadow": "0 4px 12px rgba(0,0,0,.18)",
        },
    )


def node_position(block: str, orientation: str) -> tuple[float, float]:
    i = BLOCKS.index(block)
    return (220, i * 145) if orientation == "vertical" else (i * 250, 110)


def workflow_edge(source: str, target: str, edge_id: str | None = None) -> StreamlitFlowEdge:
    """Контрастная удаляемая стрелка для ручного и автоматического соединения."""
    return StreamlitFlowEdge(
        edge_id or f"{source}-{target}", source, target,
        edge_type="smoothstep",
        marker_end={"type": "arrowclosed", "width": 24, "height": 24, "color": "#235D66"},
        style={"stroke": "#235D66", "strokeWidth": 3},
        deletable=True,
        focusable=True,
    )


def demo_state(orientation: str = "horizontal"):
    nodes = [node(k, *node_position(k, orientation), orientation) for k in BLOCKS]
    edges = [workflow_edge(a, b) for a, b in EXPECTED_EDGES]
    return StreamlitFlowState(nodes, edges)


def connect_blocks_in_order() -> None:
    """Явно заменяет связи на единственный допустимый линейный маршрут MVP."""
    flow = st.session_state.wf_flow
    present = {n.id for n in flow.nodes}
    missing = [BLOCK_TITLES[block].split(". ", 1)[-1] for block in BLOCKS if block not in present]
    if missing:
        st.session_state.wf_notice = "Сначала добавьте отсутствующие блоки: " + ", ".join(missing) + "."
        return
    st.session_state.wf_flow = StreamlitFlowState(
        list(flow.nodes),
        [workflow_edge(a, b) for a, b in EXPECTED_EDGES],
        selected_id=st.session_state.wf_active_block,
    )
    st.session_state.wf_graph_revision += 1
    st.session_state.wf_notice = "Блоки соединены по порядку: Вход → Подготовка → Анализ → Вывод."
    clear_validation()


def reorient_flow(orientation: str) -> None:
    current = st.session_state.wf_flow
    nodes = [node(n.id, *node_position(n.id, orientation), orientation)
             for n in current.nodes if n.id in BLOCKS]
    st.session_state.wf_flow = StreamlitFlowState(nodes, list(current.edges))
    st.session_state.wf_graph_revision += 1


def boot():
    defaults = {
        "wf_flow": demo_state,
        "wf_cfg": default_settings,
        "wf_csv_bytes": lambda: None,
        "wf_csv_name": lambda: None,
        "wf_file_id": lambda: None,
        "wf_last": lambda: None,
        "wf_pictures": dict,
        "wf_warnings": list,
        "wf_run_id": lambda: None,
        "wf_revision": lambda: 0,
        "wf_graph_revision": lambda: 0,
        "wf_editor_last_event": lambda: None,
        "wf_docx": lambda: None,
        "wf_preserve_settings": lambda: False,
        "wf_view": lambda: "Конструктор",
        "wf_active_block": lambda: "input",
        "wf_last_node_click": lambda: None,
        "wf_validation": lambda: None,
        "wf_run_error": lambda: None,
        "wf_expected_sha": lambda: None,
        "wf_result_source": lambda: "current",
        "wf_run_fingerprint": lambda: None,
        "wf_notice": lambda: None,
        "wf_history_selected": lambda: None,
        "wf_dataset_warnings": list,
        "wf_is_running": lambda: False,
        "wf_run_requested": lambda: False,
        "wf_run_message": lambda: None,
        "wf_restore_pending": lambda: None,
        "wf_confirm_clear": lambda: False,
        "wf_dataset_change_pending": lambda: None,
        "wf_input_control_revision": lambda: 0,
        "wf_study_name_sync": lambda: None,
        "wf_block_selection_revision": lambda: 0,
        "wf_study_name_input": lambda: st.session_state.wf_cfg["study"].get("name", ""),
    }
    for k, maker in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = maker()
    if st.session_state.wf_study_name_sync is not None:
        st.session_state.wf_study_name_input = st.session_state.wf_study_name_sync["value"]
        st.session_state.wf_study_name_sync = None
    if st.session_state.wf_is_running and not st.session_state.wf_run_requested:
        # Streamlit не продолжает прерванный Python-вызов. Оставшийся True — не
        # активная задача, а устаревший флаг, который раньше навсегда блокировал UI.
        st.session_state.wf_is_running = False
        if not st.session_state.wf_run_error:
            st.session_state.wf_run_error = (
                "Предыдущая попытка анализа была прервана. Проверьте готовность и запустите её повторно."
            )
    st.session_state.wf_cfg = normalize_settings(st.session_state.wf_cfg)


def read_dataset():
    source = st.session_state.wf_cfg["input"]["source"]
    if source == DEMO_SOURCE:
        raw = DEMO.read_bytes()
        name = "heart.csv"
    else:
        raw, name = st.session_state.wf_csv_bytes, st.session_state.wf_csv_name
    if not raw:
        return None, None, None
    df, info = load_csv(raw, name)
    token = info["sha256"]
    if token != st.session_state.wf_file_id:
        st.session_state.wf_file_id = token
        updated, warnings = initialize_dataset_settings(
            st.session_state.wf_cfg,
            df,
            preserve_existing=bool(st.session_state.wf_preserve_settings),
        )
        st.session_state.wf_cfg.clear()
        st.session_state.wf_cfg.update(updated)
        st.session_state.wf_dataset_warnings = warnings
        st.session_state.wf_preserve_settings = False
        st.session_state.wf_last = None
        st.session_state.wf_pictures = {}
        st.session_state.wf_docx = None
        st.session_state.wf_validation = None
        st.session_state.wf_run_error = None
        st.session_state.wf_revision += 1
    return df, info, raw


def scheme_payload() -> dict:
    flow = st.session_state.wf_flow
    layout = {n.id: dict(n.position) for n in flow.nodes}
    return {"schema_version": 1,
            "orientation": st.session_state.wf_cfg["editor"]["orientation"],
            "node_positions": layout,
            "edges": [{"source": e.source, "target": e.target} for e in flow.edges],
            "settings": deepcopy(st.session_state.wf_cfg)}


def save_scheme() -> bytes:
    return json.dumps(scheme_payload(), ensure_ascii=False, indent=2, default=str).encode("utf-8")


def apply_scheme(doc: dict):
    if doc.get("schema_version") != 1:
        raise ValueError("Неподдерживаемая версия схемы.")
    places = doc["node_positions"]
    if not isinstance(places, dict) or len(places) > 4 or not set(places).issubset(BLOCKS):
        raise ValueError("Схема содержит неизвестные блоки.")
    incoming = doc.get("settings", {})
    if not isinstance(incoming, dict):
        raise ValueError("Недопустимые настройки схемы.")
    cfg = normalize_settings(incoming)
    orientation = doc.get("orientation", cfg["editor"].get("orientation", "horizontal"))
    if orientation not in ("horizontal", "vertical"):
        orientation = "horizontal"
    cfg["editor"]["orientation"] = orientation
    nodes = []
    for k, xy in places.items():
        if not isinstance(xy, dict):
            raise ValueError("Ошибка координат блоков.")
        nodes.append(node(k, float(xy["x"]), float(xy["y"]), orientation))
    raw_edges = doc["edges"]
    if not isinstance(raw_edges, list) or len(raw_edges) > 10:
        raise ValueError("Слишком много соединений.")
    edges=[]
    for i, e in enumerate(raw_edges):
        a,b=e["source"], e["target"]
        if a not in places or b not in places:
            raise ValueError("Соединение указывает на отсутствующий блок.")
        edges.append(workflow_edge(a, b, f"restored_{i}"))
    method = cfg["analysis"].get("method")
    if method not in METHODS:
        raise ValueError("Неизвестный метод анализа в схеме.")
    st.session_state.wf_flow = StreamlitFlowState(nodes, edges)
    st.session_state.wf_cfg = cfg
    st.session_state.wf_study_name_sync = {"value": cfg["study"].get("name", "")}
    st.session_state.wf_file_id = None
    st.session_state.wf_csv_bytes = None
    st.session_state.wf_csv_name = None
    st.session_state.wf_preserve_settings = True
    st.session_state.wf_dataset_change_pending = None
    st.session_state.wf_input_control_revision += 1
    st.session_state.wf_active_block = next(iter(places), None)
    st.session_state.wf_last_node_click = None
    st.session_state.wf_validation = None
    st.session_state.wf_run_error = None
    st.session_state.wf_revision += 1
    st.session_state.wf_graph_revision += 1


def create_new_draft(source: str, raw: bytes | None = None, name: str | None = None) -> None:
    """Создаёт новый черновик после подтверждённой смены набора данных"""
    if source not in (DEMO_SOURCE, CUSTOM_SOURCE):
        raise ValueError("Неизвестный источник данных.")
    if source == CUSTOM_SOURCE and (not raw or not name):
        raise ValueError("Для собственного набора требуется CSV-файл.")
    cfg = default_settings()
    cfg["input"]["source"] = source
    st.session_state.wf_cfg = cfg
    st.session_state.wf_csv_bytes = raw if source == CUSTOM_SOURCE else None
    st.session_state.wf_csv_name = name if source == CUSTOM_SOURCE else None
    st.session_state.wf_file_id = None
    st.session_state.wf_preserve_settings = False
    st.session_state.wf_flow = StreamlitFlowState([], [])
    st.session_state.wf_active_block = None
    st.session_state.wf_last_node_click = None
    st.session_state.wf_expected_sha = None
    st.session_state.wf_last = None
    st.session_state.wf_pictures = {}
    st.session_state.wf_warnings = []
    st.session_state.wf_docx = None
    st.session_state.wf_run_id = None
    st.session_state.wf_study_name_sync = {"value": ""}
    st.session_state.wf_dataset_change_pending = None
    st.session_state.wf_input_control_revision += 1
    st.session_state.wf_confirm_clear = False
    dataset_name = name if source == CUSTOM_SOURCE else "heart.csv"
    st.session_state.wf_notice = (
        f"Создан новый черновик с набором «{dataset_name}». "
        "Добавьте и соедините нужные блоки."
    )
    st.session_state.wf_revision += 1
    st.session_state.wf_graph_revision += 1
    st.session_state.wf_block_selection_revision += 1
    clear_validation()


def load_scheme(raw: bytes):
    if len(raw) > 200_000:
        raise ValueError("Схема превышает 200 КБ.")
    apply_scheme(json.loads(raw.decode("utf-8-sig")))


def analysis_fingerprint() -> str:
    flow = st.session_state.wf_flow
    payload = {
        "nodes": sorted(n.id for n in flow.nodes),
        "edges": sorted((e.source, e.target) for e in flow.edges),
        "settings": {
            section: deepcopy(st.session_state.wf_cfg[section])
            for section in ("input", "preprocess", "analysis", "output")
        },
        "dataset_sha256": st.session_state.wf_file_id,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clear_validation() -> None:
    st.session_state.wf_validation = None
    st.session_state.wf_run_error = None


def validate_run_config(settings: dict, df, data_error: str) -> tuple[list[str], int | None]:
    """Единая проверка фактического черновика для обеих кнопок запуска."""
    errors: list[str] = []
    n_classes = None
    node_ids, edges = flow_parts()
    try:
        validate_edges(node_ids, edges)
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"Схема: {exc}")

    if data_error:
        errors.append(f"Блок 1 «Входные данные»: {data_error}")
    elif df is None:
        errors.append("Блок 1 «Входные данные»: загрузите читаемый CSV.")
    else:
        preprocess = settings.get("preprocess", {})
        target = preprocess.get("target", "")
        features = list(preprocess.get("features", []))
        if not target or target not in df.columns:
            errors.append("Блок 2 «Подготовка данных»: выберите существующий целевой столбец Y.")
        invalid_features = [name for name in features if name not in df.columns or name == target]
        if not features:
            errors.append("Блок 2 «Подготовка данных»: выберите хотя бы один признак X.")
        elif invalid_features or len(set(features)) != len(features):
            errors.append("Блок 2 «Подготовка данных»: список X содержит цель, дубликаты или неизвестные столбцы.")
        if not any(message.startswith("Блок 2") for message in errors):
            try:
                n_classes = validate_preflight(settings, df)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
    return list(dict.fromkeys(errors)), n_classes


def request_analysis_run() -> None:
    """Callback выполняется до rerun, поэтому следующий кадр уже показывает running."""
    if not st.session_state.wf_is_running:
        st.session_state.wf_is_running = True
        st.session_state.wf_run_requested = True
        st.session_state.wf_run_error = None
        st.session_state.wf_run_message = None


def execute_pending_analysis(settings: dict, df, info, data_error: str) -> None:
    """Запускает расчёт до отрисовки интерактивной схемы.

    Компонент streamlit-flow может прислать собственный rerun при инициализации.
    Поэтому обработка запроса внутри render_constructor была уязвима: такой rerun
    прерывал вычисление и оставлял интерфейс в состоянии ``wf_is_running``.
    """
    if not st.session_state.wf_run_requested:
        return

    # После этой строки новый rerun означает, что текущий Python-вызов был
    # действительно прерван; boot() снимет только устаревшую блокировку.
    st.session_state.wf_run_requested = False
    stage = "проверка схемы и данных"
    try:
        with st.status("Выполняется анализ…", expanded=True) as status:
            status.write("1/4 Проверка схемы и входных данных")
            errors, n_classes = validate_run_config(settings, df, data_error)
            if errors:
                st.session_state.wf_validation = {
                    "ok": False,
                    "message": "Обнаружены проблемы:\n\n" + "\n".join(f"- {message}" for message in errors),
                    "fingerprint": analysis_fingerprint(),
                }
                raise ValueError("; ".join(errors))
            stage = "подготовка данных и обучение модели"
            status.write("2/4 Подготовка Train/Test и обучение модели")
            artifact = run_analysis(df, deepcopy(settings), info)
            settings["study"]["name"] = artifact.summary["study_name"]
            artifact.summary["pipeline_scheme"] = scheme_payload()
            stage = "построение выбранных графиков"
            status.write("3/4 Расчёт выбранных графиков и SHAP")
            pictures, warnings = render_selected(artifact)
            if warnings:
                raise RuntimeError("; ".join(warnings))
            stage = "создание отчёта и сохранение запуска"
            status.write("4/4 Создание Word-отчёта и сохранение истории")
            docx = create_docx(artifact.summary, pictures, warnings)
            run_id = save_summary(artifact.summary, pictures, docx)
            status.update(label="Анализ завершён и сохранён", state="complete", expanded=False)
        st.session_state.wf_last = artifact.summary
        st.session_state.wf_pictures = pictures
        st.session_state.wf_warnings = warnings
        st.session_state.wf_docx = docx
        st.session_state.wf_run_id = run_id
        st.session_state.wf_result_source = "current"
        st.session_state.wf_run_fingerprint = analysis_fingerprint()
        st.session_state.wf_run_error = None
        st.session_state.wf_run_message = f"Анализ завершён. Сохранён запуск №{run_id}."
        st.session_state.wf_view = "Результаты"
        st.session_state.wf_is_running = False
        st.rerun()
    except Exception as exc:
        LOGGER.exception("Ошибка запуска на этапе «%s»", stage)
        st.session_state.wf_is_running = False
        st.session_state.wf_run_requested = False
        st.session_state.wf_run_error = f"Ошибка на этапе «{stage}»: {exc}"
        st.session_state.wf_view = "Конструктор"
        st.rerun()


def choose_method(cfg):
    current = cfg["analysis"].get("method", "xgboost")
    values = list(METHODS)
    chosen = st.selectbox(
        "Метод", values,
        index=values.index(current),
        format_func=lambda k: METHODS[k]["title"],
        help="Выберите алгоритм классификации. Настройки по умолчанию — отправная точка и не подбирались автоматически.",
        key=f"analysis_method_{st.session_state.wf_revision}",
        persist_state="session",
    )
    if chosen != current:
        updated = select_method(cfg, chosen)
        cfg.clear()
        cfg.update(updated)
    ps = deepcopy(cfg["analysis"]["params_by_method"][chosen])
    prefix = f"method_{chosen}_{st.session_state.wf_revision}"
    st.caption(METHODS[chosen]["description"])
    if chosen in ("xgboost", "random_forest"):
        ps["n_estimators"] = st.slider(
            "Количество деревьев", 20, 400, int(ps.get("n_estimators", 100)), 10,
            help="Число деревьев в ансамбле. Большее значение увеличивает объём обучения и может повысить риск переобучения.",
            key=f"{prefix}_n_estimators", persist_state="session",
        )
        ps["max_depth"] = st.slider(
            "Глубина деревьев", 1, 15, int(ps.get("max_depth", 3)),
            help="Максимальная глубина дерева: более глубокие деревья описывают сложные зависимости, но могут переобучаться.",
            key=f"{prefix}_max_depth", persist_state="session",
        )
        if chosen == "xgboost":
            ps["learning_rate"] = st.slider(
                "Скорость обучения (learning_rate)", .01, .3,
                float(ps.get("learning_rate", .05)), .01,
                help="Размер вклада каждого нового дерева в итоговый прогноз.",
                key=f"{prefix}_learning_rate", persist_state="session",
            )
            ps["subsample"] = st.slider(
                "Доля строк на дерево (subsample)", .5, 1.0,
                float(ps.get("subsample", .8)), .05,
                help="Доля обучающих наблюдений, случайно используемая для каждого дерева.",
                key=f"{prefix}_subsample", persist_state="session",
            )
            ps["colsample_bytree"] = st.slider(
                "Доля признаков на дерево", .5, 1.0,
                float(ps.get("colsample_bytree", .8)), .05,
                help="Доля входных признаков, доступная при построении каждого дерева.",
                key=f"{prefix}_colsample_bytree", persist_state="session",
            )
        else:
            ps["min_samples_leaf"] = st.slider(
                "Минимум объектов в листе", 1, 10,
                int(ps.get("min_samples_leaf", 2)),
                help="Минимальное число обучающих наблюдений в конечном узле дерева; увеличение делает модель более сглаженной.",
                key=f"{prefix}_min_samples_leaf", persist_state="session",
            )
    else:
        labels = {"l2": "L2", "l1": "L1", "elasticnet": "Elastic Net"}
        regularizations = list(labels)
        current_regularization = ps.get("regularization", "l2")
        ps["regularization"] = st.selectbox(
            "Регуляризация", regularizations,
            index=regularizations.index(current_regularization),
            format_func=labels.get,
            help="L1 может занулять коэффициенты, L2 сглаживает их, Elastic Net сочетает оба штрафа.",
            key=f"{prefix}_regularization", persist_state="session",
        )
        ps["C"] = st.number_input(
            "C (обратная сила регуляризации)", .01, 100.0,
            float(ps.get("C", 1.0)), .1,
            help="Меньшее C усиливает регуляризацию и сильнее ограничивает коэффициенты модели.",
            key=f"{prefix}_C", persist_state="session",
        )
        ps["class_weight"] = st.selectbox(
            "Вес классов", [None, "balanced"],
            index=1 if ps.get("class_weight") == "balanced" else 0,
            format_func=lambda value: "Не применять" if value is None else "balanced",
            help="balanced автоматически увеличивает вес редких классов обратно пропорционально их частоте.",
            key=f"{prefix}_class_weight", persist_state="session",
        )
        if ps["regularization"] == "elasticnet":
            ps["l1_ratio"] = st.slider(
                "Доля L1 в Elastic Net (l1_ratio)", 0.0, 1.0,
                float(ps.get("l1_ratio", 0.5)), 0.05,
                help="0 соответствует L2, 1 — L1, промежуточные значения смешивают два штрафа.",
                key=f"{prefix}_l1_ratio", persist_state="session",
            )
        with st.expander("Дополнительные параметры"):
            ps["max_iter"] = st.slider(
                "Максимум итераций", 200, 3000,
                int(ps.get("max_iter", 800)), 100,
                help="Предельное число итераций оптимизации коэффициентов логистической регрессии.",
                key=f"{prefix}_max_iter", persist_state="session",
            )
            ps["tol"] = st.number_input(
                "Допуск остановки (tol)", min_value=1e-6, max_value=0.1,
                value=float(ps.get("tol", 0.0001)), format="%.6f",
                help="Меньший допуск требует более точной сходимости и может увеличить время обучения.",
                key=f"{prefix}_tol", persist_state="session",
            )
            ps["fit_intercept"] = st.checkbox(
                "Оценивать свободный член (fit_intercept)",
                value=bool(ps.get("fit_intercept", True)),
                help="Добавляет свободный член в линейную модель.",
                key=f"{prefix}_fit_intercept", persist_state="session",
            )
        _, actual = logistic_solver_details(ps, 2)
        st.info(f"Решатель: {actual['solver']}. {actual['solver_reason']}")
    commit_active_params(cfg, ps)
    st.caption("Настройки по умолчанию — отправная точка; они не подбирались автоматически.")


def choose_preprocess(cfg, df):
    p = cfg["preprocess"]
    if df is None:
        st.info("Сначала выберите источник данных в блоке 1.")
        return
    columns = list(df.columns)
    guess = p["target"] if p["target"] in columns else ("target" if "target" in columns else
                     "Outcome" if "Outcome" in columns else columns[-1])
    target = st.selectbox(
        "Целевой столбец Y", columns, index=columns.index(guess),
        help="Y — результат, класс или состояние, которое должна предсказывать модель.",
    )
    if target != p["target"]:
        p["target"] = target
        p["features"] = [c for c in columns if c != target]
        p["categorical"] = [c for c in p["features"] if not pd.api.types.is_numeric_dtype(df[c])]
        p["descriptive_columns"] = []
        p["positive"] = ""
    y = df[target]
    unique = sorted(y.dropna().astype(str).unique().tolist())
    st.caption(f"Обнаружено классов: {len(unique)}. Поддерживается от 2 до 4.")
    if y.isna().any():
        st.error("У целевой переменной есть пропуски.")
    if len(unique) == 2:
        preferred = p["positive"] if p["positive"] in unique else ("1" if "1" in unique else unique[-1])
        p["positive"] = st.selectbox(
            "Положительный класс", unique, index=unique.index(preferred),
            help="Исследуемое событие для расчёта Precision, Recall и F1. Медицинский смысл класса определяет исследователь.",
        )
    else:
        p["positive"] = ""
        if 2 < len(unique) <= 4:
            st.info("Для 3–4 классов выводятся метрики по каждому классу и macro/weighted.")
    options = [c for c in columns if c != target]
    p["features"] = st.multiselect(
        "Признаки X", options,
        default=[v for v in p["features"] if v in options],
        help="X — характеристики наблюдений, по которым модель предсказывает Y. Целевой столбец в X не включается.",
    )
    p["categorical"] = st.multiselect(
        "Категориальные признаки (One-Hot)", p["features"],
        default=[c for c in p["categorical"] if c in p["features"]],
        help="Отметьте категории и числовые коды категорий. Количественные измерения отмечать не нужно; One-Hot создаст отдельные индикаторы категорий.",
    )
    suspicious = [c for c in p["categorical"] if df[c].nunique() > 12]
    if suspicious:
        st.warning("Проверьте высокую кардинальность: " + ", ".join(suspicious))
    p["test_size"] = st.slider(
        "Доля Test", .1, .4, float(p["test_size"]), .05,
        help="Часть наблюдений, отложенная для независимой проверки. Например, 0.20 означает 20% данных.",
    )
    p["random_state"] = int(st.number_input(
        "random_state", 0, 100000, int(p["random_state"]),
        help="Фиксирует случайное разделение Train/Test и позволяет воспроизвести запуск.",
    ))
    st.caption(f"Полных совпадений: {int(df.duplicated().sum())}. Удаление — только после подтверждения.")
    p["remove_duplicates"] = st.checkbox(
        "Подтверждаю исключение полных совпадений",
        value=bool(p["remove_duplicates"]),
        help="Удаляются только полностью одинаковые строки. Решение принимает пользователь после проверки смысла повторов.",
    )
    p["scaling"] = st.selectbox("Масштабирование", ["auto", "none", "standard"],
                                index=["auto", "none", "standard"].index(p.get("scaling", "auto")),
                                format_func=lambda k: {"auto":"По выбранному методу", "none":"Не применять",
                                                       "standard":"StandardScaler"}[k],
                                help="StandardScaler приводит числовые признаки к сопоставимому масштабу и подгоняется только на Train. Для логистической регрессии он обычно полезен; деревьям обычно не требуется.")
    recommended = METHODS[cfg["analysis"]["method"]]["scaling"]
    st.caption(
        "При выборе «По выбранному методу» будет применено: "
        f"{'StandardScaler' if recommended == 'standard' else 'без масштабирования'}. "
        "Статистики преобразования рассчитываются только по Train."
    )
    with st.expander("Специальные обозначения пропусков"):
        st.caption("Например, 0 для Glucose — только по описанию конкретного CSV. Такие значения учитываются как пропуски при анализе; исходный CSV не меняется.")
        st.caption("После разделения Train/Test числовые пропуски заполняются медианой Train, категориальные — самым частым значением Train.")
        if p["features"]:
            c = st.selectbox("Столбец для правила", p["features"])
            text = st.text_input("Значения через ;", "; ".join(p["missing_rules"].get(c, [])),
                                 key=f"marker_{st.session_state.wf_revision}_{c}")
            if st.button("Сохранить правило"):
                vals = list(dict.fromkeys(v.strip() for v in text.split(";") if v.strip()))
                if vals: p["missing_rules"][c] = vals
                else: p["missing_rules"].pop(c, None)
                st.success("Правило сохранено.")
            if p["missing_rules"]:
                st.json(p["missing_rules"])
    st.divider()
    p["descriptive"] = st.checkbox("Рассчитать описательную статистику",
                                   value=bool(p["descriptive"]))
    if p["descriptive"]:
        numeric = [c for c in p["features"] if c not in p["categorical"]]
        p["descriptive_columns"] = st.multiselect("Числовые признаки для обзора", numeric,
                 default=[c for c in p["descriptive_columns"] if c in numeric] or numeric[:6])
        st.caption("Обзорные показатели не применяются при обучении; графики выбираются в блоке 4.")


def choose_output(cfg, df):
    out = cfg["output"]
    p = cfg["preprocess"]
    count = len(df[p["target"]].dropna().astype(str).unique()) if df is not None and p["target"] in df else 2
    statuses = available_charts(cfg, count)
    st.caption("Отметьте только те графики, которые нужны в анализе и Word-отчёте.")
    selected = set(out.get("charts", []))
    preferences = set(out.get("chart_preferences", selected))
    for key in SELECTABLE_CHARTS:
        title = CHARTS[key]
        ok, reason = statuses[key]
        checked = st.checkbox(
            title, value=key in preferences, disabled=not ok,
            key=f"figure_{st.session_state.wf_revision}_{key}",
            persist_state="session",
            help=CHART_HELP[key],
        )
        if checked:
            preferences.add(key)
        else:
            preferences.discard(key)
        if checked and ok:
            selected.add(key)
        else:
            selected.discard(key)
        if not ok:
            st.caption(reason)
    out["chart_preferences"] = [c for c in SELECTABLE_CHARTS if c in preferences]
    out["charts"] = [c for c in SELECTABLE_CHARTS if c in selected]
    if any(key in out["charts"] for key in ("shap_beeswarm", "shap_bar")):
        classes = (sorted(df[p["target"]].dropna().astype(str).unique().tolist())
                   if df is not None and p["target"] in df else [])
        if classes:
            chosen = out.get("shap_class", "")
            if chosen not in classes:
                chosen = p.get("positive") if p.get("positive") in classes else classes[0]
            if len(classes) == 2 and p.get("positive") in classes:
                out["shap_class"] = p["positive"]
                st.caption(f"SHAP будет рассчитан для положительного класса: {p['positive']}.")
            else:
                out["shap_class"] = st.selectbox(
                    "Класс для SHAP", classes, index=classes.index(chosen),
                    key=f"shap_class_{st.session_state.wf_revision}",
                    persist_state="session",
                )
        explainer = METHODS[cfg["analysis"]["method"]]["shap_explainer"]
        scale = {
            "xgboost": "raw output модели",
            "random_forest": "вероятность класса",
            "logistic_regression": "линейный выход (log-odds)",
        }[cfg["analysis"]["method"]]
        st.caption(
            f"Оба SHAP-графика используют один расчёт {explainer}; шкала — {scale}. "
            "Для ограничения времени объясняется не более 120 Test-наблюдений на фоне до 80 Train-наблюдений. "
            "SHAP описывает поведение модели и не доказывает причинность."
        )


def editor_snapshot() -> dict:
    """StreamlitFlowState is a temporary legacy data carrier, not the editor engine."""
    flow = st.session_state.wf_flow
    return {
        "nodes": [
            {"id": n.id, "position": {"x": float(n.position["x"]), "y": float(n.position["y"])}}
            for n in flow.nodes
        ],
        "edges": [{"source": e.source, "target": e.target} for e in flow.edges],
    }


def process_editor_message(message: dict) -> bool:
    """Consume a single semantic editor action with deduplication.

    The frontend never submits style metadata, nor can it change analysis settings.
    """
    if not isinstance(message, dict) or not isinstance(message.get("event"), dict):
        return False
    event = message["event"]
    event_id = (event.get("instance"), event.get("seq"))
    if not isinstance(event_id[0], str) or not isinstance(event_id[1], int):
        return False
    if st.session_state.wf_editor_last_event == event_id:
        return False
    st.session_state.wf_editor_last_event = event_id
    orientation = st.session_state.wf_cfg["editor"]["orientation"]
    positions = {k: {"x": node_position(k, orientation)[0],
                     "y": node_position(k, orientation)[1]} for k in BLOCKS}
    try:
        updated, selected, changed = apply_graph_event(
            editor_snapshot(), event, BLOCKS, positions
        )
    except GraphEditError as exc:
        st.session_state.wf_notice = f"Действие со схемой отклонено: {exc}"
        return True
    if changed:
        nodes = [node(n["id"], n["position"]["x"], n["position"]["y"], orientation)
                 for n in updated["nodes"]]
        edges = [workflow_edge(e["source"], e["target"]) for e in updated["edges"]]
        st.session_state.wf_flow = StreamlitFlowState(nodes, edges)
        st.session_state.wf_graph_revision += 1
        clear_validation()
    if selected and selected != st.session_state.wf_active_block:
        st.session_state.wf_active_block = selected
        st.session_state.wf_block_selection_revision += 1
    if st.session_state.wf_active_block not in {n["id"] for n in updated["nodes"]}:
        st.session_state.wf_active_block = None
        st.session_state.wf_block_selection_revision += 1
    return changed or selected is not None


def flow_parts() -> tuple[list[str], list[tuple[str, str]]]:
    flow = st.session_state.wf_flow
    return [n.id for n in flow.nodes], [(e.source, e.target) for e in flow.edges]


def structure_status() -> tuple[bool, str]:
    node_ids, edges = flow_parts()
    missing = [block for block in BLOCKS if block not in node_ids]
    if missing:
        names = [BLOCK_TITLES[block].split(". ", 1)[-1] for block in missing]
        if len(names) == 1:
            return False, f"Не хватает блока «{names[0]}». Добавьте его из библиотеки и восстановите связь."
        return False, "Не хватает блоков: " + ", ".join(f"«{name}»" for name in names) + "."
    try:
        validate_edges(node_ids, edges)
        return True, "Соединения корректны: Вход → Подготовка → Анализ → Вывод."
    except ValueError:
        return False, "Проверьте соединения: требуется линейная схема Вход → Подготовка → Анализ → Вывод без лишних связей."


def render_navigation() -> None:
    pages = (
        ("Конструктор", "▦  Конструктор"),
        ("Результаты", "▥  Результаты"),
        ("История", "◷  История исследований"),
    )
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Аналитические пайплайны</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-subtitle">Анализ медико-биологических табличных данных</div>',
            unsafe_allow_html=True,
        )
        for page, label in pages:
            if st.button(
                label,
                key=f"nav_{page}",
                type="primary" if st.session_state.wf_view == page else "secondary",
                width="stretch",
            ):
                st.session_state.wf_view = page
                st.rerun()
        st.markdown("---")
        st.caption("Исследовательский прототип. Используйте только обезличенные CSV; результаты не являются клиническим заключением.")


def render_input_settings(settings: dict, df, info, data_error: str) -> None:
    current = settings["input"]
    options = [DEMO_SOURCE, CUSTOM_SOURCE]
    current_source = current["source"] if current["source"] in options else DEMO_SOURCE
    control_revision = st.session_state.wf_input_control_revision
    source = st.radio(
        "Источник данных",
        options,
        index=options.index(current_source),
        help="CSV используется локально и не передаётся во внешние сервисы.",
        key=f"workflow_source_{control_revision}",
    )

    pending = st.session_state.wf_dataset_change_pending
    if source == DEMO_SOURCE and source != current_source and pending is None:
        st.session_state.wf_dataset_change_pending = {"source": DEMO_SOURCE}
        pending = st.session_state.wf_dataset_change_pending

    if source == CUSTOM_SOURCE:
        upload = st.file_uploader(
            "CSV-файл",
            type="csv",
            key=f"workflow_csv_{control_revision}",
            help="До подтверждения активный набор данных и настройки не изменяются.",
        )
        if upload is not None:
            raw = upload.getvalue()
            current_raw = st.session_state.wf_csv_bytes
            same_file = bool(
                current_source == CUSTOM_SOURCE
                and current_raw
                and hashlib.sha256(current_raw).digest() == hashlib.sha256(raw).digest()
            )
            if same_file:
                st.caption("Этот CSV уже используется в текущем черновике.")
            elif (
                current_source == CUSTOM_SOURCE
                and st.session_state.wf_preserve_settings
                and current_raw is None
            ):
                # CSV прикрепляется к восстановленной конфигурации без очистки схемы
                st.session_state.wf_csv_bytes = raw
                st.session_state.wf_csv_name = upload.name
                st.session_state.wf_file_id = None
                st.session_state.wf_dataset_change_pending = None
                st.session_state.wf_input_control_revision += 1
                st.session_state.wf_notice = (
                    f"CSV «{upload.name}» прикреплён к восстановленной конфигурации."
                )
                st.rerun()
            elif pending is None:
                st.session_state.wf_dataset_change_pending = {
                    "source": CUSTOM_SOURCE,
                    "raw": raw,
                    "name": upload.name,
                }
                pending = st.session_state.wf_dataset_change_pending
        elif source != current_source:
            st.info("Выберите CSV. До выбора и подтверждения прежний набор остаётся активным.")

    if pending:
        next_name = pending.get("name", "heart.csv")
        st.warning(
            f"Перейти к набору «{next_name}»? После подтверждения будет создан новый "
            "черновик с пустой схемой и начальными настройками. Текущая история запусков "
            "и сохранённые отчёты останутся без изменений."
        )
        confirm, cancel = st.columns(2)
        if confirm.button("Подтвердить смену", type="primary", width="stretch"):
            create_new_draft(
                pending["source"],
                raw=pending.get("raw"),
                name=pending.get("name"),
            )
            st.rerun()
        if cancel.button("Отмена", width="stretch"):
            st.session_state.wf_dataset_change_pending = None
            st.session_state.wf_input_control_revision += 1
            st.rerun()

    if data_error:
        st.error(data_error)
    elif df is None:
        st.info("Загрузите CSV для настройки следующих блоков.")
    else:
        st.write(f"Строк: **{len(df)}**, столбцов: **{len(df.columns)}**")
        st.dataframe(df.head(6), hide_index=True, width="stretch")
        st.caption(f"Пустых ячеек: {info['missing_cells']}; полных совпадений строк: {info['duplicates']}.")
        expected = st.session_state.wf_expected_sha
        if expected:
            if info["sha256"] == expected:
                st.success("Контрольная сумма совпадает с набором сохранённого запуска.")
            else:
                st.warning("SHA-256 загруженного CSV не совпадает с набором сохранённого запуска. Проверьте файл до повторного анализа.")


def render_settings(active: str, settings: dict, df, info, data_error: str) -> None:
    if active == "input":
        render_input_settings(settings, df, info, data_error)
    elif active == "preprocess":
        choose_preprocess(settings, df)
    elif active == "analysis":
        choose_method(settings)
    else:
        choose_output(settings, df)


def render_constructor(settings: dict, df, info, data_error: str) -> None:
    st.title("Конструктор анализа")
    st.markdown(
        '<div class="page-lead">Соберите линейный пайплайн, настройте блоки, проверьте параметры и запустите анализ.</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.wf_notice:
        st.info(st.session_state.wf_notice)

    settings["study"]["name"] = st.text_input(
        "Название исследования",
        placeholder="Например, Факторы риска сердечно-сосудистых заболеваний",
        help="Название будет показано в результатах, истории и Word-отчёте. Если оставить поле пустым, название сформируется автоматически.",
        key="wf_study_name_input",
        persist_state="session",
    ).strip()

    with st.container(border=True):
        file_col, size_col = st.columns(2)
        dataset_name = info["dataset_name"] if info else "Файл не выбран"
        file_col.markdown(f"**Активные данные**  \n{dataset_name}")
        size_col.markdown(
            f"**Размер таблицы**  \n{len(df)} строк · {len(df.columns)} столбцов" if df is not None
            else "**Размер таблицы**  \n—"
        )
    for warning in st.session_state.wf_dataset_warnings:
        st.warning(warning)

    # The drawer is local React state. Its width never changes Streamlit columns,
    # so opening it does not recreate the canvas or reset its pan/zoom.
    middle, right = st.columns([3.6, 1.55], gap="medium")

    with middle:
        st.subheader("Схема анализа")
        orientation = st.segmented_control(
            "Расположение блоков", ["horizontal", "vertical"],
            default=settings["editor"].get("orientation", "horizontal"),
            format_func=lambda v: "Горизонтально" if v == "horizontal" else "Вертикально",
            key=f"orientation_{st.session_state.wf_revision}",
        ) or settings["editor"].get("orientation", "horizontal")
        if orientation != settings["editor"].get("orientation"):
            settings["editor"]["orientation"] = orientation
            reorient_flow(orientation)
            clear_validation()
            st.rerun()
        restore, join, clean = st.columns(3)
        if restore.button("Восстановить пример", width="stretch"):
            st.session_state.wf_flow = demo_state(settings["editor"]["orientation"])
            st.session_state.wf_active_block = "input"
            st.session_state.wf_last_node_click = None
            st.session_state.wf_confirm_clear = False
            st.session_state.wf_graph_revision += 1
            st.session_state.wf_block_selection_revision += 1
            clear_validation()
            st.rerun()
        if join.button(
            "Соединить по порядку", width="stretch",
            help="Восстанавливает линейный маршрут. Любую отдельную стрелку можно затем удалить или перенаправить мышью.",
        ):
            connect_blocks_in_order()
            st.rerun()
        if clean.button("Очистить поле", key="show_clear_confirmation", width="stretch"):
            st.session_state.wf_confirm_clear = True
            st.rerun()
        if st.session_state.wf_confirm_clear:
            with st.container(border=True):
                st.warning("Удалить все блоки и связи? Настройки анализа останутся в черновике.")
                yes, no = st.columns(2)
                if yes.button("Да, очистить схему", type="primary", width="stretch"):
                    st.session_state.wf_flow = StreamlitFlowState([], [])
                    st.session_state.wf_active_block = None
                    st.session_state.wf_confirm_clear = False
                    st.session_state.wf_graph_revision += 1
                    st.session_state.wf_block_selection_revision += 1
                    st.session_state.wf_notice = "Схема очищена. Добавьте блоки из библиотеки редактора."
                    clear_validation()
                    st.rerun()
                if no.button("Отмена", width="stretch"):
                    st.session_state.wf_confirm_clear = False
                    st.rerun()
        download, upload_col = st.columns(2)
        download.download_button(
            "Сохранить шаблон JSON", save_scheme(),
            file_name="workflow_scheme.json", mime="application/json", width="stretch",
        )
        with upload_col:
            with st.popover("Открыть шаблон JSON", width="stretch"):
                uploaded = st.file_uploader("Файл схемы", type="json", key="workflow_json")
                if st.button("Восстановить", disabled=uploaded is None, width="stretch"):
                    try:
                        load_scheme(uploaded.getvalue())
                        st.session_state.wf_notice = "Шаблон восстановлен. CSV необходимо выбрать отдельно."
                        st.rerun()
                    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                        st.error(str(exc))
                st.caption("JSON хранит структуру и параметры, но не медицинские CSV и не результаты.")
        st.caption(
            "Перетащите конец стрелки для перенаправления. Нажмите на линию — "
            "появится панель удаления и изменения связи. Библиотека открывается сбоку от поля."
        )
        try:
            message = graph_editor(
                editor_snapshot(), st.session_state.wf_graph_revision,
                settings["editor"]["orientation"], key="diploma_graph_v2",
            )
            if process_editor_message(message):
                st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
        ok, status_text = structure_status()
        (st.success if ok else st.warning)(f"Структура схемы: {status_text}")

    check = False
    with right:
        st.subheader("Настройки блока")
        ids = [block for block in BLOCKS if block in {n.id for n in st.session_state.wf_flow.nodes}]
        if not ids:
            st.info("Добавьте блок из библиотеки слева.")
        else:
            active = st.session_state.wf_active_block
            if active not in ids:
                active = ids[0]
                st.session_state.wf_active_block = active
            # Только реальные блоки из схемы; нет бесполезного пункта «Выберите блок».
            # Отдельная ревизия синхронизирует выбор на схеме и в этом списке.
            picked = st.selectbox(
                "Выбранный блок", ids,
                index=ids.index(active),
                format_func=lambda block: BLOCK_TITLES[block].split(". ", 1)[-1],
                key=(f"wf_block_picker_{st.session_state.wf_revision}_"
                     f"{st.session_state.wf_block_selection_revision}"),
            )
            if picked != active:
                st.session_state.wf_active_block = picked
                active = picked
            render_settings(active, settings, df, info, data_error)

        st.divider()
        check = st.button("Проверить готовность к запуску", width="stretch", disabled=st.session_state.wf_is_running)
        st.button(
            "Анализ выполняется…" if st.session_state.wf_is_running else "Выполнить анализ",
            type="primary",
            width="stretch",
            disabled=st.session_state.wf_is_running,
            on_click=request_analysis_run,
        )
        if st.session_state.wf_is_running:
            st.caption("Запрос принят. Повторный запуск временно недоступен.")

    fingerprint = analysis_fingerprint()
    validation = st.session_state.wf_validation
    if validation and validation.get("fingerprint") != fingerprint:
        st.info("Настройки изменены после последней проверки — требуется повторная проверка готовности.")

    if check:
        errors, n_classes = validate_run_config(settings, df, data_error)
        if not errors:
            st.session_state.wf_validation = {
                "ok": True,
                "message": f"Схема и настройки проверены. Анализ можно запустить. Классов — {n_classes}; обучение ещё не выполнялось.",
                "fingerprint": analysis_fingerprint(),
            }
        else:
            st.session_state.wf_validation = {
                "ok": False,
                "message": "Обнаружены проблемы:\n\n" + "\n".join(f"- {message}" for message in errors),
                "fingerprint": analysis_fingerprint(),
            }
    validation = st.session_state.wf_validation
    if validation and validation.get("fingerprint") == analysis_fingerprint():
        (st.success if validation["ok"] else st.error)(validation["message"])
        if not validation["ok"]:
            destinations = {
                "Блок 1": "input", "Блок 2": "preprocess",
                "Блок 3": "analysis", "Блок 4": "output",
            }
            destination = next(
                (block for marker, block in destinations.items() if marker in validation["message"]),
                None,
            )
            if destination in {n.id for n in st.session_state.wf_flow.nodes}:
                if st.button(
                    f"Открыть: {BLOCK_TITLES[destination].split('. ', 1)[-1]}",
                    key="open_validation_block",
                ):
                    st.session_state.wf_active_block = destination
                    st.rerun()
    if st.session_state.wf_run_error:
        st.error(st.session_state.wf_run_error)

def render_metric_grid(
    items: list[tuple[str, str, float, str, str]],
    columns_per_row: int = 3,
) -> None:
    """Карточки с определением и интерпретацией конкретного запуска."""
    for group_start in range(0, len(items), columns_per_row):
        columns = st.columns(columns_per_row, gap="medium")
        for column, (label, key, value, definition, interpretation) in zip(
            columns, items[group_start:group_start + columns_per_row]
        ):
            with column:
                with st.container(border=True):
                    st.markdown('<span class="metric-card-marker"></span>', unsafe_allow_html=True)
                    st.metric(label, f"{value:.3f}", help=METRIC_HELP[key])
                    st.caption(f"Что означает: {definition}")
                    st.write(f"**В этом запуске:** {interpretation}")


def binary_metric_items(summary: dict) -> list[tuple[str, str, float, str, str]]:
    metrics = summary["metrics"]
    matrix = summary.get("confusion_matrix", [])
    if len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        tn, fp = int(matrix[0][0]), int(matrix[0][1])
        fn, tp = int(matrix[1][0]), int(matrix[1][1])
        total = tn + fp + fn + tp
        accuracy_text = (
            f"{tp + tn} из {total} тестовых наблюдений классифицировано правильно "
            f"({metrics['accuracy']:.1%})."
            if total else "В тестовой выборке нет наблюдений для интерпретации."
        )
        precision_text = (
            f"Верны {tp} из {tp + fp} положительных прогнозов. Ложных тревог: {fp}."
            if tp + fp else "Модель не выдала положительных прогнозов; TP + FP = 0."
        )
        recall_text = (
            f"Найдено {tp} из {tp + fn} положительных наблюдений. Пропущено: {fn}."
            if tp + fn else "В Test нет реальных положительных наблюдений; TP + FN = 0."
        )
        specificity_text = (
            f"Правильно определено {tn} из {tn + fp} отрицательных наблюдений; ложных тревог: {fp}."
            if tn + fp else "В Test нет реальных отрицательных наблюдений; TN + FP = 0."
        )
    else:
        accuracy_text = "Показатель рассчитан на отложенной тестовой выборке."
        precision_text = recall_text = specificity_text = "Матрица ошибок недоступна для числовой расшифровки."
    return [
        (
            "Accuracy · Доля верных ответов", "accuracy", metrics["accuracy"],
            "доля всех наблюдений, классифицированных правильно.", accuracy_text,
        ),
        (
            "Precision (+) · Точность", "precision", metrics["precision"],
            "среди положительных прогнозов — сколько действительно положительные.", precision_text,
        ),
        (
            "Recall (+) · Чувствительность", "recall", metrics["recall"],
            "среди реальных положительных случаев — сколько модель обнаружила.", recall_text,
        ),
        (
            "F1 (+) · F1-мера", "f1", metrics["f1"],
            "гармоническое среднее Precision и Recall.",
            f"Сводный баланс точности положительных прогнозов и полноты обнаружения: {metrics['f1']:.3f}.",
        ),
        (
            "Specificity · Специфичность", "specificity", metrics["specificity"],
            "среди реальных отрицательных — сколько модель определила правильно.", specificity_text,
        ),
        (
            "ROC AUC", "roc_auc", metrics["roc_auc"],
            "насколько оценки модели разделяют классы при изменении порога.",
            f"ROC AUC = {metrics['roc_auc']:.3f}; это не доля правильных ответов.",
        ),
    ]


def multiclass_metric_items(summary: dict) -> list[tuple[str, str, float, str, str]]:
    metrics = summary["metrics"]
    matrix = summary.get("confusion_matrix", [])
    total = sum(sum(int(value) for value in row) for row in matrix)
    correct = sum(int(row[index]) for index, row in enumerate(matrix) if index < len(row))
    classes = int(summary["n_classes"])
    return [
        (
            "Accuracy · Доля верных ответов", "accuracy", metrics["accuracy"],
            "доля всех наблюдений, классифицированных правильно.",
            f"Правильно классифицировано {correct} из {total} тестовых наблюдений ({metrics['accuracy']:.1%}).",
        ),
        (
            "Precision macro", "precision_macro", metrics["precision_macro"],
            "среднее Precision по классам с одинаковым весом каждого класса.",
            f"Среднее по {classes} классам: {metrics['precision_macro']:.3f}.",
        ),
        (
            "Recall macro", "recall_macro", metrics["recall_macro"],
            "среднее Recall по классам с одинаковым весом каждого класса.",
            f"Среднее по {classes} классам: {metrics['recall_macro']:.3f}.",
        ),
        (
            "F1 macro", "f1_macro", metrics["f1_macro"],
            "среднее F1 по классам с одинаковым весом каждого класса.",
            f"Среднее по {classes} классам: {metrics['f1_macro']:.3f}.",
        ),
    ]


def additional_metrics_frame(metrics: dict, include_auc: bool = False) -> pd.DataFrame:
    labels = {
        "precision_macro": "Precision macro",
        "recall_macro": "Recall macro",
        "f1_macro": "F1 macro",
        "precision_weighted": "Precision weighted",
        "recall_weighted": "Recall weighted",
        "f1_weighted": "F1 weighted",
        "roc_auc_ovr_macro": "ROC AUC OvR macro",
    }
    keys = ["precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted"]
    if include_auc:
        keys.append("roc_auc_ovr_macro")
    return pd.DataFrame([{labels[key]: metrics[key] for key in keys if key in metrics}]).round(3)


def render_run_interpretation(summary: dict) -> None:
    """Кратко объясняет только наблюдаемые результаты, без клинических выводов."""
    metrics = summary["metrics"]
    matrix = summary.get("confusion_matrix", [])
    if summary["n_classes"] == 2 and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
        tn, fp = int(matrix[0][0]), int(matrix[0][1])
        fn, tp = int(matrix[1][0]), int(matrix[1][1])
        total = tn + fp + fn + tp
        comparison = (
            "В этом запуске чувствительность выше специфичности"
            if metrics["recall"] > metrics["specificity"]
            else "В этом запуске специфичность выше чувствительности"
            if metrics["specificity"] > metrics["recall"]
            else "В этом запуске чувствительность и специфичность равны"
        )
        body = (
            f"Модель правильно классифицировала <strong>{tp + tn} из {total}</strong> тестовых наблюдений. "
            f"Она обнаружила <strong>{tp} из {tp + fn}</strong> наблюдений положительного класса и "
            f"пропустила <strong>{fn}</strong>; для отрицательного класса правильно определено "
            f"<strong>{tn} из {tn + fp}</strong>, ложноположительных ответов — <strong>{fp}</strong>. "
            f"{comparison}. Практическая значимость этого соотношения зависит от цели исследования и цены разных ошибок."
        )
    else:
        total = sum(sum(int(value) for value in row) for row in matrix)
        correct = sum(int(row[index]) for index, row in enumerate(matrix) if index < len(row))
        body = (
            f"Модель правильно классифицировала <strong>{correct} из {total}</strong> тестовых наблюдений "
            f"для {summary['n_classes']} классов. Macro-показатели дают каждому классу одинаковый вес; "
            "различия между отдельными классами показаны в таблице ниже."
        )
    st.markdown(
        '<div class="result-interpretation"><strong>Что показывают результаты этого запуска</strong><br>'
        f'{body}<br><span style="font-size:.88rem;color:#526167">'
        'Это статистическое описание тестовой выборки, а не клиническое заключение.</span></div>',
        unsafe_allow_html=True,
    )


def render_chart_card(summary: dict, key: str, picture: bytes) -> None:
    """Равная карточка с фиксированной зоной изображения без деформации PNG."""
    base_key = figure_base_key(key)
    title = CHARTS[base_key]
    if summary["n_classes"] > 2 and base_key in ("roc", "pr"):
        title += " · каждый класс против остальных (OvR)"
    group = summary.get("descriptive_visualization", {}).get("groups", {}).get(key, [])
    if group:
        title += " · " + ", ".join(group)
    caption = CHART_HELP.get(base_key, "")
    if base_key in ("shap", "shap_beeswarm", "shap_bar") and summary.get("shap_metadata"):
        meta = summary["shap_metadata"]
        caption += (
            f" Класс: {meta.get('class', meta.get('class_name', '—'))}; "
            f"{meta.get('explainer', '—')}; шкала: {meta.get('output_scale', '—')}."
        )
    encoded = base64.b64encode(picture).decode("ascii")
    st.markdown(
        '<div class="analysis-chart-card">'
        f'<div class="analysis-chart-title">{html.escape(title)}</div>'
        f'<div class="analysis-chart-image"><img src="data:image/png;base64,{encoded}" alt="{html.escape(title)}"></div>'
        f'<div class="analysis-chart-caption">{html.escape(caption)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def selected_figure_keys(selected: list[str], pictures: dict[str, bytes]) -> list[str]:
    """Сохраняет порядок выбранных графиков и их описательных групп"""
    result: list[str] = []
    for base in selected:
        grouped = [
            key for key in pictures
            if figure_base_key(key) == base
        ]
        grouped.sort(
            key=lambda key: 1 if "__" not in key else int(key.rsplit("__", 1)[1])
        )
        result.extend(grouped)
    return result


def render_results() -> None:
    summary = st.session_state.wf_last
    title_col, action_col = st.columns([4, 1])
    with title_col:
        st.title("Результаты анализа")
        st.markdown(
            '<div class="page-lead">Метрики, графики и пояснения для выбранного сохранённого запуска.</div>',
            unsafe_allow_html=True,
        )
    with action_col:
        st.markdown('<div style="height:1.15rem"></div>', unsafe_allow_html=True)
        if st.button("← К конструктору", width="stretch"):
            st.session_state.wf_view = "Конструктор"
            st.rerun()
    if not summary:
        st.info("Выполните анализ в разделе «Конструктор» или откройте сохранённый запуск в разделе «История исследований».")
        return

    if st.session_state.wf_result_source == "current" and st.session_state.wf_run_message:
        st.success(st.session_state.wf_run_message)

    run_id = st.session_state.wf_run_id or "—"
    st.subheader(summary.get("study_name", "Исследование без названия"))
    dataset = summary.get("dataset", {})
    created = summary.get("created_at", "—")
    with st.container(border=True):
        first, second, third, fourth = st.columns([1.1, 1.5, 1.2, 1.1])
        first.markdown(f"**Запуск**  \n№{run_id}")
        second.markdown(f"**Набор данных**  \n{dataset.get('dataset_name', '—')}")
        third.markdown(f"**Метод**  \n{METHODS.get(summary['model'], {}).get('title', summary['model'])}")
        fourth.markdown(f"**Выборки**  \nTrain {summary['train_rows']} · Test {summary['test_rows']}")
        st.caption(f"Создан: {created} · классов: {summary['n_classes']}")

    if st.session_state.wf_docx:
        st.download_button(
            "Скачать отчёт Word (.docx)",
            st.session_state.wf_docx,
            file_name=f"analysis_{run_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    elif st.session_state.wf_result_source == "history":
        st.caption("Для этого старого запуска файл DOCX не был сохранён в истории.")

    if st.session_state.wf_result_source == "history":
        st.info("Показаны сохранённые результаты выбранного запуска. Они не пересчитывались.")
    elif st.session_state.wf_run_fingerprint and st.session_state.wf_run_fingerprint != analysis_fingerprint():
        st.warning(f"Результаты относятся к запуску №{run_id}. Настройки после запуска изменены — выполните новый анализ для обновления результатов.")
    else:
        st.caption("Показатели рассчитаны на отложенной тестовой части данных и относятся к зафиксированной конфигурации запуска.")

    metrics = summary["metrics"]
    if summary["n_classes"] == 2:
        render_metric_grid(binary_metric_items(summary))
    else:
        render_metric_grid(multiclass_metric_items(summary), columns_per_row=2)
    render_run_interpretation(summary)

    pictures = st.session_state.wf_pictures or {}
    selected_charts = summary.get("config", {}).get("output", {}).get("charts", [])
    available = selected_figure_keys(selected_charts, pictures)
    if available:
        st.subheader("Графики")
        for group_start in range(0, len(available), 2):
            columns = st.columns(2, gap="medium", vertical_alignment="top")
            for column, key in zip(columns, available[group_start:group_start + 2]):
                with column:
                    render_chart_card(summary, key, pictures[key])
        skipped = summary.get("descriptive_visualization", {}).get("skipped", [])
        if skipped:
            st.caption(
                "Не включены в описательные графики как неподходящие или нечисловые: "
                + ", ".join(skipped)
            )
    elif selected_charts:
        st.info("Для этого сохранённого запуска изображения не были записаны в историю. Метрики и таблицы доступны, но графики нельзя восстановить без повторного анализа.")
    for warning in st.session_state.wf_warnings:
        st.warning(warning)

    st.subheader("Подробности")
    if summary["n_classes"] == 2:
        with st.expander("Формулы и подробности основных метрик"):
            st.markdown(
                "- **Accuracy** = (TP + TN) / (TP + TN + FP + FN).\n"
                "- **Precision (+)** = TP / (TP + FP).\n"
                "- **Recall (+)** = TP / (TP + FN).\n"
                "- **F1 (+)** = 2 × Precision × Recall / (Precision + Recall).\n"
                "- **Specificity** = TN / (TN + FP).\n"
                "- **ROC AUC** оценивает ранжирование классов при разных порогах и не равен Accuracy."
            )
        with st.expander("Дополнительные macro и weighted показатели"):
            st.write("**Macro** даёт каждому классу одинаковый вес. **Weighted** учитывает число наблюдений каждого класса в Test.")
            st.dataframe(additional_metrics_frame(metrics), hide_index=True, width="stretch")
    else:
        with st.expander("Weighted показатели и ROC AUC OvR"):
            st.write("Macro усредняет классы с одинаковым весом; weighted учитывает число реальных наблюдений каждого класса. ROC AUC OvR сравнивает каждый класс со всеми остальными.")
            st.dataframe(additional_metrics_frame(metrics, include_auc=True), hide_index=True, width="stretch")

    with st.expander("Метрики по каждому классу", expanded=summary["n_classes"] > 2):
        per_class = pd.DataFrame(summary.get("metrics_per_class", [])).rename(columns={
            "class": "Класс",
            "precision": "Точность положительного прогноза (Precision)",
            "recall": "Полнота (Recall)",
            "f1": "F1",
            "support": "Число наблюдений (support)",
            "specificity_ovr": "Специфичность для класса (OvR)",
        })
        st.write("Для каждого класса показатели рассчитаны по схеме «этот класс против остальных» (OvR). Support — число реальных наблюдений класса в Test.")
        numeric = [c for c in per_class.columns if c not in ("Класс", "Число наблюдений (support)")]
        if numeric:
            per_class[numeric] = per_class[numeric].round(3)
        st.dataframe(per_class, hide_index=True, width="stretch")

    if summary.get("descriptive"):
        with st.expander("Описательная статистика"):
            descriptive = pd.DataFrame(summary["descriptive"]).rename(columns={
                "N": "Количество наблюдений (N)",
                "Пропуски": "Количество пропусков",
                "SD": "Стандартное отклонение (SD)",
                "Мин": "Минимум",
                "Макс": "Максимум",
            })
            text_columns = ("Признак", "Количество наблюдений (N)", "Количество пропусков")
            number_columns = [c for c in descriptive.columns if c not in text_columns]
            if number_columns:
                descriptive[number_columns] = descriptive[number_columns].round(3)
            st.dataframe(descriptive, hide_index=True, width="stretch")
            st.write("**Количество наблюдений (N)** — число непустых значений. **Среднее** — арифметическое среднее, **медиана** — центральное значение упорядоченного ряда, **стандартное отклонение (SD)** — мера разброса.")
            st.caption("Показатели рассчитаны по выбранным столбцам исходной таблицы с учётом специальных маркеров пропусков. Imputer и StandardScaler обучаются отдельно и только на Train.")

    st.caption(
        f"Метрики рассчитаны на Test ({summary['test_rows']} наблюдений). "
        "Они не подтверждают пригодность модели для клинического применения."
    )
    with st.expander("Настройки и воспроизводимость"):
        st.json({k: v for k, v in summary.items() if k not in ("_viz", "importance")})


def restore_as_working_copy(saved: dict) -> None:
    """Восстанавливает конфигурацию, не меняя сохранённый запуск в истории."""
    summary = saved["summary"]
    scheme = summary.get("pipeline_scheme")
    if not isinstance(scheme, dict):
        default = demo_state()
        scheme = {
            "schema_version": 1,
            "orientation": "horizontal",
            "node_positions": {n.id: dict(n.position) for n in default.nodes},
            "edges": [{"source": e.source, "target": e.target} for e in default.edges],
            "settings": summary.get("config", default_settings()),
        }
    apply_scheme(deepcopy(scheme))
    dataset = summary.get("dataset", {})
    expected_sha = dataset.get("sha256")
    st.session_state.wf_expected_sha = expected_sha
    demo_matches = (
        dataset.get("dataset_name") == "heart.csv"
        and DEMO.exists()
        and hashlib.sha256(DEMO.read_bytes()).hexdigest() == expected_sha
    )
    if demo_matches:
        st.session_state.wf_cfg["input"]["source"] = "Демонстрационный heart.csv"
        st.session_state.wf_notice = (
            f"Создана рабочая копия конфигурации запуска №{saved['id']}. "
            "Исходный сохранённый запуск не изменён; демонстрационные данные совпали по SHA-256."
        )
    else:
        st.session_state.wf_cfg["input"]["source"] = "Загрузить свой CSV"
        st.session_state.wf_csv_bytes = None
        st.session_state.wf_csv_name = None
        st.session_state.wf_file_id = None
        st.session_state.wf_notice = (
            f"Создана рабочая копия конфигурации запуска №{saved['id']}. "
            "Исходный запуск не изменён. Загрузите исходный CSV и проверьте его SHA-256."
        )
    st.session_state.wf_restore_pending = None
    st.session_state.wf_view = "Конструктор"


def render_history() -> None:
    st.title("История исследований")
    st.markdown(
        '<div class="page-lead">Сохранённые в локальной SQLite-базе запуски и их воспроизводимые настройки.</div>',
        unsafe_allow_html=True,
    )
    rows = recent_runs()
    if not rows:
        st.info("Сохранённых запусков пока нет.")
        return

    by_id = {row["id"]: row for row in rows}
    ids = list(by_id)
    if st.session_state.wf_history_selected not in ids:
        st.session_state.wf_history_selected = ids[0]

    master, detail = st.columns([1.05, 2.15], gap="large")
    with master:
        st.subheader("Сохранённые запуски")
        selected_id = st.radio(
            "Выберите исследование",
            ids,
            key="wf_history_selected",
            format_func=lambda rid: (
                f"{by_id[rid]['study_name']}\n"
                f"{by_id[rid]['created'][:10]} · №{rid}"
            ),
        )

    with detail:
        try:
            saved = get_run(selected_id)
        except ValueError as exc:
            saved = None
            st.error(str(exc))
        if saved is None:
            st.error("Запуск не найден в локальной базе данных.")
            return

        summary = saved["summary"]
        row = by_id[selected_id]
        st.subheader(row["study_name"])
        st.caption(f"Сохранённый запуск №{selected_id}")
        with st.container(border=True):
            a, b = st.columns(2)
            a.markdown(f"**Набор данных**  \n{row['dataset']}")
            b.markdown(f"**Метод**  \n{METHODS.get(row['method'], {}).get('title', row['method'])}")
            a.markdown(f"**Создано**  \n{row['created']}")
            b.markdown(
                f"**Основные метрики**  \nAccuracy {row['accuracy']:.3f} · F1 {row['f1']:.3f}"
                if row["accuracy"] is not None and row["f1"] is not None
                else "**Основные метрики**  \n—"
            )
            artifact_text = (
                f"{row['figure_count']} графиков · "
                f"Word-отчёт {'сохранён' if row['has_report'] else 'не сохранён'}"
            )
            a.markdown(f"**Классы**  \n{row['n_classes'] if row['n_classes'] is not None else '—'}")
            b.markdown(f"**Артефакты**  \n{artifact_text}")
            pipeline = summary.get("pipeline", [])
            if pipeline:
                names = [BLOCK_TITLES.get(key, key).split(". ", 1)[-1] for key in pipeline]
                st.markdown("**Пайплайн**  \n" + " → ".join(names))

        open_col, restore_col, report_col = st.columns(3)
        if open_col.button("Открыть результаты", type="primary", width="stretch"):
            st.session_state.wf_last = summary
            st.session_state.wf_pictures = saved["figures"]
            st.session_state.wf_warnings = []
            st.session_state.wf_docx = saved["report"]
            st.session_state.wf_run_id = saved["id"]
            st.session_state.wf_result_source = "history"
            st.session_state.wf_run_fingerprint = None
            st.session_state.wf_view = "Результаты"
            st.rerun()

        if restore_col.button("Восстановить настройки", width="stretch"):
            st.session_state.wf_restore_pending = selected_id

        if st.session_state.wf_restore_pending == selected_id:
            st.warning(
                "Текущий рабочий черновик будет заменён конфигурацией этого запуска. "
                "Сохранённый запуск и его результаты останутся без изменений."
            )
            confirm, cancel = st.columns(2)
            if confirm.button("Создать рабочую копию", type="primary", width="stretch"):
                restore_as_working_copy(saved)
                st.rerun()
            if cancel.button("Отмена", width="stretch"):
                st.session_state.wf_restore_pending = None
                st.rerun()

        with report_col:
            if saved["report"]:
                st.download_button(
                    "Скачать Word-отчёт",
                    saved["report"],
                    file_name=f"analysis_{selected_id}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    width="stretch",
                )
            else:
                st.button("DOCX недоступен", disabled=True, width="stretch")
        st.caption("Открытие результатов и восстановление настроек не запускают обучение. Пользовательский CSV в истории не сохраняется.")
        with st.expander("Подробности воспроизводимости"):
            st.json({k: v for k, v in summary.items() if k not in ("_viz", "importance")})


boot()
apply_interface_style(st.session_state.wf_view)
settings = st.session_state.wf_cfg
try:
    df, info, _ = read_dataset()
    data_error = ""
except (ValueError, OSError) as exc:
    df = info = None
    data_error = str(exc)

execute_pending_analysis(settings, df, info, data_error)
render_navigation()
if st.session_state.wf_view == "Конструктор":
    render_constructor(settings, df, info, data_error)
elif st.session_state.wf_view == "Результаты":
    render_results()
else:
    render_history()
