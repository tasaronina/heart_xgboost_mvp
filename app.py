"""Единый визуальный конструктор дипломного MVP."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from workflow_mvp.data_io import load_csv
from workflow_mvp.engine import run_analysis
from workflow_mvp.persistence import get_run, recent_runs, save_summary
from workflow_mvp.report_docx import create_docx
from workflow_mvp.spec import (BLOCKS, BLOCK_TITLES, CHARTS, EXPECTED_EDGES,
                               METHODS, available_charts, default_settings, validate_edges,
                               validate_preflight)
from workflow_mvp.visuals import render_selected

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "sample_data" / "heart.csv"
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

CHART_HELP = {
    "confusion": "По строкам — реальные классы, по столбцам — предсказанные. Числа на диагонали — правильные ответы. В бинарном случае обозначения FP и FN верны при порядке классов [отрицательный, положительный].",
    "roc": "По оси X — доля ложноположительных срабатываний (FPR), по Y — чувствительность (TPR). Каждая точка соответствует порогу; пунктирная диагональ — случайное ранжирование.",
    "pr": "Показывает изменение Precision и Recall при различных порогах; особенно полезна, когда важен поиск редкого класса.",
    "importance": "Показывает использованные моделью признаки и их относительную важность: gain для XGBoost, impurity для Random Forest или |коэффициент| для логистической регрессии. Не доказывает причинность.",
    "shap": "Каждая точка — одно наблюдение; положение по горизонтали показывает вклад признака в выбранный raw output модели, а не обязательно изменение вероятности.",
    "hist": "Показывает распределение значений выбранных числовых признаков.",
    "box": "Показывает медиану, квартильный разброс и отдельные необычные значения; это не автоматическое доказательство ошибки измерения.",
    "corr": "Показывает попарные статистические связи между числовыми признаками; корреляция не означает причинность.",
}


def apply_interface_style(view: str) -> None:
    """Локальные стили компоновки; цвета виджетов задаются темой Streamlit."""
    max_width = 1180 if view == "Результаты" else 1360
    st.markdown(
        f"""
        <style>
        [data-testid="stMainBlockContainer"] {{
            max-width: {max_width}px;
            width: 100%;
            padding: 1.35rem clamp(1rem, 2.4vw, 2rem) 3rem;
        }}
        h1 {{ font-size: clamp(1.75rem, 3vw, 2.25rem) !important; margin-bottom: .25rem !important; }}
        h2 {{ font-size: 1.55rem !important; }}
        h3 {{ font-size: 1.15rem !important; }}
        [data-testid="stCaptionContainer"] {{ color: #ACBACB; font-size: .86rem; }}
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {{ border-color: #34445D; min-height: 2.5rem; }}
        [data-testid="stButton"] button[kind="primary"] {{
            background: #1E4D55;
            border-color: #68C6BD;
            color: #E9F0F7;
        }}
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover {{ border-color: #68C6BD; color: #E9F0F7; }}
        [data-testid="stButton"] button:focus-visible,
        [data-testid="stDownloadButton"] button:focus-visible {{
            outline: 3px solid rgba(104, 198, 189, .45);
            outline-offset: 2px;
        }}
        [data-testid="stButton"] button:disabled {{ opacity: .46; }}
        .prototype-note {{
            color: #ACBACB;
            border-left: 3px solid #68C6BD;
            padding: .35rem .75rem;
            margin: .55rem 0 1rem;
            font-size: .9rem;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.metric-card-marker) {{
            background: #192538;
            border-color: #34445D;
            min-height: 220px;
            height: 100%;
            padding: .1rem;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.chart-card-marker) {{
            background: #192538;
            border-color: #34445D;
            height: 100%;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:has(.chart-card-marker) [data-testid="stImage"] img {{
            width: 100%;
            height: clamp(240px, 29vw, 360px);
            object-fit: contain;
            background: #F3F6FA;
        }}
        .metric-card-marker, .chart-card-marker {{ display: none; }}
        [data-testid="stMetric"] {{ padding: 0; }}
        [data-testid="stMetricLabel"] {{ font-size: .93rem; color: #E9F0F7; }}
        [data-testid="stMetricValue"] {{ font-size: clamp(1.65rem, 3vw, 2rem); color: #E9F0F7; }}
        [data-testid="stAlert"] {{ padding-top: .65rem; padding-bottom: .65rem; }}
        div[data-testid="stImage"] img {{ border-radius: 8px; }}
        @media (max-width: 900px) {{
            [data-testid="stMainBlockContainer"] {{ padding-left: .85rem; padding-right: .85rem; }}
            [data-testid="stVerticalBlockBorderWrapper"]:has(.metric-card-marker) {{ min-height: 0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def node(block: str, x: float, y: float):
    typ = "input" if block == "input" else "output" if block == "output" else "default"
    return StreamlitFlowNode(
        id=block, pos=(x, y), data={"content": BLOCK_TITLES[block]},
        node_type=typ, source_position="right", target_position="left",
        selectable=True, connectable=True, draggable=True, deletable=True,
        style={
            "width": "205px", "padding": "14px", "backgroundColor": "#ffffff",
            "color": "#173c34", "border": "2px solid #3f8f77", "borderRadius": "12px",
            "fontWeight": "600", "boxShadow": "0 4px 12px rgba(0,0,0,.18)",
        },
    )


def demo_state():
    nodes = [node(k, i*250, 110) for i, k in enumerate(BLOCKS)]
    edges = [StreamlitFlowEdge(f"{a}-{b}", a, b, marker_end={"type": "arrowclosed"})
             for a,b in EXPECTED_EDGES]
    return StreamlitFlowState(nodes, edges)


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
        "wf_analysis_method": lambda: "xgboost",
    }
    for k, maker in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = maker()


def read_dataset():
    source = st.session_state.wf_cfg["input"]["source"]
    if source == "Демонстрационный heart.csv":
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
        if not st.session_state.wf_preserve_settings:
            p = st.session_state.wf_cfg["preprocess"]
            p.update(target="", positive="", features=[], categorical=[], missing_rules={},
                     descriptive_columns=[], remove_duplicates=False)
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
    return {"schema_version": 1, "node_positions": layout,
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
    nodes = []
    for k, xy in places.items():
        if not isinstance(xy, dict):
            raise ValueError("Ошибка координат блоков.")
        nodes.append(node(k, float(xy["x"]), float(xy["y"])))
    raw_edges = doc["edges"]
    if not isinstance(raw_edges, list) or len(raw_edges) > 10:
        raise ValueError("Слишком много соединений.")
    edges=[]
    for i, e in enumerate(raw_edges):
        a,b=e["source"], e["target"]
        if a not in places or b not in places:
            raise ValueError("Соединение указывает на отсутствующий блок.")
        edges.append(StreamlitFlowEdge(f"restored_{i}", a, b,
                                      marker_end={"type": "arrowclosed"}))
    incoming = doc.get("settings", {})
    if not isinstance(incoming, dict):
        raise ValueError("Недопустимые настройки схемы.")
    cfg = default_settings()
    for section in BLOCKS:
        vals = incoming.get(section, {})
        if isinstance(vals, dict):
            cfg[section].update({k: v for k,v in vals.items() if k in cfg[section]})
    method = cfg["analysis"].get("method")
    if method not in METHODS:
        raise ValueError("Неизвестный метод анализа в схеме.")
    st.session_state.wf_flow = StreamlitFlowState(nodes, edges)
    st.session_state.wf_cfg = cfg
    st.session_state.wf_analysis_method = method
    st.session_state.wf_file_id = None
    st.session_state.wf_csv_bytes = None
    st.session_state.wf_csv_name = None
    st.session_state.wf_last = None
    st.session_state.wf_docx = None
    st.session_state.wf_pictures = {}
    st.session_state.wf_warnings = []
    st.session_state.wf_preserve_settings = True
    st.session_state.wf_active_block = next(iter(places), None)
    st.session_state.wf_last_node_click = None
    st.session_state.wf_validation = None
    st.session_state.wf_run_error = None
    st.session_state.wf_revision += 1


def load_scheme(raw: bytes):
    if len(raw) > 200_000:
        raise ValueError("Схема превышает 200 КБ.")
    apply_scheme(json.loads(raw.decode("utf-8-sig")))


def analysis_fingerprint() -> str:
    payload = {
        "scheme": scheme_payload(),
        "dataset_sha256": st.session_state.wf_file_id,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clear_validation() -> None:
    st.session_state.wf_validation = None
    st.session_state.wf_run_error = None


def choose_method(cfg):
    current = cfg["analysis"].get("method", "xgboost")
    values = list(METHODS)
    if st.session_state.get("wf_analysis_method") not in values:
        st.session_state.wf_analysis_method = current
    chosen = st.selectbox(
        "Метод", values,
        format_func=lambda k: METHODS[k]["title"],
        help="Выберите алгоритм классификации. Настройки по умолчанию — отправная точка и не подбирались автоматически.",
        key="wf_analysis_method",
    )
    if chosen != current:
        cfg["analysis"]["method"] = chosen
        cfg["analysis"]["params"] = deepcopy(METHODS[chosen]["parameters"])
        cfg["output"]["charts"] = [k for k in cfg["output"].get("charts", [])
                                    if k != "shap" or METHODS[chosen]["shap"]]
    ps = cfg["analysis"].setdefault("params", {})
    st.caption(METHODS[chosen]["description"])
    if chosen in ("xgboost", "random_forest"):
        ps["n_estimators"] = st.slider(
            "Количество деревьев", 20, 400, int(ps.get("n_estimators", 100)), 10,
            help="Число деревьев в ансамбле. Большее значение увеличивает объём обучения и может повысить риск переобучения.",
        )
        ps["max_depth"] = st.slider(
            "Глубина деревьев", 1, 15, int(ps.get("max_depth", 3)),
            help="Максимальная глубина дерева: более глубокие деревья описывают сложные зависимости, но могут переобучаться.",
        )
        if chosen == "xgboost":
            ps["learning_rate"] = st.slider(
                "Скорость обучения (learning_rate)", .01, .3,
                float(ps.get("learning_rate", .05)), .01,
                help="Размер вклада каждого нового дерева в итоговый прогноз.",
            )
            ps["subsample"] = st.slider(
                "Доля строк на дерево (subsample)", .5, 1.0,
                float(ps.get("subsample", .8)), .05,
                help="Доля обучающих наблюдений, случайно используемая для каждого дерева.",
            )
            ps["colsample_bytree"] = st.slider(
                "Доля признаков на дерево", .5, 1.0,
                float(ps.get("colsample_bytree", .8)), .05,
                help="Доля входных признаков, доступная при построении каждого дерева.",
            )
        else:
            ps["min_samples_leaf"] = st.slider(
                "Минимум объектов в листе", 1, 10,
                int(ps.get("min_samples_leaf", 2)),
                help="Минимальное число обучающих наблюдений в конечном узле дерева; увеличение делает модель более сглаженной.",
            )
    else:
        ps["C"] = st.number_input(
            "C (обратная сила регуляризации)", .01, 100.0,
            float(ps.get("C", 1.0)), .1,
            help="Меньшее C усиливает регуляризацию и сильнее ограничивает коэффициенты модели.",
        )
        ps["max_iter"] = st.slider(
            "Максимум итераций", 200, 2000,
            int(ps.get("max_iter", 800)), 100,
            help="Предельное число итераций оптимизации коэффициентов логистической регрессии.",
        )
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
    for key, title in CHARTS.items():
        ok, reason = statuses[key]
        checked = st.checkbox(title, value=key in selected and ok, disabled=not ok,
                              key=f"figure_{st.session_state.wf_revision}_{key}")
        if checked and ok: selected.add(key)
        else: selected.discard(key)
        if not ok:
            st.caption(reason)
    out["charts"] = [c for c in CHARTS if c in selected]
    if "shap" in out["charts"]:
        classes = (sorted(df[p["target"]].dropna().astype(str).unique().tolist())
                   if df is not None and p["target"] in df else [])
        if classes:
            chosen = out.get("shap_class", "")
            if chosen not in classes:
                chosen = p.get("positive") if p.get("positive") in classes else classes[0]
            out["shap_class"] = st.selectbox("Класс для SHAP", classes, index=classes.index(chosen))
        st.caption("SHAP вычисляется только после выбора графика. Для XGBoost объясняется raw output, не вероятность.")


def flow_parts() -> tuple[list[str], list[tuple[str, str]]]:
    flow = st.session_state.wf_flow
    return [n.id for n in flow.nodes], [(e.source, e.target) for e in flow.edges]


def render_navigation() -> None:
    st.title("Конструктор анализа медико-биологических данных")
    st.caption("4 блока · 3 метода · классификация 2–4 классов · DOCX · SHAP для XGBoost")
    st.markdown(
        '<div class="prototype-note">Исследовательский прототип: используйте только обезличенные CSV. '
        'Результаты не являются клиническим заключением.</div>',
        unsafe_allow_html=True,
    )
    pages = ("Конструктор", "Результаты", "История")
    columns = st.columns(3)
    for column, page in zip(columns, pages):
        if column.button(
            page,
            key=f"nav_{page}",
            type="primary" if st.session_state.wf_view == page else "secondary",
            width="stretch",
        ):
            st.session_state.wf_view = page
            st.rerun()
    st.divider()


def render_input_settings(settings: dict, df, info, data_error: str) -> None:
    current = settings["input"]
    options = ["Демонстрационный heart.csv", "Загрузить свой CSV"]
    source = st.radio(
        "Источник данных",
        options,
        index=options.index(current["source"]) if current["source"] in options else 0,
        help="CSV используется локально и не передаётся во внешние сервисы.",
    )
    if source != current["source"]:
        current["source"] = source
        st.session_state.wf_last = None
        st.session_state.wf_docx = None
        clear_validation()
        st.rerun()
    if source == "Загрузить свой CSV":
        upload = st.file_uploader("CSV-файл", type="csv", key="workflow_csv")
        if upload is not None:
            raw = upload.getvalue()
            st.session_state.wf_csv_bytes = raw
            st.session_state.wf_csv_name = upload.name
            if st.session_state.wf_file_id != hashlib.sha256(raw).hexdigest():
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
    left, middle, right = st.columns([1.05, 3.35, 1.85], gap="small")
    # Компонент схемы выполняется первым: библиотека ниже получает уже актуальные
    # node.id после ручного удаления узла во frontend streamlit_flow.
    with middle:
        st.subheader("Действия со схемой")
        a, b, c = st.columns(3)
        if a.button("Восстановить пример", width="stretch"):
            st.session_state.wf_flow = demo_state()
            st.session_state.wf_active_block = "input"
            st.session_state.wf_last_node_click = None
            st.session_state.wf_revision += 1
            clear_validation()
            st.rerun()
        if b.button("Очистить поле", width="stretch"):
            st.session_state.wf_flow = StreamlitFlowState([], [])
            st.session_state.wf_active_block = None
            st.session_state.wf_last_node_click = None
            st.session_state.wf_revision += 1
            clear_validation()
            st.rerun()
        c.download_button(
            "Сохранить схему JSON",
            save_scheme(),
            file_name="workflow_scheme.json",
            mime="application/json",
            width="stretch",
        )
        with st.expander("Открыть схему JSON"):
            saved = st.file_uploader("JSON-файл схемы", type="json", key="workflow_json")
            if st.button("Восстановить из файла", disabled=saved is None):
                try:
                    load_scheme(saved.getvalue())
                    st.rerun()
                except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                    st.error(str(exc))
            st.caption("JSON хранит структуру и настройки, но не исходный CSV и не результаты анализа.")

        st.subheader("Рабочее поле")
        st.caption("Соединяйте выход справа со входом слева. Порядок определяется стрелками, а не положением блоков.")
        flow = streamlit_flow(
            f"diploma_workflow_{st.session_state.wf_revision}",
            st.session_state.wf_flow,
            layout=ManualLayout(),
            height=470,
            fit_view=True,
            allow_new_edges=True,
            get_node_on_click=True,
            enable_edge_menu=True,
            enable_node_menu=True,
            enable_pane_menu=False,
            show_minimap=False,
            hide_watermark=True,
        )
        st.session_state.wf_flow = flow
        ids = [n.id for n in flow.nodes]
        clicked = flow.selected_id
        click_event = (clicked, getattr(flow, "timestamp", None))
        if clicked in ids and click_event != st.session_state.wf_last_node_click:
            st.session_state.wf_active_block = clicked
            st.session_state.wf_last_node_click = click_event
            st.session_state.wf_active_select = clicked
        try:
            validate_edges(ids, [(e.source, e.target) for e in flow.edges])
            st.success("Соединения корректны: Вход → Подготовка → Анализ → Вывод.")
        except ValueError as exc:
            st.warning(f"Структура схемы: {exc}")

    with left:
        st.subheader("Библиотека блоков")
        st.caption("Добавьте по одному блоку каждого типа в рабочую схему.")
        present = {n.id for n in st.session_state.wf_flow.nodes}
        for block in BLOCKS:
            if st.button(
                "+ " + BLOCK_TITLES[block],
                disabled=block in present,
                key=f"add_{block}",
                width="stretch",
            ):
                nodes = list(st.session_state.wf_flow.nodes)
                edges = list(st.session_state.wf_flow.edges)
                nodes.append(node(block, len(nodes) * 235, 110))
                st.session_state.wf_flow = StreamlitFlowState(nodes, edges, selected_id=block)
                st.session_state.wf_active_block = block
                st.session_state.wf_last_node_click = None
                st.session_state.wf_revision += 1
                clear_validation()
                st.rerun()

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
            if st.session_state.get("wf_active_select") not in ids:
                st.session_state.wf_active_select = active
            selected = st.selectbox(
                "Выбранный блок",
                ids,
                format_func=lambda key: BLOCK_TITLES[key],
                key="wf_active_select",
            )
            st.session_state.wf_active_block = selected
            render_settings(selected, settings, df, info, data_error)

    fingerprint = analysis_fingerprint()
    validation = st.session_state.wf_validation
    if validation and validation.get("fingerprint") != fingerprint:
        st.session_state.wf_validation = None

    st.divider()
    check_col, run_col = st.columns(2)
    check = check_col.button("Проверить пайплайн", width="stretch")
    run = run_col.button("Выполнить анализ", type="primary", width="stretch")
    if check:
        try:
            node_ids, edges = flow_parts()
            validate_edges(node_ids, edges)
            if data_error:
                raise ValueError(f"Блок 1 «Входные данные»: {data_error}")
            if df is None:
                raise ValueError("Блок 1 «Входные данные»: загрузите читаемый CSV.")
            n_classes = validate_preflight(settings, df)
            st.session_state.wf_validation = {
                "ok": True,
                "message": f"Предпусковая проверка пройдена: данные читаются, настройки согласованы, классов — {n_classes}. Обучение модели не выполнялось.",
                "fingerprint": analysis_fingerprint(),
            }
        except (ValueError, KeyError, TypeError) as exc:
            st.session_state.wf_validation = {
                "ok": False,
                "message": str(exc),
                "fingerprint": analysis_fingerprint(),
            }
    validation = st.session_state.wf_validation
    if validation:
        (st.success if validation["ok"] else st.error)(validation["message"])
    if st.session_state.wf_run_error:
        st.error(st.session_state.wf_run_error)

    if run:
        try:
            node_ids, edges = flow_parts()
            validate_edges(node_ids, edges)
            if data_error:
                raise ValueError(f"Блок 1 «Входные данные»: {data_error}")
            if df is None:
                raise ValueError("Блок 1 «Входные данные»: загрузите читаемый CSV.")
            validate_preflight(settings, df)
            with st.spinner("Подготовка Train/Test → обучение → метрики → выбранные графики..."):
                artifact = run_analysis(df, deepcopy(settings), info)
                artifact.summary["pipeline_scheme"] = scheme_payload()
                pictures, warnings = render_selected(artifact)
                docx = create_docx(artifact.summary, pictures, warnings)
                run_id = save_summary(artifact.summary, pictures, docx)
            st.session_state.wf_last = artifact.summary
            st.session_state.wf_pictures = pictures
            st.session_state.wf_warnings = warnings
            st.session_state.wf_docx = docx
            st.session_state.wf_run_id = run_id
            st.session_state.wf_result_source = "current"
            st.session_state.wf_run_fingerprint = analysis_fingerprint()
            st.session_state.wf_run_error = None
            st.session_state.wf_view = "Результаты"
            st.rerun()
        except Exception as exc:
            st.session_state.wf_run_error = f"Не удалось выполнить анализ: {exc}"
            st.session_state.wf_view = "Конструктор"
            st.rerun()


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


def render_results() -> None:
    st.header("Результаты анализа")
    summary = st.session_state.wf_last
    if not summary:
        st.info("Выполните анализ в разделе «Конструктор» или откройте сохранённый запуск в разделе «История».")
        return
    run_id = st.session_state.wf_run_id
    st.caption(
        f"Запуск №{run_id} · {METHODS[summary['model']]['title']} · "
        f"классов: {summary['n_classes']} · Train {summary['train_rows']} / Test {summary['test_rows']}"
    )
    if st.session_state.wf_result_source == "history":
        st.info("Показаны сохранённые результаты выбранного запуска. Они не пересчитывались.")
    elif st.session_state.wf_run_fingerprint and st.session_state.wf_run_fingerprint != analysis_fingerprint():
        st.warning(f"Результаты относятся к запуску №{run_id}. Настройки после запуска изменены — выполните новый анализ для обновления результатов.")
    else:
        st.caption("Показатели рассчитаны на тестовой части данных и относятся к зафиксированной конфигурации запуска.")

    metrics = summary["metrics"]
    if summary["n_classes"] == 2:
        render_metric_grid(binary_metric_items(summary))
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
        render_metric_grid(multiclass_metric_items(summary), columns_per_row=2)
        st.caption("Macro усредняет классы с одинаковым весом; weighted учитывает число реальных наблюдений каждого класса.")
        with st.expander("Weighted показатели и ROC AUC OvR"):
            st.write("ROC AUC OvR сравнивает каждый класс со всеми остальными и затем усредняет результат.")
            st.dataframe(additional_metrics_frame(metrics, include_auc=True), hide_index=True, width="stretch")

    st.caption(
        f"Рассчитано на отложенной тестовой выборке (Test, {summary['test_rows']} наблюдений). "
        "Метрики не подтверждают пригодность модели для клинического применения."
    )

    with st.expander("Метрики по каждому классу", expanded=summary["n_classes"] > 2):
        per_class = pd.DataFrame(summary.get("metrics_per_class", [])).rename(columns={
            "class": "Класс",
            "precision": "Точность (Precision)",
            "recall": "Полнота (Recall)",
            "f1": "F1",
            "support": "Наблюдений",
            "specificity_ovr": "Специфичность (класс против остальных)",
        })
        st.write(
            "Таблица помогает увидеть различия между классами. **Наблюдений** — число реальных записей "
            "класса в Test. Precision, Recall, F1 и специфичность для каждого класса рассчитаны по схеме "
            "«этот класс против остальных» (OvR)."
        )
        numeric = [c for c in per_class.columns if c not in ("Класс", "Наблюдений")]
        if numeric:
            per_class[numeric] = per_class[numeric].round(3)
        st.dataframe(per_class, hide_index=True, width="stretch")

    if summary.get("descriptive"):
        with st.expander("Описательная статистика"):
            descriptive = pd.DataFrame(summary["descriptive"]).rename(columns={
                "N": "Наблюдений",
                "Пропуски": "Пропусков",
                "SD": "Стандартное отклонение",
                "Мин": "Минимум",
                "Макс": "Максимум",
            })
            number_columns = [c for c in descriptive.columns if c not in ("Признак", "Наблюдений", "Пропусков")]
            if number_columns:
                descriptive[number_columns] = descriptive[number_columns].round(3)
            st.dataframe(descriptive, hide_index=True, width="stretch")
            st.write(
                "**Наблюдений** — число непустых значений признака. **Среднее** — сумма значений, "
                "делённая на их число. **Медиана** — центральное значение упорядоченного ряда. "
                "**Стандартное отклонение** — мера разброса вокруг среднего в единицах исходного показателя."
            )
            st.caption("Показатели рассчитаны по выбранным столбцам исходной таблицы с учётом специальных маркеров пропусков. Imputer и StandardScaler обучаются отдельно и только на Train.")

    pictures = st.session_state.wf_pictures or {}
    selected_charts = summary.get("config", {}).get("output", {}).get("charts", [])
    available = [key for key in selected_charts if key in pictures]
    if available:
        st.subheader("Графики")
        columns = st.columns(2)
        for index, key in enumerate(available):
            with columns[index % 2]:
                with st.container(border=True):
                    st.markdown('<span class="chart-card-marker"></span>', unsafe_allow_html=True)
                    title = CHARTS[key]
                    if summary["n_classes"] > 2 and key in ("roc", "pr"):
                        title += " · каждый класс против остальных (OvR)"
                    st.markdown(f"**{title}**")
                    st.image(pictures[key], width="stretch")
                    st.caption(CHART_HELP[key])
    elif selected_charts:
        st.info("Для этого сохранённого запуска изображения не были записаны в историю. Метрики и таблицы доступны, но графики нельзя восстановить без повторного анализа.")
    for warning in st.session_state.wf_warnings:
        st.warning(warning)

    if st.session_state.wf_docx:
        st.download_button(
            "Скачать отчёт Word (.docx)",
            st.session_state.wf_docx,
            file_name=f"analysis_{run_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    elif st.session_state.wf_result_source == "history":
        st.caption("Для этого старого запуска файл DOCX не был сохранён в истории.")
    with st.expander("Настройки и воспроизводимость"):
        st.json({k: v for k, v in summary.items() if k not in ("_viz", "importance")})


def render_history() -> None:
    st.header("История запусков")
    rows = recent_runs()
    if not rows:
        st.info("Сохранённых запусков пока нет.")
        return
    table = pd.DataFrame(rows).rename(columns={
        "id": "ID",
        "created": "Дата UTC",
        "dataset": "Набор данных",
        "method": "Метод",
        "accuracy": "Accuracy",
        "f1": "F1 / F1 macro",
    })
    table["Метод"] = table["Метод"].map(lambda key: METHODS.get(key, {}).get("title", key))
    for column in ("Accuracy", "F1 / F1 macro"):
        table[column] = table[column].map(lambda value: None if value is None else round(float(value), 3))
    st.dataframe(table, hide_index=True, width="stretch")
    by_id = {row["id"]: row for row in rows}
    selected_id = st.selectbox(
        "Выберите запуск",
        list(by_id),
        format_func=lambda rid: f"{rid} · {METHODS.get(by_id[rid]['method'], {}).get('title', by_id[rid]['method'])} · {by_id[rid]['dataset']}",
    )
    open_col, restore_col = st.columns(2)
    if open_col.button("Открыть сохранённые результаты", type="primary", width="stretch"):
        saved = get_run(selected_id)
        if saved is None:
            st.error("Запуск не найден в локальной базе данных.")
        else:
            st.session_state.wf_last = saved["summary"]
            st.session_state.wf_pictures = saved["figures"]
            st.session_state.wf_warnings = []
            st.session_state.wf_docx = saved["report"]
            st.session_state.wf_run_id = saved["id"]
            st.session_state.wf_result_source = "history"
            st.session_state.wf_run_fingerprint = None
            st.session_state.wf_view = "Результаты"
            st.rerun()
    if restore_col.button("Восстановить настройки пайплайна", width="stretch"):
        saved = get_run(selected_id)
        if saved is None:
            st.error("Запуск не найден в локальной базе данных.")
        else:
            summary = saved["summary"]
            scheme = summary.get("pipeline_scheme")
            if not isinstance(scheme, dict):
                default = demo_state()
                scheme = {
                    "schema_version": 1,
                    "node_positions": {n.id: dict(n.position) for n in default.nodes},
                    "edges": [{"source": e.source, "target": e.target} for e in default.edges],
                    "settings": summary.get("config", default_settings()),
                }
            apply_scheme(scheme)
            dataset = summary.get("dataset", {})
            st.session_state.wf_expected_sha = dataset.get("sha256")
            if dataset.get("dataset_name") == "heart.csv" and DEMO.exists() and hashlib.sha256(DEMO.read_bytes()).hexdigest() == dataset.get("sha256"):
                st.session_state.wf_cfg["input"]["source"] = "Демонстрационный heart.csv"
            else:
                st.session_state.wf_cfg["input"]["source"] = "Загрузить свой CSV"
                st.session_state.wf_csv_bytes = None
                st.session_state.wf_csv_name = None
                st.session_state.wf_file_id = None
            st.session_state.wf_view = "Конструктор"
            st.rerun()
    st.caption("Открытие результатов не обучает модель. Восстановление настроек также не запускает анализ и не восстанавливает пользовательский CSV автоматически.")


boot()
apply_interface_style(st.session_state.wf_view)
settings = st.session_state.wf_cfg
try:
    df, info, source_bytes = read_dataset()
    data_error = ""
except (ValueError, OSError) as exc:
    df = info = source_bytes = None
    data_error = str(exc)

render_navigation()
if st.session_state.wf_view == "Конструктор":
    render_constructor(settings, df, info, data_error)
elif st.session_state.wf_view == "Результаты":
    render_results()
else:
    render_history()
