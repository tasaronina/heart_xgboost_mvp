"""MVP анализа табличных данных: Streamlit, XGBoost и история экспериментов."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from services.data_validation import UserDataError, read_csv_dataset
from services.report_service import predictions_csv, report_json
from services.storage import list_runs, save_run
from services.xgboost_service import RunConfig, run_classification


ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "sample_data" / "heart.csv"

SECTIONS = [
    "1. Данные",
    "2. Настройки",
    "3. Результаты",
    "4. История"
]

TARGET_NAMES = (
    "target",
    "Outcome",
    "HeartDisease",
    "diagnosis",
    "DEATH_EVENT"
)

DUP_OPTIONS = [
    "Оставить записи",
    "Пометить для проверки",
    "Исключить подтверждённые повторы"
]


st.set_page_config(
    page_title="MVP  Анализ табличных данных",
    layout="wide"
)

st.title("Анализ табличных данных - MVP")
st.caption("Бинарная классификация с XGBoost. Прототип дипломной системы")

def init_state() -> None:
    """Создание начального состояния приложения."""
    defaults = {
        "loaded_dataset": None,
        "dataset_source": "Пример: heart.csv",
        "custom_csv_bytes": None,
        "custom_csv_name": None,
        "last_run": None,
        "last_meta": None,
        "saved_form": {},
        "preferred_target": None,
        "missing_rules": {},
        "dup_action": DUP_OPTIONS[0],
        "dup_confirmed": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_for_new_dataset() -> None:
    """Сброс настроек при загрузке другого набора данных."""
    st.session_state.last_run = None
    st.session_state.last_meta = None
    st.session_state.saved_form = {}
    st.session_state.preferred_target = None
    st.session_state.missing_rules = {}
    st.session_state.dup_action = DUP_OPTIONS[0]
    st.session_state.dup_confirmed = False


def proposed_target(df: pd.DataFrame) -> str:
    """Предложение целевой переменной по распространённым названиям."""
    remembered = st.session_state.preferred_target

    if remembered in df.columns:
        return remembered

    return next(
        (name for name in TARGET_NAMES if name in df.columns),
        str(df.columns[-1])
    )


def token_matches(series: pd.Series, raw: str) -> pd.Series:
    """Предварительный подсчёт специальных значений в интерфейсе."""
    if not raw.strip():
        return pd.Series(False, index=series.index)

    if pd.api.types.is_numeric_dtype(series):
        try:
            value = float(raw.strip())
            if np.isfinite(value):
                return series.eq(value).fillna(False)
        except ValueError:
            pass

        return pd.Series(False, index=series.index)

    return (
        series.astype("string")
        .str.strip()
        .eq(raw.strip())
        .fillna(False)
    )


def show_data() -> None:
    """Первый раздел: загрузка и предварительная проверка данных."""
    st.header("Данные и проверка качества")

    choices = ["Пример: heart.csv", "Загрузить свой CSV"]

    source = st.radio(
        "Источник данных",
        choices,
        index=choices.index(st.session_state.dataset_source),
        horizontal=True,
    )

    st.session_state.dataset_source = source

    if source == choices[0]:
        data = DEMO.read_bytes()
        name = "heart.csv"
        st.caption("Встроенный демонстрационный набор данных.")
    else:
        upload = st.file_uploader(
            "Выберите CSV",
            type=["csv"],
            help="Размер файла для этого MVP — до 15 МБ."
        )

        if upload is not None:
            st.session_state.custom_csv_bytes = upload.getvalue()
            st.session_state.custom_csv_name = upload.name

        data = st.session_state.custom_csv_bytes
        name = st.session_state.custom_csv_name

        if upload is None and data is not None:
            st.caption(
                f"Активен ранее загруженный файл: {name}. "
                "Для замены выберите новый CSV."
            )

        if not data:
            st.info("Загрузите CSV для продолжения.")
            return

    try:
        loaded = read_csv_dataset(data, name)
    except UserDataError as exc:
        st.error(str(exc))
        return

    prev = st.session_state.loaded_dataset

    if (
        prev is None
        or prev.sha256 != loaded.sha256
        or prev.name != loaded.name
    ):
        reset_for_new_dataset()

    st.session_state.loaded_dataset = loaded
    df = loaded.frame

    a, b, c, d = st.columns(4)

    a.metric("Строк", len(df))
    b.metric("Столбцов", len(df.columns))
    c.metric("Пустых ячеек", int(df.isna().sum().sum()))
    d.metric("Повторных строк", int(df.duplicated().sum()))

    st.subheader("Предпросмотр")

    st.dataframe(
        df.head(15),
        hide_index=True,
        width="stretch"
    )

    st.subheader("Характеристики столбцов")

    st.dataframe(
        pd.DataFrame({
            "Столбец": df.columns,
            "Технический тип": df.dtypes.astype(str).values,
            "Пустых ячеек": df.isna().sum().values,
            "Уникальных значений": df.nunique(dropna=False).values,
        }),
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Технический тип (int64, float64 и т. д.) "
        "не определяет медицинский смысл признака."
    )

    st.subheader("Распределение целевой переменной")

    names = list(df.columns)
    preferred = proposed_target(df)

    target = st.selectbox(
        "Что хотим предсказывать?",
        names,
        index=names.index(preferred)
    )

    st.session_state.preferred_target = target

    nunique = df[target].dropna().nunique()

    if nunique == 2:
        counts = (
            df[target]
            .astype("string")
            .fillna("(пропуск)")
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            counts,
            x_label="Класс",
            y_label="Количество строк"
        )

        st.caption(
            "График показывает доли классов до обучения "
            "и помогает заметить дисбаланс."
        )
    else:
        st.info(
            f"Выбрано значений класса: {nunique}. "
            "Для этого MVP требуется ровно два класса."
        )

    if df[target].isna().any():
        st.warning(
            "В целевой переменной есть пропуски. "
            "Текущий MVP не выполняет обучение при их наличии."
        )

    st.subheader("Проверка качества данных")

    with st.expander(
        "Особые значения и правила пропусков",
        expanded=False
    ):
        st.write(
            "Значения вроде 0, -999 или 'unknown' могут обозначать "
            "отсутствие измерения, но только если это подтверждено "
            "описанием вашего набора данных. "
            "Нули и другие числа автоматически не объявляются ошибками."
        )

        available = [c for c in df.columns if c != target]

        if not available:
            st.info("Нет доступных столбцов для настройки.")
        else:
            column = st.selectbox(
                "Столбец для правила",
                available
            )

            saved_tokens = st.session_state.missing_rules.get(
                column, []
            )

            with st.form(
                f"rule_form_{loaded.sha256[:10]}_{column}"
            ):
                raw = st.text_input(
                    "Какие значения считать пропусками?",
                    value="; ".join(saved_tokens),
                    help=(
                        "Разделяйте значения точкой с запятой, "
                        "например: 0; -999. "
                        "Обычные пустые ячейки pandas уже "
                        "распознаёт как пропуски."
                    )
                )

                apply_rule = st.form_submit_button(
                    "Сохранить правило"
                )

            if apply_rule:
                tokens = list(
                    dict.fromkeys(
                        t.strip()
                        for t in raw.split(";")
                        if t.strip()
                    )
                )

                new_rules = dict(
                    st.session_state.missing_rules
                )

                if tokens:
                    new_rules[column] = tokens
                else:
                    new_rules.pop(column, None)

                st.session_state.missing_rules = new_rules

                st.success(
                    "Правило сохранено для текущего датасета."
                )

            rules = st.session_state.missing_rules

            if column in rules:
                mask = pd.Series(
                    False,
                    index=df.index
                )

                for token in rules[column]:
                    mask |= token_matches(
                        df[column],
                        token
                    )

                st.caption(
                    f"По правилу для столбца «{column}» "
                    f"будет отмечено значений: {int(mask.sum())}."
                )

            if rules:
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Столбец": c,
                            "Считать пропусками": "; ".join(tokens)
                        }
                        for c, tokens in rules.items()
                    ]),
                    hide_index=True,
                    width="stretch"
                )

            st.caption(
                "Правила применяются при запуске к выбранным "
                "входным признакам X. Исходный CSV не меняется. "
                "Заполнение пропусков обучается только на Train."
            )

    count = int(df.duplicated().sum())

    with st.expander(
        f"Совпадающие строки: {count}",
        expanded=False
    ):
        st.write(
            "Полные совпадения не обязательно означают "
            "ошибку выгрузки: это могут быть разные наблюдения "
            "с одинаковыми показателями."
        )

        if count:
            equal_rows = df.loc[
                df.duplicated(keep=False)
            ].copy()

            equal_rows.insert(
                0,
                "Строка CSV (с заголовком)",
                equal_rows.index + 2
            )

            st.dataframe(
                equal_rows.head(40),
                hide_index=True,
                width="stretch"
            )

            current = st.session_state.dup_action

            new_choice = st.radio(
                "Решение исследователя",
                DUP_OPTIONS,
                index=DUP_OPTIONS.index(current)
            )

            if new_choice != current:
                st.session_state.dup_confirmed = False

            st.session_state.dup_action = new_choice

            if new_choice == DUP_OPTIONS[2]:
                st.session_state.dup_confirmed = st.checkbox(
                    "Подтверждаю исключение повторных строк "
                    "при следующем анализе",
                    value=st.session_state.dup_confirmed,
                )

                st.warning(
                    "Исходный файл не изменится. "
                    "Решение повлияет только на выборку, "
                    "используемую для нового запуска."
                )

            elif new_choice == DUP_OPTIONS[1]:
                st.info(
                    "Строки помечены для проверки в текущем сеансе, "
                    "но пока не исключаются из анализа."
                )

        else:
            st.success(
                "Полных совпадений строк не найдено."
            )

    st.success(
        "Теперь можно открыть «2. Настройки» в боковом меню."
    )


def show_settings() -> None:
    """Второй раздел: настройка и запуск XGBoost."""
    st.header("Настройка XGBoost")

    loaded = st.session_state.loaded_dataset

    if loaded is None:
        st.info(
            "Сначала загрузите таблицу в разделе «1. Данные»."
        )
        return

    df = loaded.frame

    st.caption(
        f"Активный файл: {loaded.name}, {len(df)} строк"
    )

    st.write(
        "**Схема анализа:** CSV → признаки → Train/Test → "
        "обработка пропусков → XGBoost → метрики"
    )

    saved = st.session_state.saved_form

    target_default = (
        saved.get("target")
        or proposed_target(df)
    )

    target = st.selectbox(
        "Целевой признак Y",
        list(df.columns),
        index=(
            list(df.columns).index(target_default)
            if target_default in df.columns
            else len(df.columns) - 1
        )
    )

    classes = sorted(
        list(pd.unique(df[target].dropna())),
        key=str
    )

    if len(classes) != 2:
        st.error(
            "Для бинарной классификации целевой признак "
            "должен иметь ровно два класса."
        )
        return

    if df[target].isna().any():
        st.error(
            "В целевом признаке есть пропуски. "
            "Сначала проверьте исходные данные."
        )
        return

    old_pos = (
        saved.get("positive_label")
        if saved.get("target") == target
        else None
    )

    default_pos = (
        classes.index(old_pos)
        if old_pos in classes
        else next(
            (
                i for i, item in enumerate(classes)
                if str(item) == "1"
            ),
            len(classes) - 1
        )
    )

    positive = st.selectbox(
        "Положительный класс (для Precision / Recall / F1)",
        classes,
        index=default_pos,
        help=(
            "Выберите исследуемое событие по описанию набора. "
            "Программа не знает медицинский смысл классов."
        )
    )

    feature_options = [
        c for c in df.columns
        if c != target
    ]

    same_target = saved.get("target") == target

    previous_features = (
        saved.get("features", feature_options)
        if same_target
        else feature_options
    )

    features = st.multiselect(
        "Признаки X",
        feature_options,
        default=[
            c for c in previous_features
            if c in feature_options
        ],
        help=(
            "По умолчанию предложены все столбцы, кроме Y. "
            "Выбор подтверждает пользователь."
        )
    )

    suggestion = [
        c for c in features
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    old_cat = (
        saved.get("categorical", suggestion)
        if same_target
        else suggestion
    )

    categorical = st.multiselect(
        "Категориальные признаки (One-Hot)",
        features,
        default=[
            c for c in old_cat
            if c in features
        ],
        help=(
            "Задайте столбцы с категориями или числовыми "
            "кодами категорий. Измеряемые количественные "
            "величины сюда не включайте."
        )
    )

    many_categories = [
        c for c in categorical
        if df[c].nunique() > 12
    ]

    if many_categories:
        st.warning(
            "Проверьте выбранные категории с большим "
            "числом уникальных значений: "
            + ", ".join(many_categories)
        )

    with st.expander(
        "Зачем отмечать категориальные признаки?"
    ):
        st.write(
            "Коды групп не обязаны быть количественной шкалой. "
            "One-Hot создаёт отдельные индикаторы категорий. "
            "Для обычных числовых признаков пропуски "
            "заполняются медианой, а для категорий — "
            "наиболее частым значением. "
            "Параметры заполнения определяются только "
            "по обучающей выборке."
        )

    st.subheader("Параметры эксперимента")

    left, right = st.columns(2)

    with left:
        test_size = st.slider(
            "Доля тестовой выборки",
            0.1,
            0.4,
            float(saved.get("test_size", 0.2)),
            0.05,
            help=(
                "Например, 0.20 означает 20% "
                "для проверки качества."
            )
        )

    with right:
        seed = st.number_input(
            "random_state",
            0,
            100000,
            int(saved.get("random_state", 42)),
            help=(
                "Позволяет воспроизвести "
                "случайное разделение."
            )
        )

    duplicate_count = int(df.duplicated().sum())

    remove_dupes = (
        duplicate_count > 0
        and st.session_state.dup_action == DUP_OPTIONS[2]
        and st.session_state.dup_confirmed
    )

    if duplicate_count:
        st.caption(
            f"Полных повторов: {duplicate_count}. "
            f"Решение: {st.session_state.dup_action}."
        )

    if (
        duplicate_count
        and st.session_state.dup_action == DUP_OPTIONS[2]
        and not remove_dupes
    ):
        st.warning(
            "Для удаления повторов вернитесь в раздел "
            "«1. Данные» и подтвердите решение."
        )

    rules = {
        c: tokens
        for c, tokens in st.session_state.missing_rules.items()
        if c in features
    }

    if rules:
        st.caption(
            "Правила пропусков: "
            + "; ".join(
                f"{c}: {', '.join(v)}"
                for c, v in rules.items()
            )
        )

    st.subheader("Параметры XGBoost")

    a, b, c = st.columns(3)

    with a:
        n_estimators = st.slider(
            "Количество деревьев",
            20,
            400,
            int(saved.get("n_estimators", 100)),
            10,
            help=(
                "Больше деревьев — больше этапов "
                "корректировки модели."
            )
        )

        max_depth = st.slider(
            "Глубина дерева",
            1,
            10,
            int(saved.get("max_depth", 3)),
            help=(
                "Большая глубина позволяет описывать "
                "более сложные зависимости, "
                "но может увеличить переобучение."
            )
        )

    with b:
        learning_rate = st.slider(
            "Скорость обучения",
            0.01,
            0.30,
            float(saved.get("learning_rate", 0.05)),
            0.01,
            help=(
                "Насколько сильно каждое новое "
                "дерево меняет модель."
            )
        )

        subsample = st.slider(
            "Доля объектов на дерево",
            0.5,
            1.0,
            float(saved.get("subsample", 0.8)),
            0.05,
            help=(
                "Какая доля обучающих строк случайно "
                "используется каждым деревом."
            )
        )

    with c:
        colsample_bytree = st.slider(
            "Доля признаков на дерево",
            0.5,
            1.0,
            float(saved.get("colsample_bytree", 0.8)),
            0.05,
            help=(
                "Какая доля входных признаков "
                "доступна дереву."
            )
        )

    st.caption(
        "Начальные значения — отправная точка, "
        "а не универсально лучший набор настроек."
    )

    st.session_state.saved_form = {
        "target": target,
        "positive_label": positive,
        "features": list(features),
        "categorical": list(categorical),
        "test_size": float(test_size),
        "random_state": int(seed),
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
    }

    blocked = (
        duplicate_count > 0
        and st.session_state.dup_action == DUP_OPTIONS[2]
        and not st.session_state.dup_confirmed
    )

    if st.button(
        "Запустить анализ",
        type="primary",
        disabled=blocked,
        width="stretch"
    ):
        cfg = RunConfig(
            target_column=target,
            positive_label=positive,
            feature_columns=list(features),
            categorical_columns=list(categorical),
            test_size=round(float(test_size), 2),
            random_state=int(seed),
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=round(float(learning_rate), 3),
            subsample=round(float(subsample), 2),
            colsample_bytree=round(float(colsample_bytree), 2),
            remove_duplicates=bool(remove_dupes),
            duplicate_action={
                DUP_OPTIONS[0]: "keep",
                DUP_OPTIONS[1]: "review",
                DUP_OPTIONS[2]: "remove"
            }[st.session_state.dup_action]
            if duplicate_count else "keep",
            missing_values=rules,
        )

        try:
            with st.spinner(
                "Обучаем XGBoost и рассчитываем метрики..."
            ):
                result = run_classification(df, cfg)

                run_id = save_run(
                    result,
                    loaded.name,
                    loaded.sha256
                )

            st.session_state.last_run = result

            st.session_state.last_meta = {
                "dataset_name": loaded.name,
                "sha256": loaded.sha256,
                "run_id": run_id
            }

            st.success(
                f"Анализ завершён. Запуск {run_id} сохранён."
            )

            st.info(
                "Откройте «3. Результаты» в боковом меню."
            )

        except UserDataError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(
                "Техническая ошибка выполнения анализа."
            )
            st.exception(exc)


def show_results() -> None:
    """Третий раздел: результаты анализа и графики."""
    st.header("Результаты")

    result = st.session_state.last_run
    meta = st.session_state.last_meta

    if result is None or meta is None:
        st.info(
            "Сначала выполните анализ на шаге «2. Настройки»."
        )
        return

    st.caption(
        f"Набор: {meta['dataset_name']} · "
        f"Запуск: {meta['run_id']} · "
        f"Положительный класс: "
        f"{result['config']['positive_label']}"
    )

    m = result["metrics"]
    cm = np.asarray(result["confusion_matrix"])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    specificity = m.get(
        "specificity",
        tn / (tn + fp) if tn + fp else 0
    )

    metrics = [
        (
            "Accuracy",
            m["accuracy"],
            "Доля всех правильных классификаций"
        ),
        (
            "Precision",
            m["precision"],
            "Доля верных положительных среди предсказанных положительных"
        ),
        (
            "Recall",
            m["recall"],
            "Доля реальных положительных случаев, которые модель обнаружила"
        ),
        (
            "Specificity",
            specificity,
            "Доля реальных отрицательных случаев, определённых правильно"
        ),
        (
            "F1",
            m["f1"],
            "Гармоническое среднее Precision и Recall"
        ),
        (
            "ROC AUC",
            m["roc_auc"],
            "Оценка способности модели ранжировать два класса"
        ),
    ]

    for group in [metrics[:3], metrics[3:]]:
        cols = st.columns(3)

        for col, (name, value, desc) in zip(cols, group):
            col.metric(
                name,
                f"{value:.3f}",
                help=desc
            )

    st.caption(
        f"На тестовой выборке обнаружено {tp} из {tp + fn} "
        f"положительных случаев; пропущено {fn}. "
        f"Ложных положительных прогнозов: {fp}."
    )

    with st.expander(
        "Как интерпретировать метрики и этот результат?"
    ):
        st.write(
            f"**Accuracy {m['accuracy']:.1%}:** "
            "правильно классифицировано наблюдений "
            "от общего числа тестовых строк."
        )

        st.write(
            f"**Precision {m['precision']:.1%}:** "
            "среди предсказанных положительных случаев "
            f"верными оказались {tp} из {tp + fp}."
        )

        st.write(
            f"**Recall {m['recall']:.1%}:** "
            f"найдено {tp} из {tp + fn} "
            "реальных положительных случаев."
        )

        st.write(
            f"**Specificity {specificity:.1%}:** "
            f"правильно определено {tn} из {tn + fp} "
            "отрицательных случаев."
        )

        st.write(
            "**F1** объединяет Precision и Recall. "
            "**ROC AUC** измеряет различение классов "
            "по оценкам модели, а не процент правильных ответов."
        )

        st.caption(
            "Приемлемость метрик зависит от задачи исследования, "
            "соотношения классов и последствий ложных прогнозов. "
            "Одна тестовая выборка не доказывает пригодность "
            "модели для клинического использования."
        )

    sample = result["sample"]

    st.caption(
        f"Train: {sample['train_rows']} · "
        f"Test: {sample['test_rows']} · "
        f"Исключено повторных строк: "
        f"{sample['duplicate_rows_removed']}"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Матрица ошибок")

        fig, ax = plt.subplots(
            figsize=(5, 4.3)
        )

        image = ax.imshow(
            cm,
            cmap="Blues"
        )

        for (i, j), value in np.ndenumerate(cm):
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > cm.max() / 2
                    else "black"
                ),
                fontsize=13
            )

        ax.set_xticks(
            [0, 1],
            ["Отрицательный", "Положительный"]
        )

        ax.set_yticks(
            [0, 1],
            ["Отрицательный", "Положительный"]
        )

        ax.set_xlabel("Прогноз")
        ax.set_ylabel("Настоящий класс")

        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046
        )

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        st.caption(
            "Слева сверху — TN; справа сверху — FP; "
            "слева снизу — FN; справа снизу — TP."
        )

    with right:
        st.subheader("ROC-кривая")

        fig, ax = plt.subplots(
            figsize=(5, 4.3)
        )

        ax.plot(
            result["roc"]["fpr"],
            result["roc"]["tpr"],
            label=f"ROC AUC = {m['roc_auc']:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            "--",
            color="gray",
            label="Случайное ранжирование"
        )

        ax.set_xlabel(
            "Доля ложноположительных (FPR)"
        )

        ax.set_ylabel(
            "Чувствительность (TPR)"
        )

        ax.grid(alpha=0.15)
        ax.legend(loc="lower right")

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)

        st.caption(
            "Различение классов при разных порогах классификации."
        )

    st.subheader(
        "Значимость признаков (gain)"
    )

    importance = pd.DataFrame(
        result["feature_importance"]
    )

    if not importance.empty:
        active = importance.loc[
            importance["importance"] > 0
        ]

        if not active.empty:
            top = (
                active.head(12)
                .sort_values("importance")
            )

            st.bar_chart(
                top.set_index("feature")["importance"],
                horizontal=True,
                x_label="Нормированная значимость (gain)"
            )

        else:
            st.info(
                "Ненулевые значения значимости отсутствуют."
            )

        hidden = (
            len(importance) - len(active)
        )

        if hidden:
            st.caption(
                f"Скрыто признаков с нулевой значимостью: {hidden}."
            )

    st.caption(
        "Это показатель важности разбиений XGBoost, "
        "не SHAP и не доказательство "
        "причинно-следственной связи."
    )

    with st.expander(
        "Предсказания тестовой выборки"
    ):
        st.dataframe(
            pd.DataFrame(result["predictions"]),
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "В колонках actual и predicted значение 1 "
            "означает положительный класс, "
            "а 0 — отрицательный. "
            "Порог предсказания по умолчанию 0.5."
        )

    with st.expander(
        "Настройки и воспроизводимость"
    ):
        st.json({
            "config": result["config"],
            "sample": sample,
            "versions": result["versions"]
        })

        st.write(
            result["methodology"]
        )

    a, b = st.columns(2)

    with a:
        st.download_button(
            "Скачать отчёт JSON",
            data=report_json(
                result,
                meta["dataset_name"],
                meta["sha256"]
            ),
            file_name=f"report_{meta['run_id']}.json",
            mime="application/json",
            width="stretch"
        )

    with b:
        st.download_button(
            "Скачать прогнозы CSV",
            data=predictions_csv(result),
            file_name=f"predictions_{meta['run_id']}.csv",
            mime="text/csv",
            width="stretch"
        )


def show_history() -> None:
    """Четвёртый раздел: история выполненных экспериментов."""
    st.header("История экспериментов")

    runs = list_runs()

    if not runs:
        st.info(
            "Сохранённых запусков пока нет."
        )
        return

    summary = []

    for item in runs:
        r = item["result"]

        summary.append({
            "ID": item["run_id"],
            "Время (UTC)": item["created_at"][:19],
            "Набор": item["dataset_name"],
            "max_depth": r["config"]["max_depth"],
            "Accuracy": round(
                r["metrics"]["accuracy"],
                3
            ),
            "Recall": round(
                r["metrics"]["recall"],
                3
            ),
            "ROC AUC": round(
                r["metrics"]["roc_auc"],
                3
            ),
        })

    st.dataframe(
        pd.DataFrame(summary),
        hide_index=True,
        width="stretch"
    )

    chosen = st.selectbox(
        "Выберите запуск",
        [r["run_id"] for r in runs]
    )

    item = next(
        r for r in runs
        if r["run_id"] == chosen
    )

    st.json({
        "config": item["result"]["config"],
        "metrics": item["result"]["metrics"],
        "sample": item["result"]["sample"]
    })

    if st.button(
        "Открыть результаты выбранного запуска"
    ):
        st.session_state.last_run = item["result"]

        st.session_state.last_meta = {
            "dataset_name": item["dataset_name"],
            "sha256": item["dataset_sha256"],
            "run_id": chosen
        }

        st.success(
            "Теперь перейдите в раздел «3. Результаты»."
        )

    st.caption(
        "SQLite хранит настройки, метрики и прогнозы, "
        "но не исходные CSV. В этом MVP нет учётных "
        "записей и разграничения доступа."
    )


init_state()

with st.sidebar:
    st.subheader("Навигация")

    section = st.radio(
        "Раздел",
        SECTIONS,
        label_visibility="collapsed"
    )

    st.divider()

    if st.session_state.loaded_dataset is not None:
        st.caption(
            f"Текущий файл: "
            f"{st.session_state.loaded_dataset.name}"
        )
    else:
        st.caption(
            "Набор данных ещё не выбран"
        )

    st.caption(
        "Пока реализован один сценарий: "
        "бинарная классификация XGBoost."
    )


if section == SECTIONS[0]:
    show_data()

elif section == SECTIONS[1]:
    show_settings()

elif section == SECTIONS[2]:
    show_results()

else:
    show_history()
