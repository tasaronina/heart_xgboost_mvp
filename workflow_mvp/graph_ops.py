"""Операции над соединениями пайплайна без зависимости от Streamlit.

Редактор и панель ручного управления используют один и тот же список стрелок.
Валидность итогового *аналитического* маршрута проверяется отдельно в spec.
"""
from __future__ import annotations

from collections.abc import Iterable

Edge = tuple[str, str]


def connect_or_replace(
    edges: Iterable[Edge], source: str, target: str, available_nodes: Iterable[str]
) -> list[Edge]:
    """Соединить две вершины, заменив их конфликтующие связи.

    Один узел имеет не более одного исходящего и одного входящего соединения.
    Другие соединения не меняются. Неправильный для обучения порядок можно
    нарисовать, но проверка пайплайна не разрешит его запуск.
    """
    present = set(available_nodes)
    if source not in present or target not in present:
        raise ValueError("Для соединения оба блока должны присутствовать на схеме.")
    if source == target:
        raise ValueError("Нельзя соединить блок сам с собой.")
    unique = list(dict.fromkeys(tuple(pair) for pair in edges))
    remaining = [(a, b) for a, b in unique if a != source and b != target]
    return [*remaining, (source, target)]


def disconnect_link(edges: Iterable[Edge], source: str, target: str) -> list[Edge]:
    """Удалить выбранное соединение и оставить остальные без изменений."""
    unique = list(dict.fromkeys(tuple(pair) for pair in edges))
    if (source, target) not in unique:
        raise ValueError("Этой стрелки больше нет в текущей схеме.")
    return [pair for pair in unique if pair != (source, target)]
