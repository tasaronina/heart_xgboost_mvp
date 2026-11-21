"""Pure graph editing reducer for the visual pipeline.

No Streamlit or frontend imports. React sends *intent*, Python owns the canonical
nodes and connections. Analytical validity is checked separately by spec.py.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from math import isfinite
from typing import Any

from .graph_ops import connect_or_replace, disconnect_link


class GraphEditError(ValueError):
    """The requested graph edit is invalid or stale."""


def _safe_xy(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise GraphEditError("Некорректные координаты блока.")
    try:
        x = float(value["x"])
        y = float(value["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GraphEditError("Некорректные координаты блока.") from exc
    if not (isfinite(x) and isfinite(y) and abs(x) <= 10000 and abs(y) <= 10000):
        raise GraphEditError("Координаты блока вне допустимого диапазона.")
    return {"x": x, "y": y}


def normalize_graph(graph: Mapping[str, Any], block_ids: Iterable[str]) -> dict[str, list[dict]]:
    """Normalize and validate the graph independently of canvas widgets."""
    allowed = set(block_ids)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise GraphEditError("Некорректный формат схемы.")
    if len(nodes) > len(allowed) or len(edges) > len(allowed) * 2:
        raise GraphEditError("В схеме слишком много элементов.")
    normalized_nodes: list[dict] = []
    seen: set[str] = set()
    for entry in nodes:
        if not isinstance(entry, Mapping):
            raise GraphEditError("Некорректный блок.")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or identifier not in allowed or identifier in seen:
            raise GraphEditError("Неизвестный или повторяющийся блок.")
        seen.add(identifier)
        normalized_nodes.append({"id": identifier, "position": _safe_xy(entry.get("position"))})
    normalized_edges: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    for item in edges:
        if not isinstance(item, Mapping):
            raise GraphEditError("Некорректная стрелка.")
        source, target = item.get("source"), item.get("target")
        if source not in seen or target not in seen or source == target:
            raise GraphEditError("Стрелка указывает на отсутствующий блок либо на саму себя.")
        pair = (source, target)
        if pair in seen_edges:
            continue
        normalized_edges.append({"source": source, "target": target})
        seen_edges.add(pair)
    return {"nodes": normalized_nodes, "edges": normalized_edges}


def apply_graph_event(
    graph: Mapping[str, Any], event: Mapping[str, Any], block_ids: Iterable[str],
    default_positions: Mapping[str, Mapping[str, float]],
) -> tuple[dict, str | None, bool]:
    """Apply one frontend intention; return (graph, selected_node, was_changed).

    Selected node is meaningful only for add_node / select_node. For every other
    action it is None; caller may keep its previously selected node if present.
    """
    allowed = tuple(block_ids)
    current = normalize_graph(graph, allowed)
    new = deepcopy(current)
    action = event.get("type")
    payload = event.get("payload", {})
    if not isinstance(payload, Mapping):
        raise GraphEditError("Некорректный формат действия редактора.")
    present = {item["id"] for item in new["nodes"]}
    selected: str | None = None
    pairs = [(e["source"], e["target"]) for e in new["edges"]]

    if action == "select_node":
        identifier = payload.get("id")
        if identifier not in present:
            raise GraphEditError("Выбранный блок отсутствует в схеме.")
        return new, identifier, False

    if action == "add_node":
        identifier = payload.get("id")
        if identifier not in allowed:
            raise GraphEditError("Неизвестный тип блока.")
        if identifier in present:
            return new, identifier, False
        new["nodes"].append({"id": identifier, "position": _safe_xy(default_positions[identifier])})
        selected = identifier

    elif action == "delete_elements":
        identifiers = payload.get("nodeIds", [])
        links = payload.get("links", [])
        if not isinstance(identifiers, list) or any(i not in allowed for i in identifiers):
            raise GraphEditError("Некорректные удаляемые блоки.")
        if not isinstance(links, list) or any(not isinstance(e, Mapping) for e in links):
            raise GraphEditError("Некорректные удаляемые соединения.")
        removed = set(identifiers)
        removed_links = {(e.get("source"), e.get("target")) for e in links}
        new["nodes"] = [n for n in new["nodes"] if n["id"] not in removed]
        new["edges"] = [e for e in new["edges"] if e["source"] not in removed
                        and e["target"] not in removed
                        and (e["source"], e["target"]) not in removed_links]

    elif action == "remove_nodes":
        identifiers = payload.get("ids")
        if not isinstance(identifiers, list) or any(i not in allowed for i in identifiers):
            raise GraphEditError("Некорректный список удаляемых блоков.")
        removed = set(identifiers)
        new["nodes"] = [n for n in new["nodes"] if n["id"] not in removed]
        new["edges"] = [e for e in new["edges"] if e["source"] not in removed and e["target"] not in removed]

    elif action == "disconnect_many":
        links = payload.get("links")
        if not isinstance(links, list):
            raise GraphEditError("Некорректный список удаляемых стрелок.")
        to_remove = {(e.get("source"),e.get("target")) for e in links if isinstance(e, Mapping)}
        new["edges"] = [e for e in new["edges"] if (e["source"],e["target"]) not in to_remove]

    elif action == "remove_node":
        identifier = payload.get("id")
        if identifier not in present:
            return new, None, False
        new["nodes"] = [n for n in new["nodes"] if n["id"] != identifier]
        new["edges"] = [e for e in new["edges"] if identifier not in (e["source"], e["target"])]

    elif action == "move_node":
        identifier = payload.get("id")
        if identifier not in present:
            raise GraphEditError("Перемещаемый блок отсутствует.")
        next_xy = _safe_xy(payload.get("position"))
        for item in new["nodes"]:
            if item["id"] == identifier:
                item["position"] = next_xy
                break

    elif action in ("connect", "reconnect"):
        source, target = payload.get("source"), payload.get("target")
        if action == "reconnect":
            old = (payload.get("oldSource"), payload.get("oldTarget"))
            if old not in pairs:
                raise GraphEditError("Исходная стрелка уже удалена. Обновите схему.")
            pairs = disconnect_link(pairs, *old)
        try:
            pairs = connect_or_replace(pairs, source, target, present)
        except (ValueError, TypeError) as exc:
            raise GraphEditError(str(exc)) from exc
        new["edges"] = [{"source": a, "target": b} for a, b in pairs]

    elif action == "disconnect":
        try:
            pairs = disconnect_link(pairs, payload.get("source"), payload.get("target"))
        except (ValueError, TypeError) as exc:
            raise GraphEditError(str(exc)) from exc
        new["edges"] = [{"source": a, "target": b} for a, b in pairs]

    else:
        raise GraphEditError(f"Неизвестное действие редактора: {action!r}.")

    changed = new != current
    return new, selected, changed

