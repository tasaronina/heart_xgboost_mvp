"""Regression tests for all semantic graph actions (no Streamlit required)."""
import pytest
from workflow_mvp.editor_logic import apply_graph_event, GraphEditError, normalize_graph

BLOCKS = ("input", "preprocess", "analysis", "output")
POSITIONS = {name: {"x": i * 265, "y": 165} for i, name in enumerate(BLOCKS)}


def event(action, **payload):
    return {"type": action, "payload": payload}


def apply(graph, action, **payload):
    return apply_graph_event(graph, event(action, **payload), BLOCKS, POSITIONS)


def blank():
    return {"nodes": [], "edges": []}


def full():
    graph = blank()
    for name in BLOCKS:
        graph, _, _ = apply(graph, "add_node", id=name)
    for a, b in zip(BLOCKS, BLOCKS[1:]):
        graph, _, _ = apply(graph, "connect", source=a, target=b)
    return graph


def pairs(graph):
    return [(e["source"], e["target"]) for e in graph["edges"]]


def test_full_graph_and_consistent_edges():
    graph = full()
    assert len(graph["nodes"]) == 4
    assert pairs(graph) == list(zip(BLOCKS, BLOCKS[1:]))


def test_delete_node_and_add_back_preserves_other_links():
    graph = full()
    graph, _, _ = apply(graph, "remove_node", id="preprocess")
    assert "preprocess" not in [node["id"] for node in graph["nodes"]]
    assert pairs(graph) == [("analysis", "output")]
    graph, selected, changed = apply(graph, "add_node", id="preprocess")
    assert changed and selected == "preprocess"
    assert len(graph["nodes"]) == 4


def test_click_edge_and_disconnect():
    graph = full()
    graph, _, changed = apply(graph, "disconnect", source="input", target="preprocess")
    assert changed
    assert ("input", "preprocess") not in pairs(graph)


def test_reconnect_dragged_end_replaces_conflicting_link():
    graph = full()
    graph, _, changed = apply(graph, "reconnect", oldSource="input", oldTarget="preprocess",
                              source="input", target="analysis")
    assert changed
    assert pairs(graph) == [("analysis", "output"), ("input", "analysis")]


def test_user_may_draw_wrong_route_but_preflight_would_reject_it():
    graph = full()
    graph, _, changed = apply(graph, "connect", source="output", target="preprocess")
    assert changed
    assert ("output", "preprocess") in pairs(graph)


def test_move_node_without_changing_other_data():
    graph = full()
    graph, _, changed = apply(graph, "move_node", id="analysis", position={"x": 333.4, "y": 120})
    assert changed
    n = next(item for item in graph["nodes"] if item["id"] == "analysis")
    assert n["position"] == {"x": 333.4, "y": 120}
    assert len(graph["edges"]) == 3


def test_bulk_delete_is_atomic():
    graph = full()
    graph, _, _ = apply(graph, "remove_nodes", ids=["input", "preprocess"])
    assert [n["id"] for n in graph["nodes"]] == ["analysis", "output"]
    assert pairs(graph) == [("analysis", "output")]
    graph, _, _ = apply(graph, "disconnect_many", links=[{"source": "analysis", "target": "output"}])
    assert pairs(graph) == []


def test_selection_has_no_graph_mutation():
    graph = full()
    next_graph, selected, changed = apply(graph, "select_node", id="analysis")
    assert next_graph == graph
    assert selected == "analysis"
    assert changed is False


def test_no_unknown_blocks_or_duplicate_nodes():
    with pytest.raises(GraphEditError):
        apply(blank(), "add_node", id="arbitrary")
    graph = full()
    same, _, changed = apply(graph, "add_node", id="input")
    assert same == graph and not changed
    with pytest.raises(GraphEditError):
        normalize_graph({"nodes": [{"id":"input","position":POSITIONS["input"]}]*2,
                         "edges":[]},BLOCKS)


def test_no_self_loop_or_stale_reconnection():
    graph = full()
    with pytest.raises(GraphEditError):
        apply(graph, "connect", source="analysis", target="analysis")
    with pytest.raises(GraphEditError):
        apply(graph, "reconnect", oldSource="input", oldTarget="output", source="input", target="analysis")


def test_coordinates_reject_nan_and_infinite():
    graph = full()
    for xy in ({"x":float("nan"),"y":1}, {"x":float("inf"),"y":0}, {"x":200000,"y":0}):
        with pytest.raises(GraphEditError):
            apply(graph,"move_node",id="input",position=xy)


def test_mixed_node_and_edge_delete_is_atomic():
    graph = full()
    graph, _, changed = apply(graph, "delete_elements", nodeIds=["preprocess"],
                              links=[{"source":"analysis", "target":"output"}])
    assert changed
    assert pairs(graph) == []
    assert "preprocess" not in [n["id"] for n in graph["nodes"]]

