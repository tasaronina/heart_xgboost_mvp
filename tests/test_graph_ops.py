"""Независимые регрессионные тесты управления стрелками."""
import pytest

from workflow_mvp.graph_ops import connect_or_replace, disconnect_link

BLOCKS = ("input", "preprocess", "analysis", "output")
EXPECTED = [("input", "preprocess"), ("preprocess", "analysis"), ("analysis", "output")]


def test_connect_from_blank_and_disconnect():
    links = connect_or_replace([], "input", "preprocess", BLOCKS)
    assert links == [("input", "preprocess")]
    assert disconnect_link(links, "input", "preprocess") == []


def test_replace_source_and_destination_preserves_unrelated():
    links = connect_or_replace(EXPECTED, "input", "analysis", BLOCKS)
    assert links == [("analysis", "output"), ("input", "analysis")]


def test_reconnecting_restores_valid_route():
    links = []
    for a, b in EXPECTED:
        links = connect_or_replace(links, a, b, BLOCKS)
    assert links == EXPECTED
    assert connect_or_replace(links, "input", "preprocess", BLOCKS) == [
        ("preprocess", "analysis"), ("analysis", "output"), ("input", "preprocess")
    ]


def test_reject_missing_and_self_connection():
    with pytest.raises(ValueError, match="должны присутствовать"):
        connect_or_replace([], "missing", "analysis", BLOCKS)
    with pytest.raises(ValueError, match="сам с собой"):
        connect_or_replace([], "analysis", "analysis", BLOCKS)
    with pytest.raises(ValueError, match="больше нет"):
        disconnect_link([], "analysis", "output")


def test_no_duplicates():
    assert connect_or_replace([("input", "preprocess"), ("input", "preprocess")],
                              "input", "preprocess", BLOCKS) == [("input", "preprocess")]
