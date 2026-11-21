"""Small Streamlit component bridge. Only the canvas is implemented in React.

The editor is built once via npm; no remote scripts are loaded at runtime.
Python retains the canonical graph and all analysis settings.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.components.v1 import declare_component

BUILD = Path(__file__).resolve().parent / "editor_frontend" / "dist"
_component = None


def graph_editor(graph: dict, revision: int, orientation: str, *, key: str = "workflow_graph"):
    global _component
    if not (BUILD / "index.html").is_file():
        raise RuntimeError(
            "Не собран визуальный редактор. Перейдите в workflow_mvp/editor_frontend "
            "и выполните npm install, затем npm run build."
        )
    if _component is None:
        _component = declare_component("diploma_graph_editor", path=str(BUILD))
    return _component(
        graph=graph,
        graphRevision=revision,
        orientation=orientation,
        key=key,
        default=None,
    )

