"""App bootstrap — RAG, MCP, planner, graph."""

from __future__ import annotations

import sys
from typing import Any

import streamlit as st

from dining_bot.config import BUILD_DB_SCRIPT, DB_PATH, PROJECT_ROOT
from dining_bot.graph.build import build_graph
from dining_bot.mcp.client import DiningMCPClient
from dining_bot.planning.agent import build_planning_agent
from dining_bot.rag.pipeline import DiningRAG

HERE = PROJECT_ROOT


def source_code_version() -> float:
    """Max mtime under dining_bot/ — busts Streamlit cache when any .py changes."""
    pkg = HERE / "dining_bot"
    if not pkg.is_dir():
        return 0.0
    return max((p.stat().st_mtime for p in pkg.rglob("*.py")), default=0.0)


def build_app() -> dict[str, Any]:
    """Shared bootstrap for Streamlit + CLI — keeps MCP child processes alive."""
    if not DB_PATH.is_file():
        import subprocess

        subprocess.run([sys.executable, str(BUILD_DB_SCRIPT)], check=True, cwd=PROJECT_ROOT)
    rag = DiningRAG().build()
    mcp = DiningMCPClient().connect()
    planner = build_planning_agent(rag, mcp)
    graph = build_graph(rag, planner, mcp)
    return {"rag": rag, "planner": planner, "graph": graph, "mcp": mcp}


@st.cache_resource(show_spinner="Starting Dining Bot (RAG + MCP servers)…")
def bootstrap(_code_version: float = 0.0):
    """Cached bootstrap; _code_version busts cache when dining_bot.py changes."""
    return build_app()


def interrupt_payload(snap) -> Any | None:
    inter = getattr(snap, "interrupts", None) or ()
    if inter:
        val = getattr(inter[0], "value", inter[0])
        return val
    tasks = getattr(snap, "tasks", None) or ()
    for t in tasks:
        ints = getattr(t, "interrupts", None) or ()
        if ints:
            return getattr(ints[0], "value", ints[0])
    return None


def brief(node_output: Any) -> dict:
    if not isinstance(node_output, dict):
        return {"raw": str(node_output)[:200]}
    out = {
        k: node_output[k]
        for k in ("intent", "confidence", "error_category", "sources", "plan_file")
        if k in node_output
    }
    plog = node_output.get("planning_log") or []
    psteps = node_output.get("planning_steps") or []
    if psteps:
        out["planning_step_count"] = len(psteps)
        out["last_planning_title"] = (psteps[-1].get("title") or "")[:120]
    elif plog:
        out["planning_step_count"] = len(plog)
        out["last_planning_step"] = plog[-1][:120]
    msgs = node_output.get("messages") or []
    if msgs:
        out["last_message"] = str(getattr(msgs[-1], "content", msgs[-1]))[:240]
    return out


def turn_bundle(
    trace,
    answer,
    chart,
    plan_file="",
    plan_markdown="",
    planning_log=None,
    planning_steps=None,
    intent: str = "",
) -> dict:
    return {
        "trace": trace,
        "answer": answer,
        "chart": chart,
        "plan_file": plan_file or "",
        "plan_markdown": plan_markdown or "",
        "planning_log": planning_log or [],
        "planning_steps": planning_steps or [],
        "intent": intent or "",
    }
