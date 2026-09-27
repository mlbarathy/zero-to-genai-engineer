"""Compile the LangGraph orchestrator."""

from __future__ import annotations

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from dining_bot.config import CHECKPOINT_DB
from dining_bot.graph.nodes import (
    action_node,
    analytics_node,
    clarify_node,
    external_node,
    knowledge_node,
    smalltalk_node,
)
from dining_bot.graph.router import router_node
from dining_bot.graph.state import BotState
from dining_bot.mcp.client import DiningMCPClient
from dining_bot.planning.node import planning_node
from dining_bot.rag.pipeline import DiningRAG

def build_graph(rag: DiningRAG, planner, mcp: DiningMCPClient):
    g = StateGraph(BotState)
    g.add_node("router", router_node)
    g.add_node("clarify", clarify_node)
    g.add_node("smalltalk", smalltalk_node)
    g.add_node("knowledge", lambda s: knowledge_node(s, rag))
    g.add_node("analytics", lambda s: analytics_node(s, mcp))
    g.add_node("external", lambda s: external_node(s, mcp))
    g.add_node("action", action_node)
    g.add_node("planning", lambda s: planning_node(s, planner))

    g.add_edge(START, "router")

    def route_after_router(state: BotState) -> str:
        return {
            "KNOWLEDGE": "knowledge",
            "ANALYTICS": "analytics",
            "EXTERNAL": "external",
            "ACTION": "action",
            "PLANNING": "planning",
            "SMALLTALK": "smalltalk",
            "CLARIFY": "clarify"
        }.get(state.get("intent", "CLARIFY"), "clarify")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "knowledge": "knowledge",
            "analytics": "analytics",
            "external": "external",
            "action": "action",
            "planning": "planning",
            "smalltalk": "smalltalk",
            "clarify": "clarify",
        },
    )
    for n in (
        "knowledge",
        "analytics",
        "external",
        "action",
        "planning",
        "smalltalk",
        "clarify",
    ):
        g.add_edge(n, END)

    # Persistent checkpointer (NFR-9) — separate file from business DB.
    import sqlite3

    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    saver = SqliteSaver(conn)
    return g.compile(checkpointer=saver)


def resume_hitl(approved: bool) -> Command:
    return Command(resume={"decisions": [{"type": "approve" if approved else "reject"}]})
