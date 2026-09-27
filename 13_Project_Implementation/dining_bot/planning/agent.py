"""Deep Agents planning harness — tools, subagents, create_deep_agent."""

from __future__ import annotations

import json
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from dining_bot.config import AGENT_MD, HERE, PLANS_DIR, SKILLS_DIR
from dining_bot.db.sql import run_analytics_sql
from dining_bot.graph.state import get_llm
from dining_bot.mcp.client import DiningMCPClient
from dining_bot.rag.pipeline import DiningRAG, format_citations

# create_deep_agent() with the full S12 stack:
#   memory (AGENT.md) · skills (SKILL.md) · tools · subagents (task) ·
#   write_file/edit_file (FilesystemBackend) · checkpointer.
# Still no SQL writes from the planning harness."


def make_planning_tools(rag: DiningRAG, mcp: DiningMCPClient) -> tuple[list, dict[str, Any]]:
    """Tools the planning harness may call — all read-only / file-safe."""

    @tool
    def run_readonly_sql(sql: str) -> str:
        """Run ONE validated SELECT against the restaurant DB. Never INSERT/UPDATE/DELETE.

        Example paid revenue by day (last 7 days):
        SELECT date(created_at) AS day, SUM(total) AS revenue
        FROM orders
        WHERE status = 'paid' AND restaurant_id = 1
          AND date(created_at) >= date('now', '-7 days')
        GROUP BY day ORDER BY day;

        Example low stock:
        SELECT name, stock, reorder_level, unit FROM ingredients
        WHERE restaurant_id = 1 AND stock <= reorder_level ORDER BY stock;
        """
        try:
            result = run_analytics_sql(sql)
        except ValueError as e:
            return json.dumps({"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"ok": False, "error": f"SQL_EXECUTION_ERROR: {e}"})
        return json.dumps(
            {
                "ok": True,
                "sql": result["sql"],
                "rows": result["rows"][:40],
                "provenance": result["provenance"],
            },
            default=str,
        )

    @tool
    def search_policies(query: str) -> str:
        """Semantic search over restaurant policy / SOP documents. Citations are code-built."""
        hits = rag.retrieve(query, k=3)
        if not hits:
            return json.dumps({"ok": False, "hits": [], "message": "RETRIEVAL_EMPTY"})
        return json.dumps(
            {
                "ok": True,
                "hits": [
                    {
                        "name": h["name"],
                        "section": h["section"],
                        "version": h["version"],
                        "source_type": h.get("source_type"),
                        "source_file": h.get("source_file"),
                        "score": h["score"],
                        "chunk": h["chunk"],
                    }
                    for h in hits
                ],
                "citations": format_citations(hits),
            }
        )

    @tool
    def get_weather(days: int = 2) -> str:
        """Forecast for the restaurant lat/lon from app config (not from the user)."""
        return json.dumps(mcp.get_forecast(days=max(1, min(int(days), 7))))

    by_name = {
        "run_readonly_sql": run_readonly_sql,
        "search_policies": search_policies,
        "get_weather": get_weather,
    }
    return list(by_name.values()), by_name


def build_planning_subagents(tools_by_name: dict[str, Any]) -> list[dict[str, Any]]:
    """S12 subagents — each gets a focused tool set and an empty chat via the `task` tool."""
    return [
        {
            "name": "sales-analyst",
            "description": (
                "Read-only SQL specialist for revenue, orders, top items, and stock. "
                "Delegate when you need analytics rows turned into short AED bullet insights. "
                "Include the question and suggested SELECT in the task message."
            ),
            "system_prompt": (
                "You are a restaurant sales analyst for Dining Bot. "
                "Use run_readonly_sql only — never invent numbers. "
                "Return 3–6 bullet points with AED amounts taken from SQL rows. "
                "If SQL fails, one bullet explaining the error. No policy text."
            ),
            "tools": [tools_by_name["run_readonly_sql"]],
        },
        {
            "name": "policy-researcher",
            "description": (
                "Policy/SOP specialist (discounts, hours, refunds, promos, food safety). "
                "Delegate document lookups; returns bullets with source document names."
            ),
            "system_prompt": (
                "You are a restaurant policy researcher. Use search_policies only. "
                "Return 3–5 bullets quoting rules from retrieved chunks. "
                "End with source document names. Never invent policy text or percentages."
            ),
            "tools": [tools_by_name["search_policies"]],
        },
        {
            "name": "weather-scout",
            "description": (
                "Weather specialist for patio/outdoor service decisions. "
                "Delegate forecast summaries; pass days=1–7 in the task message."
            ),
            "system_prompt": (
                "You are a weather scout for the restaurant (Dubai). Use get_weather only. "
                "Summarize max/min °C and rain chance in 2–4 bullets. "
                "Note if the forecast came from cache."
            ),
            "tools": [tools_by_name["get_weather"]],
        },
    ]


def build_planning_agent(rag: DiningRAG, mcp: DiningMCPClient):
    """
    create_deep_agent() = still a LangGraph graph with the full S12 feature set:
    memory, skills, custom tools, subagents (task), file tools, checkpointer.
    """
    PLANS_DIR.mkdir(exist_ok=True)
    if not AGENT_MD.is_file():
        raise FileNotFoundError(f"Missing {AGENT_MD.name} next to dining_bot.py")
    for skill_name in ("weekly-ops-plan", "stock-risk-brief", "promo-calendar-brief"):
        if not (SKILLS_DIR / skill_name / "SKILL.md").is_file():
            raise FileNotFoundError(f"Missing skills/{skill_name}/SKILL.md")

    tools, tools_by_name = make_planning_tools(rag, mcp)
    subagents = build_planning_subagents(tools_by_name)
    backend = FilesystemBackend(root_dir=HERE, virtual_mode=True)
    return create_deep_agent(
        model=get_llm(),
        tools=tools,
        subagents=subagents,
        memory=["/AGENT.md"],
        skills=["/skills/"],
        backend=backend,
        checkpointer=MemorySaver(),
        system_prompt=(
            "You are Dining Bot's planning harness for multi-step manager work.\n"
            "Standing rules live in /AGENT.md (memory). How-to playbooks live in /skills/.\n"
            "For heavy extraction, delegate to subagents via the `task` tool "
            "(sales-analyst, policy-researcher, weather-scout) — each has its own chat.\n"
            "Synthesize subagent bullets into markdown under /plans/ using write_file or edit_file.\n"
            "Never claim you changed the menu or the database. "
            "Menu adds require a separate ACTION message with human approval."
        ),
        name="dining-planning",
    )
