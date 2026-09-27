"""Intent router — deterministic rules + LLM fallback."""

from __future__ import annotations

import re

from langchain_core.messages import SystemMessage

from dining_bot.config import ROUTER_CONFIDENCE_THRESHOLD
from dining_bot.graph.state import BotState, RouteDecision, get_llm, last_user_text

def _deterministic_route(q: str) -> tuple[str, float] | None:
    """Code-first routing for unambiguous patterns (LLM proposes · code decides)."""
    ql = q.lower().strip()
    if not ql:
        return None

    # Known plan deliverables — always Deep Agents (skills write under /plans/)
    if re.search(r"\b(stock_risk|promo_brief|weekly_plan|month_end_ops_brief)\.md\b", ql):
        return "PLANNING", 0.98

    plan_paths = ("under plans/", "/plans/", "plans/")
    write_verbs = ("write ", "save ", "create ", "generate ", "synthesize into")
    if any(p in ql for p in plan_paths) and (any(w in ql for w in write_verbs) or ".md" in ql):
        return "PLANNING", 0.97

    if any(
        p in ql
        for p in (
            "plan next week",
            "plan the week",
            "plan the restaurant",
            "multi-step plan",
            "closing the month",
            "month-end",
            "month end",
            "do not answer in chat only",
        )
    ):
        return "PLANNING", 0.95

    if "delegate to" in ql and any(
        s in ql for s in ("sales-analyst", "policy-researcher", "weather-scout")
    ):
        return "PLANNING", 0.97

    return None


def router_node(state: BotState) -> dict:
    """LLM proposes {intent, confidence}; code decides whether to clarify (FR-2/4)."""
    q = last_user_text(state)
    forced = _deterministic_route(q)
    if forced:
        intent, conf = forced
        return {
            "intent": intent,
            "confidence": conf,
            "sources": [],
            "analytics": {},
            "chart_figure": {},
            "pending_action": {},
            "error_category": "",
            "plan_file": "",
            "plan_markdown": "",
            "planning_log": [],
            "planning_steps": [],
        }

    llm = get_llm().with_structured_output(RouteDecision)
    history = state["messages"][-10:]
    decision: RouteDecision = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You route ONE restaurant-manager message to exactly one intent.\n"
                    "KNOWLEDGE = answer from policy documents ONLY — no file writes, no SQL.\n"
                    "ANALYTICS = one numbers/SQL question (revenue, top items, stock levels) "
                    "with answer in chat — NOT when user asks to write a .md file under plans/.\n"
                    "EXTERNAL = weather / forecast only (no file write).\n"
                    "ACTION = add a menu item (the only DB write). Also route here when the "
                    "user tries to delete/drop menu data via natural language — the ACTION "
                    "node will refuse.\n"
                    "PLANNING = ANY request to write/save a markdown file under plans/ "
                    "(e.g. stock_risk.md, promo_brief.md, weekly_plan.md), OR multi-step work "
                    "combining sales + stock + weather + policies, OR 'plan the week'.\n"
                    "  Examples that MUST be PLANNING:\n"
                    "  - 'Summarize weekday promos. Write promo_brief.md under plans/'\n"
                    "  - 'Which ingredients are low stock? Write stock_risk.md under plans/'\n"
                    "  - 'Plan next week … write weekly_plan.md'\n"
                    "SMALLTALK = greetings with no business ask.\n"
                    "CLARIFY = ambiguous single ask (not a clear plan request, not an attack).\n"
                    "Never invent SQL or document text here — only route."
                )
            ),
            *history,
        ]
    )
    intent = decision.intent
    conf = float(decision.confidence)
    if intent != "CLARIFY" and conf < ROUTER_CONFIDENCE_THRESHOLD:
        intent = "CLARIFY"
    return {
        "intent": intent,
        "confidence": conf,
        "sources": [],
        "analytics": {},
        "chart_figure": {},
        "pending_action": {},
        "error_category": "",
        "plan_file": "",
        "plan_markdown": "",
        "planning_log": [],
        "planning_steps": [],
    }
