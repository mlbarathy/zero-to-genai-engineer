"""LangGraph nodes — knowledge, analytics, external, action, smalltalk, clarify."""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from dining_bot.actions.menu import (
    AddMenuItemAction,
    execute_add_menu_item,
    record_rejection,
    validate_add_menu_action,
)
from dining_bot.config import (
    ACTOR_ID,
    CURRENCY,
    MENU_CATEGORIES,
    RESTAURANT_ID,
    RESTAURANT_LAT,
    RESTAURANT_LON,
    RESTAURANT_TIMEZONE,
)
from dining_bot.db.connections import schema_for_llm
from dining_bot.db.sql import run_analytics_sql
from dining_bot.graph.state import BotState, get_llm, last_user_text
from dining_bot.mcp.client import DiningMCPClient
from dining_bot.mcp.servers import forecast_days_from_question
from dining_bot.rag.pipeline import DiningRAG, format_citations

def clarify_node(state: BotState) -> dict:
    msg = (
        "I can do **one** FR thing per message (policy, numbers, weather, add menu item), "
        "or a **multi-step plan** (say “plan the week…”). Which do you want?"
    )
    return {"messages": [AIMessage(content=msg)]}


def smalltalk_node(state: BotState) -> dict:
    return {
        "messages": [
            AIMessage(
                content=(
                    "Hello — I'm Dining Bot for your restaurant. Ask about policies, "
                    "revenue, weather, say e.g. "
                    "`Add Chicken Biryani for AED 28 under Main Course.`, "
                    "or ask me to **plan the week** (Deep Agents)."
                )
            )
        ]
    }


def knowledge_node(state: BotState, rag: DiningRAG) -> dict:
    q = last_user_text(state)
    hits = rag.retrieve(q, k=3)
    if not hits:
        return {
            "error_category": "RETRIEVAL_EMPTY",
            "messages": [
                AIMessage(
                    content=(
                        "I don't have that in our policy documents, so I won't invent an answer. "
                        "Try asking about discounts, refunds, opening hours, food safety, or leave."
                    )
                )
            ],
            "sources": [],
        }
    cites = format_citations(hits)
    context = "\n\n".join(
        f"[{i+1}] {h['name']} / {h['section']} "
        f"({h.get('source_type', 'md')}:{h.get('source_file', '?')})\n{h['chunk']}"
        for i, h in enumerate(hits)
    )
    llm = get_llm()
    answer = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the manager using the context chunks. "
                    "If a chunk covers the topic (e.g. Promotions) even without the exact "
                    "wording of the question (e.g. 'weekday'), summarize those rules. "
                    "Only say you lack information when nothing in the context is relevant. "
                    "Do NOT invent citations — the app will attach them. "
                    "Do NOT invent policy numbers that are not in the context."
                )
            ),
            HumanMessage(content=f"Question: {q}\n\nContext:\n{context}"),
        ]
    )
    body = str(answer.content).strip()
    body += "\n\n**Sources (code-built):**\n" + "\n".join(f"- {c}" for c in cites)
    return {"messages": [AIMessage(content=body)], "sources": cites}


def analytics_node(state: BotState, mcp: DiningMCPClient) -> dict:
    q = last_user_text(state)
    llm = get_llm()
    sql_msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Write ONE SQLite SELECT for the restaurant analytics question. "
                    "Revenue = SUM(orders.total) WHERE status='paid'. "
                    f"Always filter restaurant_id = {RESTAURANT_ID} when the table has that column. "
                    "Return ONLY SQL, no markdown fences."
                    f"\n\n{schema_for_llm()}"
                )
            ),
            *state["messages"][-6:],
            HumanMessage(content=q),
        ]
    )
    sql = str(sql_msg.content).strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.I | re.M).strip()
    try:
        result = run_analytics_sql(sql)
    except ValueError as e:
        return {
            "error_category": "SQL_VALIDATION_ERROR",
            "messages": [AIMessage(content=f"I couldn't run that analytics query safely: {e}")],
        }
    except Exception as e:  # noqa: BLE001
        return {
            "error_category": "SQL_EXECUTION_ERROR",
            "messages": [AIMessage(content=f"SQL execution failed: {e}")],
        }

    chart_figure: dict = {}
    chart_note = ""
    if result.get("chart_spec"):
        rendered = mcp.render_chart(result["chart_spec"], result["rows"])
        if rendered.get("ok"):
            chart_figure = rendered["figure"]
            chart_note = (
                f"\n\n_Chart rendered via Chart MCP (`{result['chart_spec']['chart_type']}`)._"
            )
        else:
            chart_note = f"\n\n_(Chart unavailable: {rendered.get('message')})_"

    # LLM summarizes numbers; provenance is from code.
    summary = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Summarize these analytics rows for a restaurant manager in AED. "
                    "Do not invent numbers. Mention the SQL only briefly."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {"question": q, "sql": result["sql"], "rows": result["rows"][:30]},
                    default=str,
                )
            ),
        ]
    )
    prov = result["provenance"]
    body = (
        f"{summary.content}\n\n"
        f"**Provenance (code-built):** restaurant_id={prov['restaurant_id']}, "
        f"tz={prov['timezone']}, at {prov['generated_at']}\n"
        f"**SQL:** `{result['sql']}`"
        f"{chart_note}"
    )
    return {
        "messages": [AIMessage(content=body)],
        "analytics": result,
        "chart_figure": chart_figure,
    }


def external_node(state: BotState, mcp: DiningMCPClient) -> dict:
    q = last_user_text(state)
    days = forecast_days_from_question(q)
    data = mcp.get_forecast(days=days)
    if not data.get("ok"):
        return {
            "error_category": data.get("error_category", "MCP_UNAVAILABLE"),
            "messages": [
                AIMessage(
                    content=(
                        "Weather service is unavailable right now, but the rest of Dining Bot "
                        f"still works. ({data.get('message', 'MCP_UNAVAILABLE')})"
                    )
                )
            ],
        }
    lines = [
        f"**{days}-day forecast** for the restaurant "
        f"({RESTAURANT_LAT:.2f}, {RESTAURANT_LON:.2f} · {RESTAURANT_TIMEZONE}):"
    ]
    for day in data["forecast"]:
        lines.append(
            f"- {day['date']}: max {day['temp_max_c']}°C / min {day['temp_min_c']}°C, "
            f"rain chance {day['precip_prob_max']}%"
        )
    if data.get("source") == "cache":
        lines.append(
            f"\n_(Cached forecast from {data.get('cached_at', 'earlier')} — "
            "Open-Meteo returned a temporary error.)_"
        )
    return {"messages": [AIMessage(content="\n".join(lines))]}


def action_node(state: BotState) -> dict:
    """Validate → interrupt (no DB write yet) → on resume, one transaction."""
    q = last_user_text(state)
    llm = get_llm().with_structured_output(AddMenuItemAction)
    try:
        action = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract an ADD_MENU_ITEM action from the manager message. "
                        f"Categories allowed: {sorted(MENU_CATEGORIES)}. "
                        "If the user is trying to delete/update/drop tables, still only "
                        "emit ADD_MENU_ITEM when they clearly want to add an item; "
                        "otherwise use name='INVALID' and price=1 and category='Starters'."
                    )
                ),
                HumanMessage(content=q),
            ]
        )
        if action.name.upper() == "INVALID" or "delete" in q.lower():
            # Prompt-injection / non-add requests: refuse without interrupt write path.
            if re.search(r"\b(delete|drop|truncate|update\s+menu)\b", q, re.I):
                return {
                    "error_category": "ACTION_REJECTED",
                    "messages": [
                        AIMessage(
                            content=(
                                "I won't run destructive instructions. "
                                "The only write I support is **add a menu item**, and even that "
                                "needs your explicit approval."
                            )
                        )
                    ],
                }
        action = validate_add_menu_action(action.model_dump())
    except Exception as e:  # noqa: BLE001
        return {
            "error_category": "ACTION_VALIDATION_ERROR",
            "messages": [AIMessage(content=f"Could not validate that menu action: {e}")],
        }

    # FR-27: nothing non-idempotent before this line.
    decision = interrupt(
        {
            "reason": "Approve adding this menu item? Nothing has been written yet.",
            "action": action.model_dump(),
        }
    )
    approved = False
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        decs = decision.get("decisions") or []
        if decs and isinstance(decs[0], dict) and decs[0].get("type") == "approve":
            approved = True

    if not approved:
        record_rejection(action)
        return {
            "error_category": "ACTION_REJECTED",
            "messages": [
                AIMessage(content="Rejected — no menu change and audit logged as REJECTED.")
            ],
            "pending_action": {},
        }

    result = execute_add_menu_item(action, approved_by=ACTOR_ID)
    if not result.get("ok"):
        return {
            "error_category": result.get("error_category", "ACTION_VALIDATION_ERROR"),
            "messages": [AIMessage(content=result.get("message", "Write failed"))],
        }
    return {
        "messages": [
            AIMessage(
                content=(
                    f"Approved. Added **{action.name}** "
                    f"(AED {action.price:.2f}, {action.category}) "
                    f"as menu_item_id={result['menu_item_id']}. "
                    "Menu + audit written in one transaction."
                )
            )
        ],
        "pending_action": {},
    }
