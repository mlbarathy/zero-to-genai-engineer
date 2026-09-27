"""Streamlit turn runner — live ChatGPT/Claude-style activity timeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dining_bot.planning.progress import (
    LIVE_PROGRESS_SLOT,
    LIVE_TRACKER,
    PlanningProgressTracker,
    refresh_live_progress,
)
from dining_bot.services.bootstrap import brief, interrupt_payload


def run_turn_ui(
    graph,
    thread_id: str,
    payload,
    run_name: str,
    question: str = "",
    *,
    tracker: PlanningProgressTracker | None = None,
    on_progress: Callable[[PlanningProgressTracker], None] | None = None,
):
    """Stream LangGraph with a live progress tracker (all intents, not just PLANNING)."""
    config = {"configurable": {"thread_id": thread_id}, "run_name": run_name}
    tracker = tracker or PlanningProgressTracker()
    tracker.start("think", "Understanding your request", (question or run_name)[:240])

    token = LIVE_TRACKER.set(tracker)
    try:
        trace: list[dict] = []
        chart = None
        plan_file = ""
        plan_markdown = ""
        planning_log: list[str] = []
        intent = ""

        def _tick(headline: str = "Working…") -> None:
            if on_progress:
                on_progress(tracker)
            else:
                refresh_live_progress(tracker, headline=headline)

        _tick("Working…")

        for update in graph.stream(payload, config, stream_mode="updates"):
            for node_name, node_output in update.items():
                if node_name == "__interrupt__":
                    value = node_output[0].value if node_output else {}
                    tracker.ingest_graph_node("PAUSED", value)
                    trace.append({"node": "PAUSED", "data": value})
                    _tick("Waiting for your approval…")
                    continue

                if node_name == "router" and isinstance(node_output, dict):
                    intent = str(node_output.get("intent") or intent)

                tracker.advance_graph(node_name, node_output)
                trace.append({"node": node_name, "data": brief(node_output)})

                if isinstance(node_output, dict):
                    if node_output.get("chart_figure"):
                        chart = node_output["chart_figure"]
                    if node_output.get("plan_file"):
                        plan_file = node_output["plan_file"]
                    if node_output.get("plan_markdown"):
                        plan_markdown = node_output["plan_markdown"]
                    if node_output.get("planning_log"):
                        planning_log = node_output["planning_log"]

                headline = {
                    "PLANNING": "Working on your plan…",
                    "KNOWLEDGE": "Searching policies…",
                    "ANALYTICS": "Running analytics…",
                    "EXTERNAL": "Checking weather…",
                    "ACTION": "Preparing menu change…",
                }.get(intent, "Working…")
                _tick(headline)

        tracker.finalize(headline="Complete")
        _tick("Complete")

        snap = graph.get_state(config)
        pending = interrupt_payload(snap)
        planning_steps = list(tracker.steps)

        if pending is not None:
            return (
                trace,
                None,
                chart,
                pending,
                plan_file,
                plan_markdown,
                planning_log,
                planning_steps,
                intent,
            )

        messages = (snap.values or {}).get("messages") or []
        answer = getattr(messages[-1], "content", str(messages[-1])) if messages else None
        values = snap.values or {}
        if values.get("chart_figure"):
            chart = values["chart_figure"]
        if values.get("plan_markdown"):
            plan_markdown = values["plan_markdown"]
        if values.get("plan_file"):
            plan_file = values["plan_file"]
        if values.get("planning_log"):
            planning_log = values["planning_log"]
        if values.get("planning_steps"):
            planning_steps = values["planning_steps"]
        if values.get("intent"):
            intent = str(values["intent"])

        return (
            trace,
            answer,
            chart,
            None,
            plan_file,
            plan_markdown,
            planning_log,
            planning_steps,
            intent,
        )
    finally:
        LIVE_TRACKER.reset(token)
