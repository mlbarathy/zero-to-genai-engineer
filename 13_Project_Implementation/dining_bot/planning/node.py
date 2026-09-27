"""LangGraph planning node — streams Deep Agents graph."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from dining_bot.config import PLANS_DIR, PROJECT_ROOT
from dining_bot.graph.state import BotState, last_user_text
from dining_bot.planning.progress import (
    LIVE_PROGRESS_SLOT,
    LIVE_TRACKER,
    PlanningProgressTracker,
    refresh_live_progress,
)

def _newest_plan_file() -> Path | None:
    files = sorted(PLANS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _run_planning_stream(
    planner,
    q: str,
    *,
    on_progress: Any | None = None,
) -> tuple[dict[str, Any], PlanningProgressTracker]:
    """Stream the Deep Agents graph with a detailed progress tracker."""
    cfg = {"configurable": {"thread_id": f"plan-{hash(q) % 10_000_000}"}}
    last_err: Exception | None = None
    for attempt in range(5):
        shared = LIVE_TRACKER.get()
        tracker = shared or PlanningProgressTracker()
        own_tracker = shared is None
        if own_tracker:
            tracker.start("think", "Understanding your request", q[:240])
        else:
            tracker.start("think", "Understanding plan request", q[:240], nested=True)
        try:
            for chunk in planner.stream(
                {"messages": [{"role": "user", "content": q}]},
                cfg,
                stream_mode=["updates", "messages"],
            ):
                tracker.ingest_stream_chunk(chunk)
                if on_progress:
                    on_progress(tracker)
                else:
                    refresh_live_progress(tracker, headline="Working on your plan…")
            tracker.finalize(headline="Plan ready")
            if tracker.steps and tracker.steps[0]["status"] == "running":
                tracker.complete(tracker.steps[0]["id"])
            snap = planner.get_state(cfg)
            return snap.values or {}, tracker
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            if "rate_limit" in msg or "429" in msg:
                time.sleep(2**attempt)
                continue
            tracker.fail_tool(None, "planning", str(e))
            raise
    raise last_err or RuntimeError("planning stream failed")


def planning_node(state: BotState, planner) -> dict:
    """Hand multi-step work to Deep Agents; FR path stays one-intent."""
    q = last_user_text(state)
    tracker = LIVE_TRACKER.get() or PlanningProgressTracker()
    values: dict[str, Any] = {}

    def on_progress(t: PlanningProgressTracker) -> None:
        refresh_live_progress(t, headline="Working on your plan…")

    try:
        values, tracker = _run_planning_stream(planner, q, on_progress=on_progress)
        refresh_live_progress(tracker, headline="Plan complete")
    except Exception:
        try:
            values, tracker = _run_planning_stream(planner, q)
        except Exception as e:
            log = tracker.log_lines() if tracker.steps else []
            return {
                "error_category": "PLANNING_ERROR",
                "messages": [
                    AIMessage(
                        content=(
                            "Planning harness hit an error — try a simpler one-intent question "
                            f"via the normal router, or retry. ({type(e).__name__}: {e})"
                        )
                    )
                ],
                "plan_file": "",
                "plan_markdown": "",
                "planning_log": log,
                "planning_steps": tracker.steps,
            }

    log = tracker.log_lines()
    steps = tracker.steps
    messages = values.get("messages") or []
    answer = ""
    if messages:
        answer = str(getattr(messages[-1], "content", messages[-1])).strip()

    plan_path = _newest_plan_file()
    plan_rel = ""
    plan_body = ""
    if plan_path and plan_path.is_file():
        plan_rel = str(plan_path.relative_to(PROJECT_ROOT))
        plan_body = plan_path.read_text(encoding="utf-8")

    note = ""
    if plan_rel:
        note = f"\n\n_Plan file:_ `{plan_rel}`"
    if not answer:
        answer = "Planning finished." + note
    elif note not in answer:
        answer = answer + note

    return {
        "messages": [AIMessage(content=answer)],
        "plan_file": plan_rel,
        "plan_markdown": plan_body,
        "planning_log": log,
        "planning_steps": steps,
    }
