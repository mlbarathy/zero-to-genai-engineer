"""CLI entry — python dining_bot.py --ask "..."."""

from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.messages import HumanMessage

from dining_bot.services.bootstrap import brief, build_app, interrupt_payload, turn_bundle


def run_turn_cli(graph, thread_id: str, payload, run_name: str):
    """Same orchestration as Streamlit, without session_state (for --ask demos)."""
    config = {"configurable": {"thread_id": thread_id}, "run_name": run_name}
    trace = []
    chart = None
    plan_file = ""
    plan_markdown = ""
    planning_log: list[str] = []
    planning_steps: list[dict[str, Any]] = []
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node_name, node_output in update.items():
            if node_name == "__interrupt__":
                value = node_output[0].value if node_output else {}
                trace.append({"node": "PAUSED", "data": value})
                continue
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
                if node_output.get("planning_steps"):
                    planning_steps = node_output["planning_steps"]
    snap = graph.get_state(config)
    pending = interrupt_payload(snap)
    if pending is not None:
        return trace, None, chart, pending, plan_file, plan_markdown, planning_log, planning_steps
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
    return trace, answer, chart, None, plan_file, plan_markdown, planning_log, planning_steps


def cli_ask(question: str, thread_id: str = "cli-demo") -> None:
    """Classroom / CI: python dining_bot.py --ask 'Show me daily revenue…'"""
    graph = build_app()["graph"]
    print(f"Q: {question}\n")
    trace, answer, _chart, pending, plan_file, plan_md, plog, psteps = run_turn_cli(
        graph, thread_id, {"messages": [HumanMessage(content=question)]}, question[:80]
    )
    print("TRACE:", json.dumps(trace, indent=2, default=str)[:4000])
    if psteps:
        print("\nPLANNING TIMELINE:")
        for s in psteps:
            print(f" [{s.get('status','?')}] {s.get('title','')} — {(s.get('detail') or '')[:100]}")
    elif plog:
        print("\nPLANNING STEPS:")
        for step in plog:
            print(" -", step)
    if pending:
        print("\nHITL PENDING (no write yet):", json.dumps(pending, indent=2, default=str)[:2000])
        return
    print("\nA:", answer)
    if plan_file:
        print(f"\nPLAN FILE: {plan_file}\n")
        print(plan_md[:6000] if plan_md else "(empty)")


def main() -> None:
    """Run: python dining_bot.py --ask 'your question'"""
    if "--ask" not in sys.argv:
        raise SystemExit("Usage: python dining_bot.py --ask 'your question'")
    idx = sys.argv.index("--ask")
    if idx + 1 >= len(sys.argv):
        raise SystemExit("Usage: python dining_bot.py --ask 'your question'")
    cli_ask(sys.argv[idx + 1])


if __name__ == "__main__":
    main()
