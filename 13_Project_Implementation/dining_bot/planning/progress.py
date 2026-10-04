"""Planning activity timeline tracker."""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any

from dining_bot.ui.timeline import render_planning_timeline_html

class PlanningProgressTracker:
    """Claude/ChatGPT-style activity timeline for graph nodes + Deep Agents streams."""

    _ICON = {
        "context": "📋",
        "think": "💭",
        "route": "🧭",
        "delegate": "🧑‍💼",
        "sql": "🗄️",
        "policy": "📚",
        "weather": "🌤️",
        "file": "📝",
        "skill": "📖",
        "tool": "🔧",
        "chart": "📊",
        "hitl": "✋",
        "done": "✅",
        "error": "⚠️",
    }

    _NODE_LABELS = {
        "router": ("route", "Classifying your request"),
        "knowledge": ("policy", "Searching policy documents"),
        "analytics": ("sql", "Running analytics query"),
        "external": ("weather", "Fetching weather forecast"),
        "action": ("hitl", "Preparing menu change"),
        "planning": ("think", "Building multi-step plan"),
        "smalltalk": ("think", "Composing reply"),
        "clarify": ("think", "Asking for clarification"),
    }

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self._open: dict[str, str] = {}  # tool_call_id → step_id
        self._seen_tool_ids: set[str] = set()
        self._seen_result_ids: set[str] = set()
        self._t0 = time.time()
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"step-{self._seq}"

    def _append(
        self,
        kind: str,
        title: str,
        detail: str = "",
        *,
        status: str = "done",
        tool_id: str | None = None,
        nested: bool = False,
    ) -> str:
        sid = self._next_id()
        step = {
            "id": sid,
            "kind": kind,
            "title": title,
            "detail": detail,
            "status": status,
            "icon": self._ICON.get(kind, "🔧"),
            "elapsed_s": round(time.time() - self._t0, 1),
            "nested": nested,
        }
        self.steps.append(step)
        if tool_id and status == "running":
            self._open[tool_id] = sid
        return sid

    def start(
        self,
        kind: str,
        title: str,
        detail: str = "",
        *,
        tool_id: str | None = None,
        nested: bool = False,
    ) -> str:
        return self._append(
            kind, title, detail, status="running", tool_id=tool_id, nested=nested
        )

    def complete(self, step_id: str, detail: str | None = None) -> None:
        for step in self.steps:
            if step["id"] == step_id:
                step["status"] = "done"
                step["elapsed_s"] = round(time.time() - self._t0, 1)
                if detail:
                    step["detail"] = detail
                return

    def complete_tool(self, tool_id: str | None, tool_name: str, preview: str) -> None:
        preview = (preview or "").strip().replace("\n", " ")[:220]
        sid = self._open.pop(tool_id or "", None)
        if sid is None and tool_name:
            for step in reversed(self.steps):
                if step["status"] == "running" and tool_name in step.get("title", ""):
                    sid = step["id"]
                    break
                if (
                    step["status"] == "running"
                    and tool_name == "task"
                    and step.get("kind") == "delegate"
                ):
                    sid = step["id"]
                    break
        if sid is None and tool_name == "task":
            for step in reversed(self.steps):
                if step.get("kind") == "delegate":
                    if step["status"] == "running":
                        sid = step["id"]
                    else:
                        return
                    break
        if sid:
            self.complete(sid, preview or "Finished.")
            return
        # Already logged via sql/policy/weather/file step — skip orphan "Finished tool" row
        kind_map = {
            "run_readonly_sql": "sql",
            "search_policies": "policy",
            "get_weather": "weather",
            "write_file": "file",
            "edit_file": "file",
            "task": "delegate",
        }
        mapped = kind_map.get(tool_name)
        if mapped:
            for step in reversed(self.steps):
                if step.get("kind") == mapped and step["status"] == "done":
                    return
        self._append("tool", f"Finished `{tool_name}`", preview or "Done.", nested=True)

    def fail_tool(self, tool_id: str | None, tool_name: str, err: str) -> None:
        sid = self._open.pop(tool_id or "", None)
        if sid:
            for step in self.steps:
                if step["id"] == sid:
                    step["status"] = "error"
                    step["detail"] = err[:220]
                    return
        self._append("error", f"`{tool_name}` failed", err[:220], status="error", nested=True)

    def log_lines(self) -> list[str]:
        lines = []
        for s in self.steps:
            mark = {"running": "…", "done": "✓", "error": "✗"}.get(s["status"], "·")
            prefix = "  " if s.get("nested") else ""
            line = f"{prefix}{mark} {s['title']}"
            if s.get("detail"):
                line += f" — {s['detail'][:160]}"
            lines.append(line)
        return lines

    def _graph_detail(self, node_name: str, node_output: Any) -> str:
        if not isinstance(node_output, dict):
            return "Done."
        if node_name == "router":
            intent = node_output.get("intent", "?")
            conf = float(node_output.get("confidence") or 0)
            return f"→ {intent} ({conf:.0%} confidence)"
        if node_name == "knowledge":
            n = len(node_output.get("sources") or [])
            return f"{n} source(s) cited" if n else "Answer generated from documents"
        if node_name == "analytics":
            if node_output.get("chart_figure"):
                return "SQL executed · chart rendered via MCP"
            return "SQL executed · summarizing results"
        if node_name == "external":
            return "Forecast retrieved via Weather MCP"
        if node_name == "planning":
            pf = node_output.get("plan_file") or ""
            return f"Saved `{pf}`" if pf else "Plan synthesized"
        if node_name == "action":
            if node_output.get("pending_action"):
                return "Waiting for your approval"
            return "Action processed"
        if node_output.get("error_category"):
            return str(node_output["error_category"])
        return "Done."

    def advance_graph(self, completed_node: str, node_output: Any) -> None:
        """Mark the active top-level step done; after router, start the routed phase."""
        if completed_node in {"__start__", "__end__", "PAUSED"}:
            return

        detail = self._graph_detail(completed_node, node_output)
        closed = False
        for step in reversed(self.steps):
            if not step.get("nested") and step["status"] == "running":
                self.complete(step["id"], detail)
                closed = True
                break

        if not closed and completed_node != "router":
            kind, title = self._NODE_LABELS.get(completed_node, ("tool", completed_node))
            self._append(kind, title, detail, status="done")

        if completed_node == "router" and isinstance(node_output, dict):
            intent = node_output.get("intent", "")
            nxt = {
                "PLANNING": ("think", "Building multi-step plan", "Deep Agents + subagents"),
                "KNOWLEDGE": ("policy", "Searching policy documents", "RAG over sample_docs/"),
                "ANALYTICS": ("sql", "Running analytics query", "Read-only SQL + optional chart"),
                "EXTERNAL": ("weather", "Fetching weather forecast", "Open-Meteo via MCP"),
                "ACTION": ("hitl", "Preparing menu change", "Validation before approval"),
                "SMALLTALK": ("think", "Composing reply", ""),
                "CLARIFY": ("think", "Asking for clarification", ""),
            }.get(str(intent))
            if nxt:
                kind, title, det = nxt
                self.start(kind, title, det or "In progress…")

    def ingest_graph_node(self, node_name: str, node_output: Any) -> None:
        """Backward-compatible alias for graph stream updates."""
        if node_name == "__interrupt__":
            self.start("hitl", "Waiting for your approval", "No database write yet")
            return
        self.advance_graph(node_name, node_output)

    def ingest_update(self, node_name: str, node_out: Any) -> None:
        if not isinstance(node_out, dict):
            return
        if "SkillsMiddleware" in node_name:
            meta = node_out.get("skills_metadata") or []
            names = ", ".join(m.get("name", "?") for m in meta[:6])
            if self.steps and self.steps[0].get("kind") == "think" and self.steps[0]["status"] == "running":
                self.complete(self.steps[0]["id"], "Request understood")
            self._append("skill", "Loaded on-demand skills", names or "skills/", nested=True)
            return
        if "MemoryMiddleware" in node_name:
            keys = list((node_out.get("memory_contents") or {}).keys())
            self._append(
                "context", "Loaded standing rules", ", ".join(keys) or "/AGENT.md", nested=True
            )
            return
        if node_name in {"model", "agent"} and node_out.get("messages"):
            for msg in node_out["messages"]:
                self._ingest_ai_tool_calls(msg)
        if node_name == "tools" and node_out.get("messages"):
            for msg in node_out["messages"]:
                self._ingest_tool_result(msg)

    def _ingest_ai_tool_calls(self, msg: Any) -> None:
        for tc in getattr(msg, "tool_calls", None) or []:
            if isinstance(tc, dict):
                name = tc.get("name") or ""
                args = tc.get("args") or {}
                tid = tc.get("id")
            else:
                name = getattr(tc, "name", "") or ""
                args = getattr(tc, "args", {}) or {}
                tid = getattr(tc, "id", None)
            if not name:
                continue
            if tid and tid in self._seen_tool_ids:
                continue
            if tid:
                self._seen_tool_ids.add(tid)
            self._register_tool_start(name, args, tid)

    def _register_tool_start(self, name: str, args: dict, tool_id: str | None) -> None:
        if name == "task":
            sub = args.get("subagent_type") or args.get("name") or ""
            desc = (args.get("description") or args.get("prompt") or "")[:240]
            if not sub or sub == "subagent":
                for candidate in ("sales-analyst", "policy-researcher", "weather-scout"):
                    if candidate in desc:
                        sub = candidate
                        break
            if not sub:
                sub = "subagent"
            self.start(
                "delegate",
                f"Delegating to {sub}",
                desc or "Running isolated subagent chat…",
                tool_id=tool_id,
                nested=True,
            )
            return
        if name == "write_file":
            path = args.get("file_path") or args.get("path") or "/plans/…"
            self.start(
                "file", f"Writing {path}", "Saving markdown plan to disk", tool_id=tool_id, nested=True
            )
            return
        if name == "edit_file":
            path = args.get("file_path") or args.get("path") or "plan file"
            self.start(
                "file", f"Editing {path}", "Updating plan sections", tool_id=tool_id, nested=True
            )
            return
        if name == "read_file":
            path = args.get("file_path") or args.get("path") or "file"
            self.start("file", f"Reading {path}", "", tool_id=tool_id, nested=True)
            return
        if name == "run_readonly_sql":
            sql = (args.get("sql") or "")[:200]
            self.start(
                "sql", "Querying restaurant database", sql or "SELECT …", tool_id=tool_id, nested=True
            )
            return
        if name == "search_policies":
            q = (args.get("query") or "")[:200]
            self.start(
                "policy",
                "Searching policy documents",
                q or "semantic retrieval",
                tool_id=tool_id,
                nested=True,
            )
            return
        if name == "get_weather":
            days = args.get("days") or args.get("day") or args.get("num_days") or 3
            self.start(
                "weather",
                f"Fetching {days}-day forecast",
                "Open-Meteo via MCP",
                tool_id=tool_id,
                nested=True,
            )
            return
        self.start("tool", f"Running `{name}`", str(args)[:200], tool_id=tool_id, nested=True)

    def _ingest_tool_result(self, msg: Any) -> None:
        result_id = getattr(msg, "tool_call_id", None)
        if result_id and result_id in self._seen_result_ids:
            return
        if result_id:
            self._seen_result_ids.add(result_id)
        name = getattr(msg, "name", None) or "tool"
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", str(b)) if isinstance(b, dict) else str(b) for b in content
            )
        preview = str(content).strip()
        if str(getattr(msg, "status", "")).lower() == "error":
            self.fail_tool(getattr(msg, "tool_call_id", None), name, preview)
        else:
            self.complete_tool(getattr(msg, "tool_call_id", None), name, preview)

    def ingest_stream_chunk(self, chunk: Any) -> None:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
        else:
            mode, payload = "updates", chunk
        if mode == "updates" and isinstance(payload, dict):
            for node_name, node_out in payload.items():
                if node_name in {"__start__", "__end__"}:
                    continue
                self.ingest_update(node_name, node_out)
        elif mode == "messages" and isinstance(payload, tuple) and payload:
            msg = payload[0]
            self._ingest_ai_tool_calls(msg)
            if getattr(msg, "type", "") == "tool" or msg.__class__.__name__ == "ToolMessage":
                self._ingest_tool_result(msg)

    def finalize(self, *, headline: str = "Complete") -> None:
        for step in self.steps:
            if step["status"] == "running":
                step["status"] = "done"
                step["elapsed_s"] = round(time.time() - self._t0, 1)
        nested_done = sum(1 for s in self.steps if s.get("nested") and s["status"] == "done")
        if self.steps and not any(s.get("kind") == "done" for s in self.steps):
            detail = f"{nested_done} sub-step(s)" if nested_done else "Finished"
            self._append("done", headline, detail)


LIVE_TRACKER: ContextVar[PlanningProgressTracker | None] = ContextVar("LIVE_TRACKER", default=None)
LIVE_PROGRESS_SLOT: ContextVar[Any] = ContextVar("LIVE_PROGRESS_SLOT", default=None)


def refresh_live_progress(
    tracker: PlanningProgressTracker, *, headline: str = "Working…"
) -> None:
    slot = LIVE_PROGRESS_SLOT.get()
    if slot is not None:
        slot.markdown(
            render_planning_timeline_html(tracker.steps, headline=headline),
            unsafe_allow_html=True,
        )
