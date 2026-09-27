"""HTML activity timeline for planning progress."""

from __future__ import annotations

import html as html_module
from typing import Any


def inject_planning_progress_css() -> None:
    """Backward-compatible alias."""
    from dining_bot.ui.css import inject_app_css

    inject_app_css()


def render_planning_timeline_html(
    steps: list[dict[str, Any]], *, headline: str = "Working on your plan"
) -> str:
    if not steps:
        return ""
    parts = [
        '<div class="db-plan-shell">',
        f'<div class="db-plan-head">{html_module.escape(headline)}</div>',
        '<div class="db-plan-steps">',
    ]
    for step in steps:
        status = step.get("status", "done")
        icon = html_module.escape(step.get("icon") or "🔧")
        title = html_module.escape(step.get("title") or "Step")
        detail = html_module.escape(step.get("detail") or "")
        elapsed = step.get("elapsed_s")
        nested = " nested" if step.get("nested") else ""
        if status == "done":
            badge = "✓"
            badge_class = "db-plan-badge"
        elif status == "error":
            badge = "!"
            badge_class = "db-plan-badge"
        else:
            badge = ""
            badge_class = "db-plan-badge spin"
        meta = f"{elapsed}s" if elapsed is not None else ""
        parts.append(f'<div class="db-plan-step {status}{nested}">')
        parts.append('<div class="db-plan-rail">')
        parts.append(f'<div class="{badge_class}">{badge}</div>')
        parts.append("</div>")
        parts.append('<div class="db-plan-body">')
        parts.append('<div class="db-plan-title-row">')
        parts.append(f'<div class="db-plan-title">{title}</div>')
        if icon and status != "running":
            parts.append(f'<span class="db-plan-chip">{icon}</span>')
        parts.append("</div>")
        if detail:
            parts.append(f'<div class="db-plan-detail">{detail}</div>')
        if meta and status == "done":
            parts.append(f'<div class="db-plan-meta">{meta}</div>')
        parts.append("</div></div>")
    parts.append("</div></div>")
    return "\n".join(parts)
