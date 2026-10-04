"""Weather and Chart MCP server implementations."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from dining_bot.config import RESTAURANT_LAT, RESTAURANT_LON, RESTAURANT_TIMEZONE, WEATHER_CACHE

# The graph talks to them over MCP stdio — it does not bake chart rendering
# into the Analytics node."


def forecast_days_from_question(text: str) -> int:
    """Map natural language → Open-Meteo forecast_days (1–7)."""
    t = (text or "").lower()
    if any(p in t for p in ("next week", "7 day", "seven day", "whole week", "coming week")):
        return 7
    if any(p in t for p in ("3 day", "three day", "few days")):
        return 3
    if "tomorrow" in t and "week" not in t:
        return 2
    if "week" in t:
        return 7
    return 2


def _load_weather_cache() -> dict[str, Any] | None:
    if not WEATHER_CACHE.is_file():
        return None
    try:
        data = json.loads(WEATHER_CACHE.read_text(encoding="utf-8"))
        return data if data.get("forecast") else None
    except Exception:  # noqa: BLE001
        return None


def _save_weather_cache(data: dict[str, Any]) -> None:
    payload = {
        "ok": True,
        "restaurant": data.get("restaurant"),
        "forecast": data.get("forecast"),
        "cached_at": datetime.now(ZoneInfo(RESTAURANT_TIMEZONE)).isoformat(),
    }
    WEATHER_CACHE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_open_meteo_payload(payload: dict[str, Any]) -> dict[str, Any]:
    daily = payload.get("daily") or {}
    out = []
    for i, day in enumerate(daily.get("time") or []):
        out.append(
            {
                "date": day,
                "temp_max_c": (daily.get("temperature_2m_max") or [None])[i],
                "temp_min_c": (daily.get("temperature_2m_min") or [None])[i],
                "precip_prob_max": (daily.get("precipitation_probability_max") or [None])[i],
            }
        )
    return {
        "ok": True,
        "source": "live",
        "restaurant": {"lat": RESTAURANT_LAT, "lon": RESTAURANT_LON, "tz": RESTAURANT_TIMEZONE},
        "forecast": out,
    }


def get_forecast_impl(days: int = 1) -> dict[str, Any]:
    """Weather capability. Location ALWAYS from restaurant config (not user text)."""
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    days = max(1, min(int(days), 7))
    params = urllib.parse.urlencode(
        {
            "latitude": RESTAURANT_LAT,
            "longitude": RESTAURANT_LON,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": RESTAURANT_TIMEZONE,
            "forecast_days": days,
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    headers = {"User-Agent": "DiningBot-S13/1.0 (GenAI-2026 capstone; educational use)"}

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as resp:
                payload = json.loads(resp.read().decode())
            result = _parse_open_meteo_payload(payload)
            _save_weather_cache(result)
            return result
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in {429, 502, 503, 504} and attempt < 3:
                time.sleep(0.75 * (2**attempt))
                continue
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < 3:
                time.sleep(0.75 * (2**attempt))
                continue
            break

    cached = _load_weather_cache()
    if cached:
        forecast = (cached.get("forecast") or [])[:days]
        return {
            "ok": True,
            "source": "cache",
            "cached_at": cached.get("cached_at"),
            "restaurant": cached.get("restaurant")
            or {"lat": RESTAURANT_LAT, "lon": RESTAURANT_LON, "tz": RESTAURANT_TIMEZONE},
            "forecast": forecast,
            "note": f"Live API unavailable ({last_err}); serving last good forecast.",
        }

    return {
        "ok": False,
        "error_category": "MCP_TOOL_ERROR",
        "message": f"Weather upstream failed: {last_err}",
    }


def render_chart_impl(chart_spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Chart MCP: line|bar only (FR-22). Returns Plotly figure JSON for the UI."""
    import plotly.graph_objects as go

    ctype = (chart_spec or {}).get("chart_type", "bar")
    if ctype not in {"line", "bar"}:
        return {
            "ok": False,
            "error_category": "MCP_TOOL_ERROR",
            "message": "v1 only supports chart_type line|bar",
        }
    x_field = chart_spec["x_field"]
    y_field = chart_spec["y_field"]
    title = chart_spec.get("title") or f"{y_field} by {x_field}"
    xs = [r.get(x_field) for r in rows]
    ys = [r.get(y_field) for r in rows]
    if ctype == "line":
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    else:
        fig = go.Figure(go.Bar(x=xs, y=ys))
    fig.update_layout(title=title, template="plotly_white", height=380)
    return {"ok": True, "figure": json.loads(fig.to_json()), "title": title}


def run_mcp_server(kind: str) -> None:
    """Expose SECTION 4 tools over FastMCP stdio."""
    from mcp.server.fastmcp import FastMCP

    if kind == "weather":
        mcp = FastMCP("dining-weather")

        @mcp.tool()
        def get_forecast(days: int = 1) -> str:
            """Return the restaurant's weather forecast (location from config)."""
            return json.dumps(get_forecast_impl(days))

        mcp.run(transport="stdio")
        return

    if kind == "chart":
        mcp = FastMCP("dining-chart")

        @mcp.tool()
        def render_chart(chart_spec_json: str, rows_json: str) -> str:
            """Render a line/bar chart from ChartSpec + AnalyticsResult rows."""
            return json.dumps(
                render_chart_impl(json.loads(chart_spec_json), json.loads(rows_json))
            )

        mcp.run(transport="stdio")
        return

    raise SystemExit(f"Unknown MCP kind: {kind!r} (use weather|chart)")
