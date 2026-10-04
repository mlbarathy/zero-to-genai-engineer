"""MCP client — stdio host for weather + chart servers."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from dining_bot.config import MCP_ENTRY_SCRIPT


def _run_async(coro):
    """Run an async MCP call from sync LangGraph nodes / Streamlit."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Jupyter / nested loops — same pattern as S11 orchestrator.
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coro)


def _parse_mcp_tool_json(raw: Any) -> dict[str, Any]:
    """langchain-mcp-adapters returns MCP content blocks; servers return JSON text."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                return json.loads(block["text"])
            text = getattr(block, "text", None)
            if text:
                return json.loads(text)
    raise ValueError(f"Unexpected MCP tool payload: {raw!r}")


class DiningMCPClient:
    """Keep-alive stdio client for dining-weather + dining-chart MCP servers."""

    def __init__(self) -> None:
        self._client: Any = None
        self._tools: dict[str, Any] = {}

    async def _connect(self) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        script = str(MCP_ENTRY_SCRIPT)
        self._client = MultiServerMCPClient(
            {
                "weather": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [script, "--mcp", "weather"],
                },
                "chart": {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [script, "--mcp", "chart"],
                },
            }
        )
        tools = await self._client.get_tools()
        self._tools = {t.name: t for t in tools}
        missing = {"get_forecast", "render_chart"} - set(self._tools)
        if missing:
            raise RuntimeError(f"MCP servers missing tools: {sorted(missing)}")

    def connect(self) -> DiningMCPClient:
        _run_async(self._connect())
        return self

    async def _call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "ok": False,
                "error_category": "MCP_UNAVAILABLE",
                "message": f"Tool {tool_name!r} not on MCP bus.",
            }
        try:
            raw = await tool.ainvoke(args)
            return _parse_mcp_tool_json(raw)
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "error_category": "MCP_TOOL_ERROR",
                "message": str(e),
            }

    def get_forecast(self, days: int = 1) -> dict[str, Any]:
        return _run_async(self._call_tool("get_forecast", {"days": int(days)}))

    def render_chart(
        self, chart_spec: dict[str, Any], rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return _run_async(
            self._call_tool(
                "render_chart",
                {
                    "chart_spec_json": json.dumps(chart_spec),
                    "rows_json": json.dumps(rows, default=str),
                },
            )
        )
