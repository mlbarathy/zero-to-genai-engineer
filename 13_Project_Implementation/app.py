#!/usr/bin/env python3
"""
Dining Bot — Streamlit + CLI entry point.

Do not use `streamlit run dining_bot.py` (shadows the dining_bot/ package).

Run:
  streamlit run app.py
  python app.py --ask "Show me daily revenue for last week"
  python app.py --mcp weather
  python app.py --mcp chart
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI_SCRIPT = HERE / "dining_bot" / "ui" / "streamlit_app.py"


def _mcp_mode() -> str | None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--mcp":
        return sys.argv[2].strip().lower()
    return None


def _run_ui() -> None:
    # Re-exec the UI script every Streamlit rerun (import would be cached → blank page).
    runpy.run_path(str(UI_SCRIPT), run_name="__main__")


_mcp = _mcp_mode()
if _mcp:
    from dining_bot.mcp.servers import run_mcp_server

    run_mcp_server(_mcp)
    raise SystemExit(0)

if __name__ == "__main__" and "--ask" in sys.argv:
    from dining_bot.cli import main

    main()
    raise SystemExit(0)

_run_ui()
