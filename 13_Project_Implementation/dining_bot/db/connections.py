"""SQLite read-only vs write connections."""

from __future__ import annotations

from typing import Any

from dining_bot.config import DB_PATH, SQL_QUERY_TIMEOUT_MS


def connect_readonly() -> Any:
    """SQLite URI mode=ro — writes fail at the connection (FR-13 / NFR-1)."""
    import sqlite3

    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Missing {DB_PATH.name}. Run: python3 build_db.py")
    uri = f"file:{DB_PATH}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=SQL_QUERY_TIMEOUT_MS / 1000)
    con.row_factory = sqlite3.Row
    return con


def connect_write() -> Any:
    """Normal connection — ONLY for Action subgraph after approval."""
    import sqlite3

    con = sqlite3.connect(str(DB_PATH), timeout=10)
    con.execute("PRAGMA foreign_keys = ON;")
    con.row_factory = sqlite3.Row
    return con


def schema_for_llm() -> str:
    """Compact schema card so the model can write SELECTs (not invent tables)."""
    return """
Tables (restaurant_id is ALWAYS 1 — bind it; never take it from the user):
- menu_items(id, restaurant_id, name, description, category, price, is_veg, active, created_at)
- orders(id, restaurant_id, table_no, order_type, status, subtotal, discount, tax, total, created_at)
  status: paid | open | cancelled | refunded
  CANONICAL REVENUE = SUM(orders.total) WHERE status='paid'
- order_items(id, order_id, menu_item_id, qty, unit_price, line_total)
- payments(id, order_id, method, amount, status, paid_at)
- ingredients(id, restaurant_id, name, unit, stock, reorder_level, updated_at)
Timestamps are TEXT 'YYYY-MM-DD HH:MM:SS' (UTC). Use date(created_at) / strftime.
""".strip()
