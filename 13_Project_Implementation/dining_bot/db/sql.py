"""SQL validation and read-only analytics execution."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from dining_bot.config import CURRENCY, MAX_ANALYTICS_ROWS, RESTAURANT_ID, RESTAURANT_TIMEZONE
from dining_bot.db.connections import connect_readonly


def validate_readonly_select(sql: str) -> str:
    """Return cleaned SQL or raise ValueError with error category SQL_VALIDATION_ERROR."""
    import sqlglot
    from sqlglot import exp

    text = (sql or "").strip().rstrip(";")
    if not text:
        raise ValueError("SQL_VALIDATION_ERROR: empty SQL")
    if ";" in text:
        raise ValueError("SQL_VALIDATION_ERROR: multiple statements not allowed")
    try:
        parsed = sqlglot.parse(text, read="sqlite")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"SQL_VALIDATION_ERROR: parse failed ({e})") from e
    if len(parsed) != 1 or parsed[0] is None:
        raise ValueError("SQL_VALIDATION_ERROR: expected exactly one statement")
    tree = parsed[0]
    if not isinstance(tree, exp.Select):
        raise ValueError("SQL_VALIDATION_ERROR: only a single SELECT is allowed")
    forbidden = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        exp.Alter,
        exp.Command,
    )
    for node in tree.walk():
        if isinstance(node, forbidden):
            raise ValueError("SQL_VALIDATION_ERROR: write/DDL keyword rejected")
    return text


def run_analytics_sql(sql: str) -> dict[str, Any]:
    """Validate → read-only execute → AnalyticsResult-ish dict + optional ChartSpec."""
    cleaned = validate_readonly_select(sql)
    con = connect_readonly()
    try:
        cur = con.execute(cleaned)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows_raw = cur.fetchmany(MAX_ANALYTICS_ROWS + 1)
    finally:
        con.close()

    truncated = len(rows_raw) > MAX_ANALYTICS_ROWS
    rows_raw = rows_raw[:MAX_ANALYTICS_ROWS]
    rows = [dict(zip(cols, row)) for row in rows_raw]

    # Chart decision from shape (FR-17): time-ish x + numeric y → line; else bar if categorical.
    chart_spec = None
    if len(cols) >= 2 and rows:
        x_field, y_field = cols[0], cols[1]
        sample_x = str(rows[0].get(x_field, ""))
        y_vals = [r.get(y_field) for r in rows if isinstance(r.get(y_field), (int, float))]
        if y_vals:
            if re.match(r"^\d{4}-\d{2}", sample_x) or "week" in x_field.lower() or x_field.lower() in {
                "day",
                "date",
                "hour",
            }:
                chart_type = "line"
            else:
                chart_type = "bar"
            chart_spec = {
                "chart_type": chart_type,
                "title": f"{y_field} by {x_field}",
                "x_field": x_field,
                "y_field": y_field,
            }

    return {
        "metric": cols[1] if len(cols) > 1 else (cols[0] if cols else "result"),
        "unit": CURRENCY,
        "granularity": cols[0] if cols else "row",
        "dimensions": cols[:1],
        "rows": rows,
        "truncated": truncated,
        "sql": cleaned,
        "provenance": {
            "filters": "see SQL",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "restaurant_id": RESTAURANT_ID,
            "timezone": RESTAURANT_TIMEZONE,
        },
        "chart_spec": chart_spec,
    }
