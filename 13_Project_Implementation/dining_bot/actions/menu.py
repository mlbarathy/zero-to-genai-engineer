"""ADD_MENU_ITEM action + HITL write path."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from dining_bot.config import ACTOR_ID, MENU_CATEGORIES, RESTAURANT_ID
from dining_bot.db.connections import connect_write

class AddMenuItemAction(BaseModel):
    action: Literal["ADD_MENU_ITEM"] = "ADD_MENU_ITEM"
    name: str = Field(min_length=2, max_length=80)
    price: float = Field(gt=0, lt=10_000)
    category: str
    description: str = ""
    is_veg: bool = True


def validate_add_menu_action(raw: dict[str, Any]) -> AddMenuItemAction:
    obj = AddMenuItemAction.model_validate(raw)
    if obj.category not in MENU_CATEGORIES:
        raise ValueError(
            f"ACTION_VALIDATION_ERROR: category must be one of {sorted(MENU_CATEGORIES)}"
        )
    return obj


def execute_add_menu_item(action: AddMenuItemAction, approved_by: str) -> dict[str, Any]:
    """ONE transaction: menu insert + audit. Dup guard on name+category (FR-28/29)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    con = connect_write()
    try:
        con.execute("BEGIN")
        existing = con.execute(
            """
            SELECT id FROM menu_items
            WHERE restaurant_id = ? AND lower(name) = lower(?) AND category = ? AND active = 1
            """,
            (RESTAURANT_ID, action.name.strip(), action.category),
        ).fetchone()
        if existing:
            con.execute("ROLLBACK")
            return {
                "ok": False,
                "error_category": "ACTION_VALIDATION_ERROR",
                "message": f"Duplicate menu item already exists (id={existing['id']}).",
            }
        cur = con.execute(
            """
            INSERT INTO menu_items
              (restaurant_id, name, description, category, price, is_veg, active, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                RESTAURANT_ID,
                action.name.strip(),
                action.description or None,
                action.category,
                float(action.price),
                1 if action.is_veg else 0,
                now,
            ),
        )
        new_id = cur.lastrowid
        payload = action.model_dump()
        con.execute(
            """
            INSERT INTO audit_log
              (restaurant_id, actor_id, action, payload, approval_status,
               approved_by, approved_at, executed_at)
            VALUES (?, ?, ?, ?, 'APPROVED', ?, ?, ?)
            """,
            (
                RESTAURANT_ID,
                ACTOR_ID,
                "ADD_MENU_ITEM",
                json.dumps(payload),
                approved_by,
                now,
                now,
            ),
        )
        con.execute("COMMIT")
        return {"ok": True, "menu_item_id": new_id, "payload": payload}
    except Exception as e:  # noqa: BLE001
        con.execute("ROLLBACK")
        return {"ok": False, "error_category": "ACTION_VALIDATION_ERROR", "message": str(e)}
    finally:
        con.close()


def record_rejection(action: AddMenuItemAction) -> None:
    """Audit-only rejection row (no menu write)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    con = connect_write()
    try:
        con.execute(
            """
            INSERT INTO audit_log
              (restaurant_id, actor_id, action, payload, approval_status,
               approved_by, approved_at, executed_at)
            VALUES (?, ?, ?, ?, 'REJECTED', NULL, NULL, NULL)
            """,
            (RESTAURANT_ID, ACTOR_ID, "ADD_MENU_ITEM", json.dumps(action.model_dump())),
        )
        con.commit()
    finally:
        con.close()
