---
name: stock-risk-brief
description: Quick stock-risk briefing from SQL only. Load when the manager asks about reorder levels, low stock, or ingredient shortages — without a full weekly plan.
---

# Stock risk brief

## Goal

One short file: `/plans/stock_risk.md` (under ~25 lines).

## Steps

1. Delegate to the `sales-analyst` subagent (`task` tool) with:
   - "Run low-stock SQL and summarize risks in bullets."
   - Include this SELECT in the task message:

```sql
SELECT name, stock, reorder_level, unit FROM ingredients
WHERE restaurant_id = 1 AND stock <= reorder_level ORDER BY stock
```

2. Use **only** the subagent's bullets — do not re-run SQL in the main chat unless the subagent failed.
3. Write `/plans/stock_risk.md` with:
   - Title + date
   - Bullets: ingredient, stock vs reorder, unit
   - Exactly 2 recommended manager actions (no DB writes)
4. Do not copy raw JSON or the full SQL result into the file.
