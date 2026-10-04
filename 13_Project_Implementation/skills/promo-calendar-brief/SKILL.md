---
name: promo-calendar-brief
description: Summarize weekday promotions and discount rules from policy documents. Load when the manager asks about promos, discounts, or marketing calendar — not revenue SQL.
---

# Promo calendar brief

## Goal

One file: `/plans/promo_brief.md` (under ~30 lines).

## Steps

1. Delegate to the `policy-researcher` subagent (`task` tool) twice:
   - Task A: "Search policies for weekday promotion and discount rules."
   - Task B: "Search policies for promo calendar or marketing schedule."
2. Use the subagent bullets only — do not re-search in the main conversation.
3. Write `/plans/promo_brief.md` with:
   - Weekday promo rules (from policy hits only)
   - Any calendar / timing notes
   - Source document names at the bottom
   - 2 manager actions (no DB writes)
4. Never invent discount percentages not present in policy chunks.
