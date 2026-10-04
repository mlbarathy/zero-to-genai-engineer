"""Paths, env vars, and trusted restaurant context (FR-15)."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REPO = PROJECT_ROOT.parent

# Back-compat alias used across modules
HERE = PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "dining_bot.db"
CHECKPOINT_DB = PROJECT_ROOT / "dining_bot_checkpoints.db"
SAMPLE_DOCS = PROJECT_ROOT / "sample_docs"
PLANS_DIR = PROJECT_ROOT / "plans"
AGENT_MD = PROJECT_ROOT / "AGENT.md"
SKILLS_DIR = PROJECT_ROOT / "skills"
WEATHER_CACHE = PROJECT_ROOT / "weather_cache.json"
MCP_ENTRY_SCRIPT = PROJECT_ROOT / "dining_bot.py"
BUILD_DB_SCRIPT = PROJECT_ROOT / "build_db.py"

from dotenv import load_dotenv

for env_path in (
    PROJECT_ROOT / ".env",
    REPO / "11_LangGraph" / ".env",
    REPO / "10_RAG" / ".env",
    REPO / ".env",
):
    load_dotenv(env_path)

RESTAURANT_ID = 1
RESTAURANT_TIMEZONE = "Asia/Dubai"
RESTAURANT_LAT = 25.2048
RESTAURANT_LON = 55.2708
CURRENCY = "AED"
ACTOR_ID = "manager-001"

ROUTER_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.75"))
SQL_QUERY_TIMEOUT_MS = int(os.getenv("SQL_QUERY_TIMEOUT_MS", "5000"))
MAX_ANALYTICS_ROWS = int(os.getenv("MAX_ANALYTICS_ROWS", "1000"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MENU_CATEGORIES = {
    "Starters",
    "Main Course",
    "Breads",
    "Rice",
    "Desserts",
    "Beverages",
}
