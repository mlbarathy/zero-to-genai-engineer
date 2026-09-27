"""Graph state types and LLM factory."""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
import operator

from dining_bot.config import OPENAI_MODEL

class RouteDecision(BaseModel):
    intent: Literal[
        "KNOWLEDGE",
        "ANALYTICS",
        "EXTERNAL",
        "ACTION",
        "PLANNING",
        "SMALLTALK",
        "CLARIFY",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class BotState(TypedDict):
    messages: Annotated[list, operator.add]
    intent: str
    confidence: float
    sources: list[str]
    analytics: dict
    chart_figure: dict
    pending_action: dict
    error_category: str
    plan_file: str
    plan_markdown: str
    planning_log: list[str]
    planning_steps: list[dict[str, Any]]


def get_llm() -> ChatOpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in 10_RAG/.env or 13_Project_Implementation/.env")
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def last_user_text(state: BotState) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
            content = m.content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in content
                )
            return str(content)
    return ""
