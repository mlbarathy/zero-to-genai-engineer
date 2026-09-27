"""Small UI helpers."""

from __future__ import annotations

from typing import Any


def normalize_answer(content: Any) -> str | None:
    """Turn LangChain message content into displayable text."""
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        text = " ".join(parts).strip()
        return text or None
    text = str(content).strip()
    return text or None
