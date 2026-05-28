"""Reasoning-tag stripping for DeepSeek-R1 / O1-style models.

Kept in its own module so unit tests can import it without pulling in
``openai``, ``aiolimiter``, or other heavy runtime deps that
``openai_utils.py`` requires.
"""
from __future__ import annotations

import os
import re

# Regex for <think> ... </think> (and the closely-related <reasoning>),
# multi-line, case-insensitive. Captures optional leading/trailing whitespace
# so removal doesn't leave double newlines.
_THINK_TAG_RE = re.compile(
    r"\s*<(?:think|reasoning)>.*?</(?:think|reasoning)>\s*",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a model response.

    Only active when ``WEBARENA_STRIP_REASONING_TAGS=1`` is set in the env.

    A common failure mode of reasoning models on WebArena is that the final
    action line (``click [N]`` / etc.) is buried after a long reasoning
    preamble — the WebArena parser expects the action to be parseable as the
    model output's tail. Stripping reasoning tags restores that contract.

    If the closing ``</think>`` tag was truncated (model hit max_tokens
    mid-reasoning), we drop everything from the unclosed opener forward so
    reasoning is not mistaken for the action.
    """
    if os.environ.get("WEBARENA_STRIP_REASONING_TAGS", "") != "1":
        return text
    cleaned = _THINK_TAG_RE.sub("\n", text)
    # If a <think> opened but never closed (truncation), drop it forward.
    lower = cleaned.lower()
    if "<think>" in lower:
        idx = lower.rfind("<think>")
        cleaned = cleaned[:idx]
    return cleaned.strip()
