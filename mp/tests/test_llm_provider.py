"""Tests for the reasoning-strip logic and OpenAI-backend-selection logic.

The reasoning-strip helper lives in its own module (``llms.providers._reasoning``)
so these tests don't need ``openai`` / ``aiolimiter`` installed.

The ``_setup_openai_api`` test is only run when ``openai`` is importable;
otherwise it is skipped.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_strip_reasoning():
    """Load llms/providers/_reasoning.py directly without triggering
    llms/__init__.py (which imports text_generation, a heavy runtime dep)."""
    path = REPO_ROOT / "llms" / "providers" / "_reasoning.py"
    spec = importlib.util.spec_from_file_location("_isolated_reasoning", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_isolated_reasoning"] = mod
    spec.loader.exec_module(mod)
    return mod.strip_reasoning


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Each test starts with a clean OPENAI_* env."""
    for k in (
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENAI_ORGANIZATION",
        "WEBARENA_STRIP_REASONING_TAGS",
    ):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Reasoning-strip tests (no openai/aiolimiter dependency)
# ---------------------------------------------------------------------------

def test_strip_reasoning_disabled_by_default() -> None:
    strip_reasoning = _load_strip_reasoning()

    text = "<think>let me think</think>click [60]"
    assert strip_reasoning(text) == text


def test_strip_reasoning_removes_think_block(monkeypatch) -> None:
    strip_reasoning = _load_strip_reasoning()

    monkeypatch.setenv("WEBARENA_STRIP_REASONING_TAGS", "1")
    text = "<think>I should click element 60.</think>click [60]"
    assert strip_reasoning(text) == "click [60]"


def test_strip_reasoning_handles_multiline(monkeypatch) -> None:
    strip_reasoning = _load_strip_reasoning()

    monkeypatch.setenv("WEBARENA_STRIP_REASONING_TAGS", "1")
    text = (
        "<think>\n"
        "Step 1: Look at the page.\n"
        "Step 2: Find the search box.\n"
        "</think>\n"
        "Let's think step-by-step. The action is: click [42]"
    )
    out = strip_reasoning(text)
    assert "Step 1" not in out
    assert "click [42]" in out


def test_strip_reasoning_handles_unclosed_think_tag(monkeypatch) -> None:
    """If max_tokens truncates the response mid-reasoning, drop the
    unclosed tail rather than passing reasoning through as the action."""
    strip_reasoning = _load_strip_reasoning()

    monkeypatch.setenv("WEBARENA_STRIP_REASONING_TAGS", "1")
    text = "preamble\n<think>let me reason about this for a while and then I get truncated"
    out = strip_reasoning(text)
    assert "let me reason" not in out
    assert "<think>" not in out.lower()
    assert "preamble" in out


def test_strip_reasoning_supports_reasoning_tag(monkeypatch) -> None:
    """O1-style models may use <reasoning> instead of <think>."""
    strip_reasoning = _load_strip_reasoning()

    monkeypatch.setenv("WEBARENA_STRIP_REASONING_TAGS", "1")
    text = "<reasoning>analyze the dom</reasoning>click [9]"
    assert strip_reasoning(text) == "click [9]"


# ---------------------------------------------------------------------------
# OpenAI-backend-selection tests (require openai + aiolimiter)
# ---------------------------------------------------------------------------

_HAS_OPENAI = (
    importlib.util.find_spec("openai") is not None
    and importlib.util.find_spec("aiolimiter") is not None
)


def _import_openai_utils_module():
    """Load openai_utils.py as a fresh module."""
    path = REPO_ROOT / "llms" / "providers" / "openai_utils.py"
    spec = importlib.util.spec_from_file_location("_isolated_openai_utils", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_isolated_openai_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai/aiolimiter not installed locally")
def test_setup_openai_api_uses_ollama_when_base_set(monkeypatch) -> None:
    mod = _import_openai_utils_module()
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
    mod._setup_openai_api()
    assert mod.openai.api_base == "http://127.0.0.1:11434/v1"
    assert mod.openai.api_key  # populated to a dummy value, not empty


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai/aiolimiter not installed locally")
def test_setup_openai_api_raises_without_key_or_base(monkeypatch) -> None:
    mod = _import_openai_utils_module()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        mod._setup_openai_api()


@pytest.mark.skipif(not _HAS_OPENAI, reason="openai/aiolimiter not installed locally")
def test_setup_openai_api_uses_real_key_when_no_base(monkeypatch) -> None:
    mod = _import_openai_utils_module()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-here")
    mod._setup_openai_api()
    assert mod.openai.api_key == "sk-real-key-here"
