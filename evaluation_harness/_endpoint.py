"""Helpers to pin LLM-judge calls to a fixed inference endpoint.

WebArena's evaluator uses an LLM-as-judge (``llm_fuzzy_match`` and
``llm_ua_match``) for several task types. Both helpers call
``generate_from_openai_chat_completion`` which reads the global
``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` settings. When the same env vars are
also used to point the *agent* at TSA vs SGLang for a backend comparison,
the judge would silently follow whichever backend the agent is using —
conflating agent quality with judge quality.

This module exposes ``judge_endpoint()``, a context manager that swaps in the
``WEBARENA_EVAL_API_BASE`` / ``WEBARENA_EVAL_API_KEY`` env vars (and saves the
existing ones) for the duration of a judge call, then restores them. The
context manager is a no-op when ``WEBARENA_EVAL_API_BASE`` is unset, preserving
the historical behaviour of pointing the judge at whatever the agent uses.

Use:

    from evaluation_harness._endpoint import judge_endpoint
    with judge_endpoint():
        response = generate_from_openai_chat_completion(...)
"""
from __future__ import annotations

import contextlib
import os
from typing import Iterator

import openai


_SENTINEL = object()


@contextlib.contextmanager
def judge_endpoint() -> Iterator[None]:
    """Temporarily point ``openai`` and the env vars at the fixed judge endpoint.

    No-op when ``WEBARENA_EVAL_API_BASE`` is not set in the environment.
    """
    eval_base = os.environ.get("WEBARENA_EVAL_API_BASE", "").strip()
    if not eval_base:
        yield
        return

    eval_key = os.environ.get("WEBARENA_EVAL_API_KEY", "") or "judge"

    saved_env_base = os.environ.get("OPENAI_API_BASE", _SENTINEL)
    saved_env_key = os.environ.get("OPENAI_API_KEY", _SENTINEL)
    saved_client_base = getattr(openai, "api_base", None)
    saved_client_key = getattr(openai, "api_key", None)

    os.environ["OPENAI_API_BASE"] = eval_base
    os.environ["OPENAI_API_KEY"] = eval_key
    try:
        openai.api_base = eval_base
        openai.api_key = eval_key
        yield
    finally:
        # Restore the prior environment exactly (including unset state).
        if saved_env_base is _SENTINEL:
            os.environ.pop("OPENAI_API_BASE", None)
        else:
            os.environ["OPENAI_API_BASE"] = saved_env_base  # type: ignore[assignment]
        if saved_env_key is _SENTINEL:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = saved_env_key  # type: ignore[assignment]
        openai.api_base = saved_client_base
        openai.api_key = saved_client_key
