"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/

Backend selection (controlled by env vars at call time):

* ``OPENAI_API_BASE`` — if set, every OpenAI client call is redirected to this
  base URL. Tested with Ollama / SGLang / TSA OpenAI-compatible endpoints
  reachable via SSH tunnel. If unset, falls back to api.openai.com.
* ``OPENAI_API_KEY`` — required for the real OpenAI API; when
  ``OPENAI_API_BASE`` is set we accept ``"ollama"`` or any non-empty value
  since most OpenAI-compatible endpoints do not validate the key.
* ``WEBARENA_STRIP_REASONING_TAGS`` — if set to "1", strip any ``<think>...
  </think>`` blocks from responses before returning them. Required for
  DeepSeek-R1 / O1-style reasoning models so the WebArena prompt parser sees
  only the final action.
* ``WEBARENA_RESULT_DIR`` — if set, per-call (messages, response) pairs are
  appended as JSONL to ``$WEBARENA_RESULT_DIR/llm_logs.jsonl`` for offline
  debugging. Falls back to ``./llm_logs.jsonl`` when unset.
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

import aiolimiter
import openai
from tqdm.asyncio import tqdm_asyncio

from llms.providers._reasoning import strip_reasoning as _strip_reasoning


def _setup_openai_api() -> None:
    """Validate env vars and pre-populate ``openai`` module attributes.

    Called by every public entrypoint so the per-process env (which may be
    set differently by each multi-worker child) takes effect for that call.

    With the openai 1.x SDK the actual base_url + key are passed to
    ``openai.OpenAI(...)`` at construction time (see ``_make_client``); this
    helper only enforces the env-var contract and keeps the legacy module
    attributes set for any caller still reading them (e.g. tests).
    """
    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    if api_base:
        openai.api_base = api_base
        # Most OpenAI-compatible servers do not validate the key but the
        # openai SDK refuses to construct a client without one. Default to
        # "ollama" so callers running against Ollama/SGLang/TSA need not
        # bother exporting OPENAI_API_KEY.
        openai.api_key = os.environ.get("OPENAI_API_KEY", "") or "ollama"
        openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")
        return
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set when using OpenAI API."
        )
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ.get("OPENAI_ORGANIZATION", "")


def _resolved_api_key() -> str:
    """Effective API key for the openai 1.x client.

    Mirrors ``_setup_openai_api`` precedence: real key if set, else "ollama"
    when only ``OPENAI_API_BASE`` is set, else "dummy" as a final stub so
    the SDK can construct the client.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    if os.environ.get("OPENAI_API_BASE", "").strip():
        return "ollama"
    return "dummy"


def _make_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=_resolved_api_key(),
        base_url=os.environ.get("OPENAI_API_BASE", None),
    )


def _make_async_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=_resolved_api_key(),
        base_url=os.environ.get("OPENAI_API_BASE", None),
    )


_llm_log_file = None  # file handle, opened lazily


def _get_llm_log_file():
    """Return an open file handle for the per-run LLM exchange log (JSONL)."""
    global _llm_log_file
    if _llm_log_file is None:
        result_dir = os.environ.get("WEBARENA_RESULT_DIR", "")
        path = os.path.join(result_dir, "llm_logs.jsonl") if result_dir else "llm_logs.jsonl"
        _llm_log_file = open(path, "a", buffering=1)  # line-buffered
    return _llm_log_file


def _log_exchange(messages: Any, response: str) -> None:
    """Best-effort: persist (messages, response) as one JSONL row."""
    try:
        record = {"messages": messages, "response": response}
        _get_llm_log_file().write(json.dumps(record) + "\n")
    except Exception as e:
        logging.warning(f"Failed to write LLM log: {e}")


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay} seconds.")
                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> Any:
    async with limiter:
        for _ in range(3):
            try:
                client = _make_async_client()
                return await client.completions.create(
                    model=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break

        class _FakeChoice:
            text = ""
        class _FakeResp:
            choices = [_FakeChoice()]
        return _FakeResp()


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    _setup_openai_api()

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [_strip_reasoning(x.choices[0].text) for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    _setup_openai_api()
    client = _make_client()
    response = client.completions.create(
        model=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    answer: str = response.choices[0].text
    return _strip_reasoning(answer)


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> Any:
    async with limiter:
        for _ in range(3):
            try:
                client = _make_async_client()
                return await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break

        class _FakeMessage:
            content = ""
        class _FakeChoice:
            message = _FakeMessage()
        class _FakeResp:
            choices = [_FakeChoice()]
        return _FakeResp()


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    _setup_openai_api()

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [_strip_reasoning(x.choices[0].message.content) for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    _setup_openai_api()
    client = _make_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token] if stop_token else None,
    )
    answer: str = response.choices[0].message.content
    _log_exchange(messages, answer)
    return _strip_reasoning(answer)


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer
