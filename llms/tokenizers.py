"""Tokenizer wrapper used by the agent for max_obs_length truncation.

For OpenAI-compatible backends (real OpenAI, Ollama, TSA, SGLang, vLLM, …) the
choice of tokenizer affects *only* how aggressively the observation is
truncated before being inserted into the prompt — it does not change the model
that runs inference. Still, we try to pick the model's native tokenizer when
possible (e.g. Qwen3 for `tree-sparse` / `qwen3vl-dense`) because token-count
drift between webarena's tokenizer and the inference server's tokenizer would
produce subtly different truncated prompts for the same logical observation
and bias TSA-vs-dense comparisons. Fall back to `cl100k_base` for any model
name that is not specifically recognised.

Environment overrides:

* ``WEBARENA_TOKENIZER_PATH`` — path or HF model id to load via
  ``transformers.AutoTokenizer.from_pretrained``. Used when the model name is
  identified as a Qwen3 family member (including the synthetic ids the TSA
  and SGLang launchers advertise). Defaults to ``Qwen/Qwen3-VL-4B-Instruct``.
"""
from __future__ import annotations

import os
from typing import Any

import tiktoken


# Model names we know are served by Qwen3-VL backends in our setup. These are
# the strings the orchestrator passes as ``--model`` to the agent; they're
# also returned by /v1/models on the TSA and SGLang servers.
_QWEN3_MODEL_ALIASES = {
    "tree-sparse",
    "qwen3vl-dense",
    "qwen3vl-judge",
}


def _looks_like_qwen3(model_name: str) -> bool:
    if not model_name:
        return False
    lower = model_name.lower()
    if model_name in _QWEN3_MODEL_ALIASES:
        return True
    if "qwen3" in lower or lower.startswith("qwen/"):
        return True
    if lower.startswith("qwen"):
        # Earlier Qwen lines (qwen2.5:7b-instruct served by Ollama) intentionally
        # fall through to the cl100k_base fallback because tiktoken counts are
        # the historical behaviour and changing it for a non-comparison run
        # would silently shift max_obs_length truncation.
        return False
    return False


def _load_qwen3_tokenizer():
    from transformers import AutoTokenizer

    tokenizer_path = os.environ.get(
        "WEBARENA_TOKENIZER_PATH", "Qwen/Qwen3-VL-4B-Instruct"
    )
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # Qwen3-VL backends (TSA, SGLang dense, SGLang judge) get the real
            # Qwen tokenizer so token counts match what the inference server
            # sees. Important for TSA-vs-dense parity: cl100k_base over-counts
            # Qwen tokens by ~10%, which would truncate observations more
            # aggressively on this side than on the server side.
            if _looks_like_qwen3(model_name):
                try:
                    self.tokenizer = _load_qwen3_tokenizer()
                    return
                except Exception:
                    # If transformers isn't installed in this env, or the
                    # weights aren't present, degrade silently to cl100k_base.
                    # Truncation will be ~10% off but the run still completes.
                    pass

            # tiktoken's encoding_for_model only knows OpenAI's own model
            # names. For everything else (Ollama models like
            # ``qwen2.5:7b-instruct``, ``deepseek-r1:7b``), fall back to
            # ``cl100k_base`` — it is what GPT-3.5/4 use and is a sensible
            # proxy for byte counts when we only need approximate truncation
            # for ``max_obs_length``.
            try:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif provider == "huggingface":
            # lazy-import transformers — it's a heavy dependency and isn't
            # needed when the user is on the OpenAI-compatible backend.
            from transformers import LlamaTokenizer

            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
