from typing import Any

import tiktoken


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
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
