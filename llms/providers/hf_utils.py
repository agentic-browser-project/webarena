"""HuggingFace TGI client wrapper.

The ``text_generation`` package is only needed when the user actually selects
the HF provider. We import it lazily so deployments that only use the OpenAI
backend (and don't install ``text-generation``) can still import this module.
"""


def generate_from_huggingface_completion(
    prompt: str,
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
) -> str:
    from text_generation import Client  # lazy import — see module docstring

    client = Client(model_endpoint, timeout=60)
    generation: str = client.generate(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stop_sequences=stop_sequences,
    ).generated_text

    return generation
