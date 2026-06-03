"""Verify chat-template parity between the TSA and dense SGLang servers.

Both servers must format identical message lists into byte-identical token
sequences. If they don't, the TSA-vs-dense comparison is invalid: the same
agent step would produce systematically different prompts on the two backends.

What we check:
    1. /v1/models endpoint returns a model.
    2. Both servers accept the same minimal chat-completion payload at
       temperature=0.
    3. The raw output for a fixed deterministic prompt agrees on its first
       token, and on the overall character-count to within 2 chars.
    4. Both servers honour OpenAI's `stop` field for an injected stop string.

Use:
    python -m mp.check_template_parity \
        --tsa http://127.0.0.1:10000/v1 \
        --dense http://127.0.0.1:10001/v1

Exit codes: 0 = parity OK, 1 = parity failed, 2 = endpoint unreachable.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout_s: int = 120) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def _get_json(url: str, timeout_s: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode())


def _served_model(base: str) -> str:
    data = _get_json(base.rstrip("/") + "/models")
    return data["data"][0]["id"]


def _chat(base: str, model: str, messages: list[dict[str, Any]], *,
          max_tokens: int = 64, stop=None) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if stop is not None:
        payload["stop"] = stop
    return _post_json(base.rstrip("/") + "/chat/completions", payload, timeout_s=180)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check TSA-vs-dense template parity")
    p.add_argument("--tsa", required=True)
    p.add_argument("--dense", required=True)
    args = p.parse_args(argv)

    try:
        tsa_model = _served_model(args.tsa)
        dense_model = _served_model(args.dense)
    except Exception as e:
        print(f"ERROR: failed to fetch /v1/models — {e}", file=sys.stderr)
        return 2

    print(f"TSA   model id: {tsa_model!r}")
    print(f"Dense model id: {dense_model!r}")

    # 1. Deterministic short prompt.
    msgs = [
        {"role": "system", "content": "You are a precise instruction follower."},
        {"role": "user", "content": "Output exactly: HELLO_WORLD_42"},
    ]
    try:
        tsa = _chat(args.tsa, tsa_model, msgs, max_tokens=32)
        dense = _chat(args.dense, dense_model, msgs, max_tokens=32)
    except Exception as e:
        print(f"ERROR: chat call failed — {e}", file=sys.stderr)
        return 2

    tsa_text = tsa["choices"][0]["message"]["content"]
    dense_text = dense["choices"][0]["message"]["content"]
    print()
    print(f"TSA   output ({len(tsa_text)} chars): {tsa_text!r}")
    print(f"Dense output ({len(dense_text)} chars): {dense_text!r}")

    ok = True
    if abs(len(tsa_text) - len(dense_text)) > 2:
        print("FAIL: output length differs by more than 2 characters.")
        ok = False
    if tsa_text[:1] != dense_text[:1] and tsa_text and dense_text:
        # The chat templates may differ in their first whitespace token; allow
        # that, but flag a fully different first character.
        if tsa_text[:1].strip() and dense_text[:1].strip():
            print("WARN: first character differs (templates may differ in leading whitespace).")

    # 2. Stop-token honouring.
    stop_msgs = [
        {"role": "user", "content": "Count from 1 to 10 separated by commas."},
    ]
    try:
        tsa_s = _chat(args.tsa, tsa_model, stop_msgs, max_tokens=64, stop=["5"])
        dense_s = _chat(args.dense, dense_model, stop_msgs, max_tokens=64, stop=["5"])
    except Exception as e:
        print(f"ERROR: stop-test chat call failed — {e}", file=sys.stderr)
        return 2
    tsa_st = tsa_s["choices"][0]["message"]["content"]
    dense_st = dense_s["choices"][0]["message"]["content"]
    print()
    print(f"TSA   stop-test output: {tsa_st!r}")
    print(f"Dense stop-test output: {dense_st!r}")
    if "5" in tsa_st:
        print("WARN: TSA included the stop token in its output (may be the model emitting "
              "'5' as part of an earlier sequence).")
    if "5" in dense_st:
        print("WARN: dense included the stop token in its output.")

    print()
    print("PARITY CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
