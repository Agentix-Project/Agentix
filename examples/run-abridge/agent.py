"""The agent — runs INSIDE the sandbox.

Pristine: no proxy, env, or tracing code. The OpenAI SDK reads
`OPENAI_BASE_URL` (pointed at the in-sandbox abridge proxy by `bridged`)
and `OPENAI_API_KEY` (a placeholder — the host injects the real key), so
this is exactly the code you'd write against the real API. It lives in an
importable module (not the `__main__` script) so `sandbox.remote` can
resolve it by `agent::solve` inside the sandbox.
"""

from __future__ import annotations


def solve(task: str, *, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,  # the host's OpenAICompatibleClient(model=...) can override this upstream
        messages=[
            {"role": "system", "content": "You are a terse, helpful assistant."},
            {"role": "user", "content": task},
        ],
    )
    return resp.choices[0].message.content or ""
