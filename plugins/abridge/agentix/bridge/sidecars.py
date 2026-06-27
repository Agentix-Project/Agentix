"""Presets that wire the bundled gateway sidecars into a `Sidecar`.

These are thin convenience builders — abridge core stays protocol-blind;
all translation/pretokenization lives inside the external sidecar process.
Pair the returned `Sidecar` with a `Forward` pointed at its URL:

    async with cc_convert_sidecar(binary="cc_convert_sidecar",
                                  upstream_url="https://api.openai.com/v1/chat/completions",
                                  upstream_key="sk-...") as url:
        proxy = Proxy(Forward(url, paths=["/v1/messages"]))
        async with proxy.session(sandbox) as handle:
            await sandbox.remote(agent, env=anthropic_env(handle))

`cc_convert_sidecar` runs the `cc_convert_sidecar` Rust binary, which
terminates the Anthropic Messages shape on `/v1/messages`, translates to
OpenAI Chat Completions, forwards to `upstream_url`, and translates the
(streamed) response back to Anthropic — vLLM/SGLang quirks and all.
"""

from __future__ import annotations

from .sidecar import Sidecar, _free_port


def cc_convert_sidecar(
    *,
    upstream_url: str,
    upstream_key: str | None = None,
    binary: str = "cc_convert_sidecar",
    host: str = "127.0.0.1",
    port: int = 0,
    litellm_compat: bool = False,
    ready_timeout: float = 30.0,
) -> Sidecar:
    """Build a `Sidecar` running the cc_convert translation binary.

    `upstream_url` is the OpenAI-compatible `/v1/chat/completions` URL the
    sidecar forwards to; `upstream_key` is the bearer token it sends (kept
    on the host — never in the sandbox). `litellm_compat` switches the
    default `pragmatic` profile to byte-for-byte LiteLLM parity.
    """
    bound = port or _free_port(host)
    env = {
        "CC_CONVERT_LISTEN_ADDR": f"{host}:{bound}",
        "CC_CONVERT_UPSTREAM_URL": upstream_url,
    }
    if upstream_key:
        env["CC_CONVERT_UPSTREAM_API_KEY"] = upstream_key
    if litellm_compat:
        env["CC_CONVERT_LITELLM_COMPAT"] = "1"
    return Sidecar(
        command=[binary],
        host=host,
        port=bound,
        env=env,
        health_path="/healthz",
        ready_timeout=ready_timeout,
    )


__all__ = ["cc_convert_sidecar"]
