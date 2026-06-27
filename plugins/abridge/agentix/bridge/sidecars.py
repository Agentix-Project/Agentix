"""Experimental presets for externally installed gateway binaries.

These are thin convenience builders: translation and pretokenization live
inside the external sidecar process while abridge forwards decoded JSON
objects without understanding their provider-specific schema.
Pair the returned `Sidecar` with a `Forward` pointed at its URL:

    async with cc_convert_sidecar(binary="cc_convert_sidecar",
                                  upstream_url="https://api.openai.com/v1/chat/completions",
                                  upstream_key="sk-...") as url:
        proxy = Proxy(Forward(url, paths=["/v1/messages"]))
        async with proxy.session(sandbox) as handle:
            await sandbox.remote(agent, base_url=handle.url)

The `cc_convert_sidecar` preset only supplies the executable's expected
environment contract. abridge does not install, build, publish, or otherwise
verify that external executable. If it produces SSE, the current abridge
`Forward` path buffers the complete payload before returning it to the
sandbox.
"""

from __future__ import annotations

from .sidecar import Sidecar


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
    external executable's LiteLLM compatibility mode.
    """
    env = {
        "CC_CONVERT_UPSTREAM_URL": upstream_url,
    }
    if upstream_key:
        env["CC_CONVERT_UPSTREAM_API_KEY"] = upstream_key
    if litellm_compat:
        env["CC_CONVERT_LITELLM_COMPAT"] = "1"

    def sidecar_env(bound_host: str, bound_port: int) -> dict[str, str]:
        return {**env, "CC_CONVERT_LISTEN_ADDR": f"{bound_host}:{bound_port}"}

    return Sidecar(
        command=[binary],
        host=host,
        port=port,
        env=sidecar_env,
        health_path="/healthz",
        ready_timeout=ready_timeout,
    )


__all__ = ["cc_convert_sidecar"]
