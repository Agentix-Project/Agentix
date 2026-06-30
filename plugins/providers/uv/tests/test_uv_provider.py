"""uv provider: launch the runtime from a venv and drive a real remote() call.

Uses `reuse_venv` pointed at the interpreter running the tests (it already has
`agentixx` + uvicorn), so the test needs no uv materialization. Remote targets
are stdlib functions (`math.*`) — always importable by the worker, so the test
exercises the provider's runtime wiring without packaging a fixture module. (A
user's own rollout module is reached the same way every provider does it:
installed into the venv via `UvProviderConfig.project` / `install`.)
"""

from __future__ import annotations

import math
import sys

import pytest
from agentix.provider.uv import UvProvider, UvProviderConfig

from agentix.provider.base import SandboxConfig, SandboxProvider


def _reuse_venv() -> str:
    # venv root of the interpreter running the tests. Use sys.prefix, NOT a
    # resolved sys.executable: the venv's bin/python is a symlink, and resolving
    # it jumps to the base interpreter (whose env lacks agentixx).
    return sys.prefix


def test_config_requires_project_or_venv():
    with pytest.raises(ValueError):
        UvProviderConfig()


def test_is_sandboxprovider():
    provider = UvProvider(UvProviderConfig(reuse_venv=_reuse_venv()))
    assert isinstance(provider, SandboxProvider)


@pytest.mark.asyncio
async def test_remote_roundtrip():
    provider = UvProvider(UvProviderConfig(reuse_venv=_reuse_venv()))
    try:
        async with provider.session(SandboxConfig(image="uv", bundle="uv")) as sandbox:
            assert (await sandbox.health()).version
            assert await sandbox.remote(math.factorial, 5) == 120
            assert await sandbox.remote(math.gcd, 12, 8) == 4
    finally:
        await provider.aclose()


@pytest.mark.asyncio
async def test_get_and_delete():
    provider = UvProvider(UvProviderConfig(reuse_venv=_reuse_venv()))
    try:
        sandbox = await provider.create(SandboxConfig(image="uv", bundle="uv"))
        info = await provider.get(sandbox.sandbox_id)
        assert info.status == "running"
        await sandbox.aclose()
        await provider.delete(sandbox.sandbox_id)
        with pytest.raises(KeyError):
            await provider.get(sandbox.sandbox_id)
    finally:
        await provider.aclose()
